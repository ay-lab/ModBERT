"""
Create positive (tf peaks) and negative examples as (L,10) probabilistic matrices for each tf bed file. Produces per-tf train/dev/test .npz files. The negative examples are dinucleotide-frequency matched
(based on reference genome) sequences of equal length to the positive sequences that do not overlap any peaks.

Negative selection:
- Each split (train/dev/test) gets its own candidate pool, built by placing seq_len windows every --neg_stride
  bp (default seq_len: non-overlapping tiles) across ALL chromosomes of that split, so a negative may come
  from a different chromosome than its positive. Chosen negatives never overlap each other.
- Windows overlapping any peak (full original peak extents, the centered seq_len windows, and any extra
  --exclude_beds, optionally padded by --neg_peak_pad) are removed, as are windows containing N's.
- Each window and each positive is described by its 16 dinucleotide frequencies (counts / valid dinucleotides).
- Each positive gets one negative, drawn at random (without replacement) from pool windows whose 16
  frequencies are all within +/- --dinuc_tol of the positive's (L-infinity distance). This 1:1 matching
  makes the negative set follow the positive set's dinucleotide-frequency distribution.
- Optional extra limits, applied together with --dinuc_tol:
    --gc_content_tol: max |GC content_pos - GC content_neg| (fraction of G+C bases, e.g. 0.01 = 1 percentage point)
    --cpg_tol:        max |f_CG,pos - f_CG,neg| for the CpG (C followed by G) dinucleotide frequency
                      (e.g. 0.003 = +/-3 CpGs per 1000 dinucleotides); only has an effect if smaller than --dinuc_tol
  A pair counts as "within tolerance" only if it satisfies every active limit.
- Each .npz also stores per-row coordinates aligned with X/y: coord_chrom, coord_start, coord_end
  (0-based, end-exclusive), pair_idx (row of the matched positive/negative, -1 if none), and per-pair
  match quality: dinuc_dist (L-inf distance between the 16 dinucleotide frequencies), gc_content_diff,
  cpg_diff, and within_tol (True if the pair met every active tolerance; False for fallbacks).
- In-tolerance matches are made hardest positives first (fewest similar windows in the pool).
- If no unused window is within tolerance, the nearest unused window is taken, closest pairs first, with the
  search widening until an unused window is found, so windows are never reused unless the pool runs out.
  With --strict_dinuc the positive's negative is dropped instead.
- Finally, whole pairs (positive and negative) are dropped if their CpG frequency differs by more than
  --max_pair_cpg_diff (default 0.01) or their GC content by more than --max_pair_gc_diff (default 0.02).

Channels (indices):
[0] UNK, [1] CLS, [2] SEP, [3] PAD, [4] MASK, [5] A, [6] C, [7] G, [8] T, [9] M

Assumptions:
- Input bed files are standard 0-based BED (pybedtools will parse).
- VCF is bgzipped + indexed (pysam can open it).
- Bedmethyl files are named merged_chr{chrom_no}_cpg.bed and contain percent methylation in column 11 (0-based index 10).
"""

import os
import sys
import argparse
import gzip
import numpy as np
import pandas as pd
from tqdm import tqdm
import pysam
import pybedtools
from collections import defaultdict
from scipy.spatial import cKDTree

# -----------------------------
# Globals
# -----------------------------
VOCAB = {"[UNK]":0,"[CLS]":1,"[SEP]":2,"[PAD]":3,"[MASK]":4,
         "A":5,"C":6,"G":7,"T":8,"M":9}

BASE_TO_IDX = {"A":5, "C":6, "G":7, "T":8}
A_IDX, C_IDX, G_IDX, T_IDX, M_IDX = 5,6,7,8,9
UNK_IDX = 0

# Dinucleotide encoding: code = 4*first + second with A,C,G,T = 0..3 (AA=0, AC=1, ..., TT=15);
# any dinucleotide containing a non-ACGT base gets INVALID_DINUC.
DINUCS = [a + b for a in "ACGT" for b in "ACGT"]
N_DINUC = 16
INVALID_DINUC = 16
_BASE_LUT = np.full(256, 4, dtype=np.uint8)
for _i, _b in enumerate("ACGT"):
    _BASE_LUT[ord(_b)] = _i
    _BASE_LUT[ord(_b.lower())] = _i

# -----------------------------
# Utility Functions
# -----------------------------

def load_bedmethyl_for_chrom(chrom, bedmethyl_dir, fname_template="merged_{chrom}_cpg.bed"):
    """
    Load methylation file for a given chrom into a dict {(chrom, pos0): methyl_fraction}
    Assumes bed file with start at column 1 (0-based start) and percent methylation at column 11 (index 10).
    Fname template receives {chrom} (e.g., 'chr1' or '1' depending on how files are named). Default template
    assumes 'merged_chr1_cpg.bed', so pass chrom='chr1' -> 'merged_chr1_cpg.bed'.
    """
    # Matching user note: files named merged_chr{i}_cpg.bed
    # If files are like merged_chr1_cpg.bed and chrom is 'chr1' we use that directly.
    fname = fname_template.format(chrom=chrom)
    path = os.path.join(bedmethyl_dir, fname)
    if not os.path.exists(path):
        # try without 'chr' prefix (chrom might be 'chr1' but file names use '1')
        chrom_no_prefix = chrom.replace("chr", "")
        alt_fname = fname_template.format(chrom=chrom_no_prefix)
        alt_path = os.path.join(bedmethyl_dir, alt_fname)
        if os.path.exists(alt_path):
            path = alt_path
        else:
            print(f"[WARN] Missing methylation file for {chrom}: looked for {path} and {alt_path}")
            return {}

    # Read as plain TSV; no header
    # Typical BEDMethyl format: chrom, start, end, ..., percent_methylation at column index 10 (0-based)
    meth = {}
    # Use pandas to read columns lazily
    try:
        df = pd.read_csv(path, sep="\t", header=None, comment='#', dtype={0:str})
    except Exception as e:
        print(f"[WARN] Failed to read methylation file {path}: {e}")
        return {}

    # Defensive: if df has fewer than 11 columns, try to infer percent from last column
    percent_col = 10 if df.shape[1] > 10 else (df.shape[1] - 1)
    for _, row in df.iterrows():
        chrom_name = row[0]
        start_pos = int(row[1])   # bed start, 0-based
        try:
            percent_meth = float(row[percent_col])
        except Exception:
            # fallback: try last column
            percent_meth = float(row.iloc[-1])
        meth[(chrom_name, start_pos)] = min(max(percent_meth / 100.0, 0.0), 1.0)
    return meth

def overlay_vcf_on_seq(seq, chrom, seq_start, vcf_handle):
    """
    seq: reference string (already uppercased)
    seq_start: 0-based genomic coordinate corresponding to seq[0]
    vcf_handle: an open pysam.VariantFile
    Returns arr (L,4) with probabilities for A,C,G,T per position.
    Notes:
      - We convert VCF rec.pos (1-based) to 0-based before comparing with seq_start.
      - Only SNVs with single-nucleotide alts are handled. Multiallelic and indels are ignored.
    """
    L = len(seq)
    arr = np.zeros((L, 4), dtype=np.float32)
    base_to_idx_local = {"A":0,"C":1,"G":2,"T":3}
    # Build variants dict keyed by 0-based genomic pos -> dict(base->prob)
    variants = {}
    # fetch uses 0-based start, end (end exclusive) — pysam accepts these as ints
    seq_end = seq_start + L
    try:
        for rec in vcf_handle.fetch(chrom, seq_start, seq_end):
            # only handle biallelic SNVs
            if rec.alts is None:
                continue
            if len(rec.ref) == 1 and all(len(a) == 1 for a in rec.alts):
                gt = None
                try:
                    # if multiple samples, we'll just take first sample's GT
                    gt = rec.samples[0].get('GT', None)
                except Exception:
                    # if no samples or different schema, skip
                    gt = None
                if gt is None:
                    continue
                # convert rec.pos (1-based) to 0-based
                pos0 = rec.pos - 1
                if pos0 < seq_start or pos0 >= seq_end:
                    continue
                alleles = [rec.ref] + list(rec.alts)
                # handle phased/unphased ploidy 2 assumption
                a0 = gt[0]; a1 = gt[1] if len(gt) > 1 else gt[0]
                if a0 is None or a1 is None:
                    continue
                probs = {}
                if a0 != a1:
                    # heterozygous
                    probs[alleles[a0]] = probs.get(alleles[a0], 0.0) + 0.5
                    probs[alleles[a1]] = probs.get(alleles[a1], 0.0) + 0.5
                else:
                    probs[alleles[a0]] = probs.get(alleles[a0], 0.0) + 1.0
                variants[pos0] = probs
    except ValueError:
        # chrom not present in VCF header or region invalid -> leave variants empty
        pass

    # Fill array
    for i, base in enumerate(seq):
        pos = seq_start + i
        if pos in variants:
            prob_dict = variants[pos]
            for b, p in prob_dict.items():
                if b in base_to_idx_local:
                    arr[i, base_to_idx_local[b]] = p
                else:
                    # if allele is N or ambiguous, distribute uniformly
                    arr[i, :] = 0.25
        else:
            if base in base_to_idx_local:
                arr[i, base_to_idx_local[base]] = 1.0
            else:
                arr[i, :] = 0.25  # ambiguous base -> uniform
    return arr

def integrate_methylation(arr_acgt, seq, chrom, seq_start, meth_dict):
    """
    arr_acgt: (L,4) for A,C,G,T probabilities (sum to 1 per row if confident)
    seq: reference sequence string (upper)
    seq_start: 0-based genomic coordinate for seq[0]
    meth_dict: {(chrom, pos0): methyl_fraction}
    Returns (L,10) array with channels:
    [UNK, CLS, SEP, PAD, MASK, A, C, G, T, M]
    Notes:
      - M channel takes fraction of C probability according to methyl fraction at that position.
      - If base is not A/C/G/T, UNK channel is set to 1.0 for that row.
    """
    L = len(seq)
    out = np.zeros((L, 10), dtype=np.float32)
    for i, base in enumerate(seq):
        if base not in "ACGT":
            out[i, UNK_IDX] = 1.0
            continue
        probs_acgt = arr_acgt[i].astype(np.float32).copy()  # [A,C,G,T]
        pos0 = seq_start + i
        # methylation applies only to C channel (index 1 within probs_acgt)
        if (chrom, pos0) in meth_dict:
            m = meth_dict[(chrom, pos0)]
            # move fraction m * C -> M channel
            c_prob = probs_acgt[1]
            m_amount = m * c_prob
            probs_acgt[1] = (1.0 - m) * c_prob
            # normalize if sum > 0
            s = probs_acgt.sum() + m_amount
            if s > 0:
                # set A,C,G,T in final channels
                out[i, 5:9] = probs_acgt / s
                out[i, M_IDX] = m_amount / s
            else:
                # fallback
                out[i, 5:9] = probs_acgt
                out[i, M_IDX] = m_amount
        else:
            s = probs_acgt.sum()
            if s > 0:
                out[i, 5:9] = probs_acgt / s
            else:
                out[i, 5:9] = 0.25
    return out

# -----------------------------
# Core per-tf processing
# -----------------------------
def make_centered_intervals_from_bed(bed_path, seq_len, valid_chroms):
    """
    Return list of pybedtools.Interval objects centered to seq_len.
    """
    bed = pybedtools.BedTool(bed_path)
    centered = []
    half = seq_len // 2
    for p in bed:
        if p.chrom not in valid_chroms:
            continue
        center = int((p.start + p.end) // 2)
        start = center - half
        end = center + half + 1  # exclusive end so length == seq_len
        if start < 0:
            continue
        centered.append(pybedtools.Interval(p.chrom, start, end))
    return centered

def chrom_sort_key(chrom):
    c = chrom.replace("chr", "")
    return (0, int(c), "") if c.isdigit() else (1, 0, c)

# -----------------------------
# Dinucleotide utilities
# -----------------------------
def seq_to_dinuc_codes(seq):
    """Return uint8 array of length len(seq)-1 with dinucleotide codes 0..15, or INVALID_DINUC (16)."""
    b = _BASE_LUT[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]
    if len(b) < 2:
        return np.zeros(0, dtype=np.uint8)
    codes = (b[:-1] * 4 + b[1:]).astype(np.uint8)
    codes[(b[:-1] > 3) | (b[1:] > 3)] = INVALID_DINUC
    return codes

def dinuc_freq(seq):
    """16-dim dinucleotide frequency vector (normalized over valid dinucleotides)."""
    counts = np.bincount(seq_to_dinuc_codes(seq), minlength=N_DINUC + 1)[:N_DINUC].astype(np.float64)
    tot = counts.sum()
    if tot == 0:
        return np.zeros(N_DINUC, dtype=np.float32)
    return (counts / tot).astype(np.float32)

_GC_BASES = np.array([0, 1, 1, 0], dtype=bool)  # A, C, G, T -> is C or G
_FIRST_IS_GC = _GC_BASES[np.arange(N_DINUC) // 4]
_SECOND_IS_GC = _GC_BASES[np.arange(N_DINUC) % 4]
CG_IDX = DINUCS.index("CG")

def gc_content_from_dinuc_freqs(freqs):
    """
    GC content (fraction of G+C bases) from 16 dinucleotide frequencies: the average of the GC fraction of
    first and second bases. For an N-free 101-bp window this differs from base-level GC by < 0.001.
    """
    freqs = np.atleast_2d(freqs)
    return 0.5 * (freqs[:, _FIRST_IS_GC].sum(1) + freqs[:, _SECOND_IS_GC].sum(1))

def matching_features(freqs, dinuc_tol, gc_content_tol=None, cpg_tol=None):
    """
    Features for the L-infinity nearest-neighbour search, rescaled so that a single threshold (dinuc_tol)
    enforces every limit at once:
      - the CG column is multiplied by dinuc_tol / cpg_tol (|diff| <= dinuc_tol on it <=> |dCpG| <= cpg_tol);
      - GC content is appended, multiplied by dinuc_tol / gc_content_tol.
    """
    F = np.array(freqs, dtype=np.float32, copy=True)
    if cpg_tol is not None and cpg_tol < dinuc_tol:
        F[:, CG_IDX] *= dinuc_tol / cpg_tol
    if gc_content_tol is not None:
        gc = gc_content_from_dinuc_freqs(freqs).astype(np.float32)
        F = np.hstack([F, (gc * (dinuc_tol / gc_content_tol))[:, None]])
    return F

def window_dinuc_freqs(codes, starts, seq_len, chunk=8192):
    """
    Vectorized dinucleotide frequencies for many windows [s, s+seq_len) of one chromosome.
    codes: output of seq_to_dinuc_codes on the whole chromosome.
    Returns (freqs (W,16) float32, invalid_frac (W,) float32).
    """
    n_d = seq_len - 1
    offs = np.arange(n_d, dtype=np.int64)
    W = len(starts)
    freqs = np.empty((W, N_DINUC), dtype=np.float32)
    invalid_frac = np.empty(W, dtype=np.float32)
    for c0 in range(0, W, chunk):
        s = starts[c0:c0 + chunk]
        w = len(s)
        sub = codes[s[:, None] + offs[None, :]]                       # (w, n_d) uint8
        flat = (np.arange(w, dtype=np.int64)[:, None] * (N_DINUC + 1) + sub).ravel()
        counts = np.bincount(flat, minlength=w * (N_DINUC + 1)).reshape(w, N_DINUC + 1)
        valid = counts[:, :N_DINUC].sum(axis=1)
        freqs[c0:c0 + w] = counts[:, :N_DINUC] / np.maximum(valid, 1)[:, None]
        invalid_frac[c0:c0 + w] = counts[:, N_DINUC] / n_d
    return freqs, invalid_frac

# -----------------------------
# Peak exclusion + candidate windows
# -----------------------------
def _open_text_maybe_gzip(path):
    """Open plain or gzipped text, detected from the file's first bytes rather than its extension."""
    with open(path, "rb") as fh:
        magic = fh.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rt")
    return open(path, "rt")

def collect_peak_intervals(bed_paths, valid_chroms, excl=None):
    """
    Add raw (uncentered) intervals from BED files (plain or gzipped) to excl {chrom: [(start, end), ...]}.
    Skips blank, '#', 'track' and 'browser' lines; accepts tab- or space-separated columns; uses only the
    first three columns. Any other line without a valid chrom/start/end raises an error naming the file and
    line, instead of silently dropping intervals that negatives are supposed to avoid.
    """
    if excl is None:
        excl = defaultdict(list)
    for path in bed_paths:
        n_used = 0
        with _open_text_maybe_gzip(path) as fh:
            for lineno, line in enumerate(fh, 1):
                stripped = line.strip()
                if not stripped or stripped.startswith(("#", "track", "browser")):
                    continue
                fields = stripped.split()
                try:
                    chrom, start, end = fields[0], int(fields[1]), int(fields[2])
                except (IndexError, ValueError):
                    hint = ""
                    if stripped.lower().startswith(("<!doctype", "<html", "<")):
                        hint = (" This looks like an HTML page, not a BED file; if you downloaded it from GitHub, "
                                "use the 'Raw' file link.")
                    raise ValueError(f"{path}, line {lineno}: expected 'chrom start end', got "
                                     f"{stripped[:120]!r}.{hint}") from None
                if chrom in valid_chroms:
                    excl[chrom].append((start, end))
                    n_used += 1
        print(f"  exclusion file {os.path.basename(path)}: {n_used} intervals on the selected chromosomes")
        if n_used == 0:
            print(f"  [WARN] no usable intervals in {path}; check its chromosome names (expected e.g. 'chr1')")
    return excl

def merge_exclusions(excl, pad=0):
    """Sort + merge exclusion intervals (optionally padded). Returns {chrom: (n,2) int64 array}."""
    merged = {}
    for chrom, ivs in excl.items():
        ivs = sorted((max(0, s - pad), e + pad) for s, e in ivs)
        out = []
        for s, e in ivs:
            if out and s <= out[-1][1]:
                out[-1][1] = max(out[-1][1], e)
            else:
                out.append([s, e])
        merged[chrom] = np.array(out, dtype=np.int64).reshape(-1, 2)
    return merged

def overlaps_exclusion(merged_chrom, start, end):
    """True if half-open [start, end) overlaps any merged (sorted, disjoint) exclusion interval."""
    if merged_chrom is None or len(merged_chrom) == 0:
        return False
    i = np.searchsorted(merged_chrom[:, 0], end, side="left")
    return i > 0 and merged_chrom[i - 1, 1] > start

def build_valid_windows(chrom_len, merged_chrom, seq_len, stride=None):
    """
    Start coordinates (ascending) of seq_len windows lying entirely in gaps between exclusions, one every
    `stride` bp (default seq_len, i.e. non-overlapping tiles; a smaller stride gives overlapping candidates).
    """
    stride = seq_len if stride is None else stride
    starts = []
    prev_end = 0
    if merged_chrom is None:
        merged_chrom = np.zeros((0, 2), dtype=np.int64)
    for s, e in merged_chrom:
        if s - prev_end >= seq_len:
            starts.append(np.arange(prev_end, s - seq_len + 1, stride, dtype=np.int64))
        prev_end = max(prev_end, int(e))
    if chrom_len - prev_end >= seq_len:
        starts.append(np.arange(prev_end, chrom_len - seq_len + 1, stride, dtype=np.int64))
    return np.concatenate(starts) if starts else np.zeros(0, dtype=np.int64)

def build_negative_pool(fasta, split_chroms, merged_excl, seq_len, max_n_frac, stride=None):
    """
    Candidate negative windows across all chromosomes of a split.
    Returns (chrom_names, chrom_idx (P,), starts (P,), freqs (P,16)).
    """
    chrom_names, idx_list, start_list, freq_list = [], [], [], []
    fasta_refs = set(fasta.references)
    for chrom in sorted(split_chroms, key=chrom_sort_key):
        if chrom not in fasta_refs:
            print(f"    [WARN] {chrom} not in FASTA; skipping for negative pool")
            continue
        chrom_len = fasta.get_reference_length(chrom)
        starts = build_valid_windows(chrom_len, merged_excl.get(chrom), seq_len, stride)
        if len(starts) == 0:
            continue
        codes = seq_to_dinuc_codes(fasta.fetch(chrom))
        freqs, inv = window_dinuc_freqs(codes, starts, seq_len)
        del codes
        keep = inv <= max_n_frac
        if not keep.any():
            continue
        # Number of candidate windows per chromosome (after removing peaks, excluded regions and N's)
        print(f"[DEBUG] {chrom}: {int(keep.sum())} valid windows")
        ci = len(chrom_names)
        chrom_names.append(chrom)
        idx_list.append(np.full(int(keep.sum()), ci, dtype=np.int32))
        start_list.append(starts[keep])
        freq_list.append(freqs[keep])
    if not chrom_names:
        return [], np.zeros(0, np.int32), np.zeros(0, np.int64), np.zeros((0, N_DINUC), np.float32)
    return chrom_names, np.concatenate(idx_list), np.concatenate(start_list), np.concatenate(freq_list)

# -----------------------------
# Dinucleotide matching
# -----------------------------
def _kdtree_query(tree, X, k, bound):
    kwargs = dict(k=k, p=np.inf, distance_upper_bound=bound)
    try:
        d, i = tree.query(X, workers=-1, **kwargs)
    except TypeError:  # scipy < 1.6
        d, i = tree.query(X, n_jobs=-1, **kwargs)
    if k == 1:
        d, i = d[:, None], i[:, None]
    return d, i

class _PoolClaims:
    """
    Tracks which pool windows can no longer be chosen. With overlapping candidate windows (pool_keys given),
    claiming a window also blocks every pool window that overlaps it (same chromosome, starts less than
    seq_len apart), so chosen negatives never overlap each other.
    pool_keys: chrom_index * 2**40 + start for each pool window, sorted ascending.
    """
    def __init__(self, n_pool, pool_keys=None, seq_len=None):
        self.used = np.zeros(n_pool, dtype=bool)
        self.keys, self.seq_len = pool_keys, seq_len
        if pool_keys is not None:
            assert np.all(np.diff(pool_keys) > 0), "pool must be sorted by chromosome, then start"

    def claim(self, a):
        if self.keys is None:
            self.used[a] = True
            return
        k = self.keys[a]
        lo = np.searchsorted(self.keys, k - self.seq_len + 1, side="left")
        hi = np.searchsorted(self.keys, k + self.seq_len, side="left")
        self.used[lo:hi] = True

def _assign_rows(rows, d, nn, bound, claims, rng, assign, dist):
    """
    For positives `rows` (in the given order), assign a random unused neighbour with distance <= bound from
    their neighbour lists (d, nn). Returns the positives that got nothing.
    """
    still = []
    for r, j in enumerate(rows):
        m = d[r] <= bound                          # missing neighbours have d = inf and are masked out here
        cand, cd = nn[r][m], d[r][m]
        free = ~claims.used[cand]
        cand, cd = cand[free], cd[free]
        if len(cand) == 0:
            still.append(j)
            continue
        c = rng.integers(len(cand))
        claims.claim(cand[c])
        assign[j], dist[j] = cand[c], cd[c]
    return np.array(still, dtype=np.int64)

def _assign_round(tree, feats, rows, kk, bound, claims, rng, assign, dist, max_entries=16_000_000):
    """Query kk neighbours for `rows` in chunks (bounding memory to ~max_entries neighbours) and assign."""
    still = []
    bs = max(1, max_entries // kk)
    for c0 in range(0, len(rows), bs):
        chunk = rows[c0:c0 + bs]
        d, nn = _kdtree_query(tree, feats[chunk], kk, bound)
        still.append(_assign_rows(chunk, d, nn, bound, claims, rng, assign, dist))
    return np.concatenate(still) if still else np.zeros(0, dtype=np.int64)

def _assign_closest_first(tree, feats, rows, kk, claims, assign, dist, max_entries=16_000_000):
    """
    Fallback assignment that gives scarce windows to the positives closest to them: every unmatched positive
    proposes its nearest unused window among its kk nearest, each proposed window goes to its closest
    proposer, and the others propose again; repeated until every positive is matched or has run out of
    unused windows among its kk nearest (those are returned). Close pairs are thereby formed before distant
    ones, which maximizes how many pairs stay within the pair-dropping cutoffs.
    """
    left = []
    bs = max(1, max_entries // kk)
    for c0 in range(0, len(rows), bs):
        chunk = np.asarray(rows[c0:c0 + bs])
        d, nn = _kdtree_query(tree, feats[chunk], kk, np.inf)
        active = np.arange(len(chunk))
        while len(active):
            free = ~claims.used[nn[active]]
            has = free.any(1)
            if not has.all():
                left.append(chunk[active[~has]])
                active = active[has]
                free = free[has]
                if not len(active):
                    break
            c = free.argmax(1)
            prop_w = nn[active, c]
            prop_d = d[active, c]
            o = np.lexsort((prop_d, prop_w))                       # by window, then distance
            first = np.ones(len(o), dtype=bool)
            first[1:] = prop_w[o][1:] != prop_w[o][:-1]
            win = o[first]                                         # closest proposer for each window
            if claims.keys is None:
                claims.used[prop_w[win]] = True
            else:
                # Overlapping candidates: winning windows may overlap each other. Accept them closest first,
                # skipping any already blocked by an accepted one (those positives propose again).
                win = win[np.argsort(prop_d[win], kind="stable")]
                acc = []
                for w in win:
                    if not claims.used[prop_w[w]]:
                        claims.claim(prop_w[w])
                        acc.append(w)
                win = np.asarray(acc, dtype=np.int64)
            assign[chunk[active[win]]] = prop_w[win]
            dist[chunk[active[win]]] = prop_d[win]
            lost = np.ones(len(active), dtype=bool)
            lost[win] = False
            active = active[lost]
    return np.concatenate(left) if left else np.zeros(0, dtype=np.int64)

def match_negatives(pos_feats, pool_feats, tol, k, rng, strict=False, pool_keys=None, seq_len=None):
    """
    For each positive, choose a distinct pool window whose features (from matching_features) are all within
    +/- tol (L-infinity distance), chosen at random among in-tolerance candidates.

    Passes:
      1. random unused window within tol among the k nearest;
      2. same among the 16*k nearest;
      3. fallback (unless strict): positives still unmatched get the nearest unused window, assigned
         closest pairs first (see _assign_closest_first), widening the search x4 per round until every
         positive has one, so a window is reused only if every window in the pool is already taken.
    Passes 1-2 handle positives hardest first: difficulty is the distance to the k-th nearest pool window
    (large = few similar windows, e.g. very CpG-rich peaks), ties broken at random. The fallback instead forms
    the closest pairs first, so scarce windows are not spent on pairs too different to keep; in simulations
    this minimized the pairs later dropped by --max_pair_cpg_diff / --max_pair_gc_diff.
    pool_keys / seq_len: give these when pool windows overlap; choosing a window then also blocks the windows
    overlapping it, so no two chosen negatives overlap (see _PoolClaims).
    Returns (assign (n_pos,) pool index or -1, dist (n_pos,) L-inf distance in feature space, stats dict).
    """
    n_pos, n_pool = len(pos_feats), len(pool_feats)
    assign = np.full(n_pos, -1, dtype=np.int64)
    dist = np.full(n_pos, np.inf)
    stats = dict(within_tol=0, fallback=0, reused=0, dropped=0)
    if n_pos == 0 or n_pool == 0:
        stats["dropped"] = n_pos
        return assign, dist, stats

    tree = cKDTree(pool_feats)
    claims = _PoolClaims(n_pool, pool_keys, seq_len)
    tol_eps = tol + 1e-7

    # Pass 1 (also gives the difficulty used for ordering)
    k1 = min(k, n_pool)
    d0, nn0 = _kdtree_query(tree, pos_feats, k1, np.inf)
    order = np.lexsort((rng.random(n_pos), -d0[:, -1]))   # descending k-th neighbour distance
    pending = _assign_rows(order, d0[order], nn0[order], tol_eps, claims, rng, assign, dist)
    del d0, nn0

    # Pass 2
    k2 = min(k * 16, n_pool)
    if len(pending) and k2 > k1:
        pending = _assign_round(tree, pos_feats, pending, k2, tol_eps, claims, rng, assign, dist)

    if len(pending) and strict:
        stats["dropped"] = len(pending)
    elif len(pending):
        # Pass 3: closest pairs first, widening the search until every positive has an unused window
        kk = k2
        while len(pending):
            pending = _assign_closest_first(tree, pos_feats, pending, kk, claims, assign, dist)
            if kk == n_pool:
                break
            kk = min(kk * 4, n_pool)
        # Pool exhausted (every window used or blocked): reuse each remaining positive's nearest window
        if len(pending):
            d, nn = _kdtree_query(tree, pos_feats[pending], 1, np.inf)
            assign[pending], dist[pending] = nn[:, 0], d[:, 0]
            stats["reused"] = len(pending)

    matched = assign >= 0
    stats["within_tol"] = int((matched & (dist <= tol_eps)).sum())
    stats["fallback"] = int(matched.sum()) - stats["within_tol"]
    return assign, dist, stats

# -----------------------------
# Core per-split processing
# -----------------------------
def make_coords(chroms, starts, seq_len, pair_idx, dinuc_dist, gc_content_diff, cpg_diff, within_tol):
    """
    Per-row coordinates saved alongside X/y (row i of every array describes row i of X).
      coord_chrom (str), coord_start / coord_end (0-based, end-exclusive, int64),
      pair_idx (int64): for a negative, the row of the positive it was matched to; for a positive, the row of
                        its matched negative (-1 if its negative was dropped),
      dinuc_dist (float32): L-inf distance between the pair's 16 dinucleotide frequencies,
      gc_content_diff (float32): |GC content difference| of the pair,
      cpg_diff (float32): |CpG (CG dinucleotide) frequency difference| of the pair,
      within_tol (bool): True if the pair met every active tolerance (False for fallbacks / unpaired).
    The three distances are NaN for an unpaired positive.
    """
    starts = np.asarray(starts, dtype=np.int64)
    return {
        "coord_chrom": np.asarray(chroms, dtype="U32"),
        "coord_start": starts,
        "coord_end": starts + seq_len,
        "pair_idx": np.asarray(pair_idx, dtype=np.int64),
        "dinuc_dist": np.asarray(dinuc_dist, dtype=np.float32),
        "gc_content_diff": np.asarray(gc_content_diff, dtype=np.float32),
        "cpg_diff": np.asarray(cpg_diff, dtype=np.float32),
        "within_tol": np.asarray(within_tol, dtype=bool),
    }

def empty_coords(seq_len):
    return make_coords([], [], seq_len, [], [], [], [], [])

def build_examples_for_intervals(intervals, fasta, vcf, meth_dir, seq_len, split_chroms, merged_excl,
                                 rng, dinuc_tol=0.01, dinuc_k=64, strict_dinuc=False, max_n_frac=0.0,
                                 gc_content_tol=None, cpg_tol=None, max_pair_cpg_diff=None,
                                 max_pair_gc_diff=None, neg_stride=None):
    empty = (np.zeros((0, seq_len, 10), dtype=np.float32), np.zeros((0,), dtype=np.int8),
             empty_coords(seq_len))

    # 1) Positive reference sequences + dinucleotide frequencies
    pos_records = []  # (chrom, start, seq)
    for iv in intervals:
        chrom, start, end = iv.chrom, int(iv.start), int(iv.end)
        try:
            pos_seq = fasta.fetch(chrom, start, end).upper()
        except Exception:
            continue
        if len(pos_seq) != seq_len:
            continue
        pos_records.append((chrom, start, pos_seq))
    if len(pos_records) == 0:
        return empty
    pos_freqs = np.stack([dinuc_freq(s) for _, _, s in pos_records])

    # 2) Candidate negatives from every chromosome of this split, excluding all peaks
    stride = seq_len if neg_stride is None else neg_stride
    pool_chroms, pool_ci, pool_starts, pool_freqs = build_negative_pool(
        fasta, split_chroms, merged_excl, seq_len, max_n_frac, stride)
    print(f"    negative pool: {len(pool_starts)} windows (stride {stride} bp) across {len(pool_chroms)} "
          f"chromosomes for {len(pos_records)} positives")
    pool_keys = (pool_ci.astype(np.int64) << 40) + pool_starts if stride < seq_len else None
    if len(pool_starts) < len(pos_records):
        print("    [WARN] pool smaller than number of positives; some windows will be reused")

    # 3) Dinucleotide-frequency (+ optional GC content / CpG) matching
    assign, dist, stats = match_negatives(
        matching_features(pos_freqs, dinuc_tol, gc_content_tol, cpg_tol),
        matching_features(pool_freqs, dinuc_tol, gc_content_tol, cpg_tol),
        dinuc_tol, dinuc_k, rng, strict_dinuc, pool_keys=pool_keys, seq_len=seq_len)
    ok = assign >= 0
    within = ok & (dist <= dinuc_tol + 1e-6)

    # Real (unscaled) per-pair differences, for logging, pair dropping and saving
    n_pos = len(pos_records)
    pair_dinuc = np.full(n_pos, np.nan, dtype=np.float32)
    pair_gc = np.full(n_pos, np.nan, dtype=np.float32)
    pair_cpg = np.full(n_pos, np.nan, dtype=np.float32)
    if ok.any():
        pf, nf = pos_freqs[ok], pool_freqs[assign[ok]]
        pair_dinuc[ok] = np.abs(pf - nf).max(1)
        pair_gc[ok] = np.abs(gc_content_from_dinuc_freqs(pf) - gc_content_from_dinuc_freqs(nf))
        pair_cpg[ok] = np.abs(pf[:, CG_IDX] - nf[:, CG_IDX])

    limits = f"dinuc_tol {dinuc_tol}"
    if gc_content_tol is not None:
        limits += f", gc_content_tol {gc_content_tol}"
    if cpg_tol is not None:
        limits += f", cpg_tol {cpg_tol}"
    print(f"    matching ({limits}): {stats['within_tol']} within tol, "
          f"{stats['fallback']} nearest-available fallback, {stats['dropped']} dropped, "
          f"{stats['reused']} reused windows")

    # Drop whole pairs (positive AND its negative) whose CpG or GC content differ too much, so the remaining
    # classes cannot be separated by CpG density / GC content beyond these cutoffs.
    eps = 1e-6  # float32 rounding: a difference of exactly the cutoff is kept
    drop = np.zeros(n_pos, dtype=bool)
    if max_pair_cpg_diff is not None:
        drop |= ok & (pair_cpg > max_pair_cpg_diff + eps)
    if max_pair_gc_diff is not None:
        drop |= ok & (pair_gc > max_pair_gc_diff + eps)
    if drop.any():
        print(f"    dropped {int(drop.sum())} pairs ({100 * drop.mean():.2f}% of positives) with "
              f"|dCpG freq| > {max_pair_cpg_diff} or |dGC content| > {max_pair_gc_diff} "
              f"({int((drop & ~within).sum())} of them fallbacks)")
        keep = ~drop
        pos_records = [r for r, kp in zip(pos_records, keep) if kp]
        pos_freqs, assign, dist, ok, within = pos_freqs[keep], assign[keep], dist[keep], ok[keep], within[keep]
        pair_dinuc, pair_gc, pair_cpg = pair_dinuc[keep], pair_gc[keep], pair_cpg[keep]
        n_pos = len(pos_records)
        if n_pos == 0:
            return empty

    if ok.any():
        q = lambda a: f"median {np.median(a[ok]):.4f} / max {np.max(a[ok]):.4f}"
        neg_freqs = pool_freqs[assign[ok]]
        mean_gap = np.abs(pos_freqs[ok].mean(0) - neg_freqs.mean(0))
        print(f"    per kept pair: dinuc L-inf {q(pair_dinuc)}; |dGC content| {q(pair_gc)}; |dCpG freq| {q(pair_cpg)}")
        print(f"    max |mean pos - mean neg| dinuc freq = {mean_gap.max():.5f} ({DINUCS[int(mean_gap.argmax())]})")

    # Sanity check: no negative overlaps any excluded interval
    neg_records = []  # aligned with kept positives order
    neg_pos_row = []  # row of the positive each negative was matched to
    for j in np.flatnonzero(ok):
        a = assign[j]
        chrom, nstart = pool_chroms[pool_ci[a]], int(pool_starts[a])
        assert not overlaps_exclusion(merged_excl.get(chrom), nstart, nstart + seq_len), \
            f"negative {chrom}:{nstart} overlaps a peak"
        neg_records.append((chrom, nstart))
        neg_pos_row.append(int(j))
    if stats["reused"] == 0 and neg_records:
        srt = sorted(neg_records, key=lambda r: (chrom_sort_key(r[0]), r[1]))
        for (c1, s1), (c2, s2) in zip(srt, srt[1:]):
            assert c1 != c2 or s2 - s1 >= seq_len, f"negatives overlap: {c1}:{s1} and {c2}:{s2}"
    if len(neg_records) == 0:
        return empty

    # 4) Probabilistic matrices, grouped by chromosome so each methylation file is loaded once
    n_neg = len(neg_records)
    X = np.zeros((n_pos + n_neg, seq_len, 10), dtype=np.float32)
    jobs = defaultdict(list)
    for j, (chrom, start, seq) in enumerate(pos_records):
        jobs[chrom].append((j, start, seq))
    for r, (chrom, start) in enumerate(neg_records):
        jobs[chrom].append((n_pos + r, start, None))

    for chrom in sorted(jobs, key=chrom_sort_key):
        meth_dict = load_bedmethyl_for_chrom(chrom, meth_dir)
        for row, start, seq in jobs[chrom]:
            if seq is None:
                seq = fasta.fetch(chrom, start, start + seq_len).upper()
            arr_acgt = overlay_vcf_on_seq(seq, chrom, start, vcf)
            X[row] = integrate_methylation(arr_acgt, seq, chrom, start, meth_dict)
        del meth_dict

    y = np.concatenate([np.ones(n_pos, dtype=np.int8), np.zeros(n_neg, dtype=np.int8)])

    pos_pair = np.full(n_pos, -1, dtype=np.int64)
    pos_pair[neg_pos_row] = n_pos + np.arange(n_neg)
    both = lambda a: np.concatenate([a, a[neg_pos_row]])  # positives' values, then each negative's pair value
    coords = make_coords(
        [c for c, _, _ in pos_records] + [c for c, _ in neg_records],
        [s for _, s, _ in pos_records] + [s for _, s in neg_records],
        seq_len,
        np.concatenate([pos_pair, np.asarray(neg_pos_row, dtype=np.int64)]),
        both(pair_dinuc), both(pair_gc), both(pair_cpg), both(within),
    )
    return X, y, coords

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Create tf train/dev/test sets from peak BEDs with methylation+VCF overlay")
    parser.add_argument("--fasta", required=True, help="Reference genome FASTA (indexed)")
    parser.add_argument("--vcf", required=True, help="VCF (bgzipped + indexed) with variants for sample (GM12878)")
    parser.add_argument("--bedmethyl_dir", required=True, help="Directory containing merged_chr{i}_cpg.bed files")
    parser.add_argument("--bed_list", required=True, nargs="+", help="One or more tf peak bed.gz files (paths). Output will be created for each independently.")
    parser.add_argument("--seq_len", type=int, default=101, help="Sequence (and matrix) length (odd is recommended). Default 101.")
    parser.add_argument("--test_chroms", nargs="+", default=["chr5", "chr10", "chr14", "chr18", "chr20", "chr22"],
                        help="Chromosomes for test split")
    parser.add_argument("--dev_chroms", nargs="+", default=["chr6", "chr21"],
                        help="Chromosomes for validation (dev) split")
    parser.add_argument("--out_dir", required=True, help="Output directory for .npz files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for negatives generation")
    parser.add_argument("--dinuc_tol", type=float, default=0.01,
                        help="Max allowed |freq_pos - freq_neg| for every one of the 16 dinucleotide frequencies "
                             "(L-infinity). 0.01 ~ +/-10 dinucleotides out of 1000. Default 0.01.")
    parser.add_argument("--gc_content_tol", type=float, default=None,
                        help="Optional max |GC content difference| (fraction of G+C bases) within a pair, "
                             "e.g. 0.01 = 1 percentage point. Default: not applied.")
    parser.add_argument("--cpg_tol", type=float, default=None,
                        help="Optional max |CpG (CG dinucleotide) frequency difference| within a pair, "
                             "e.g. 0.003 = +/-3 CpGs per 1000 dinucleotides. Only tightens the CG limit if "
                             "smaller than --dinuc_tol. Default: not applied.")
    parser.add_argument("--max_pair_cpg_diff", type=float, default=0.01,
                        help="Drop the positive AND its negative when their CpG (CG dinucleotide) frequencies "
                             "differ by more than this (0.01 = 10 CpGs per 1000 bp). Default 0.01; "
                             "pass inf to keep all pairs.")
    parser.add_argument("--max_pair_gc_diff", type=float, default=0.02,
                        help="Drop the positive AND its negative when their GC contents differ by more than "
                             "this (0.02 = 2 percentage points). Default 0.02; pass inf to keep all pairs.")
    parser.add_argument("--dinuc_k", type=int, default=64,
                        help="Nearest neighbours examined per positive (random pick among those within tolerance).")
    parser.add_argument("--strict_dinuc", action="store_true",
                        help="Drop the negative for a positive with no unused window within --dinuc_tol "
                             "(default: take the nearest unused window instead).")
    parser.add_argument("--max_neg_n_frac", type=float, default=0.0,
                        help="Max fraction of dinucleotides containing N in a negative window. Default 0 (no N's).")
    parser.add_argument("--neg_stride", type=int, default=None,
                        help="Spacing (bp) between candidate negative windows. Default: seq_len (non-overlapping "
                             "tiles). A smaller value gives ~seq_len/stride times more, overlapping "
                             "candidates, which helps match scarce CpG-rich positives; chosen negatives still "
                             "never overlap each other. Uses more memory and time.")
    parser.add_argument("--neg_peak_pad", type=int, default=0,
                        help="Extra bp padding added around peaks when excluding them from negatives.")
    parser.add_argument("--exclude_beds", nargs="*", default=[],
                        help="Extra BED(.gz) files whose intervals negatives must also avoid "
                             "(e.g., other marks' peaks or the ENCODE blacklist).")
    args = parser.parse_args()
    if args.neg_stride is not None and args.neg_stride < 1:
        parser.error("--neg_stride must be >= 1")
    for name in ("dinuc_tol", "gc_content_tol", "cpg_tol", "max_pair_cpg_diff", "max_pair_gc_diff"):
        v = getattr(args, name)
        if v is not None and v <= 0:
            parser.error(f"--{name} must be > 0")
    if args.cpg_tol is not None and args.cpg_tol >= args.dinuc_tol:
        print(f"[WARN] --cpg_tol {args.cpg_tol} >= --dinuc_tol {args.dinuc_tol}: CG is already limited by "
              f"--dinuc_tol, so --cpg_tol has no effect")

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    valid_chroms = {f"chr{i}" for i in range(1, 23)}  # autosomes only
    test_chroms = set(args.test_chroms)
    dev_chroms = set(args.dev_chroms)
    train_chroms = valid_chroms - test_chroms - dev_chroms

    os.makedirs(args.out_dir, exist_ok=True)

    # Open fasta and vcf once
    fasta = pysam.FastaFile(args.fasta)
    vcf = pysam.VariantFile(args.vcf)

    for bed_path in args.bed_list:
        tfname = os.path.basename(bed_path).replace(".bed.gz", "").replace(".bed", "")
        # print(f"[INFO] Processing tf: {tfname} from {bed_path}")

        # Build intervals, split by chrom sets
        all_intervals = make_centered_intervals_from_bed(bed_path, args.seq_len,
                                                        train_chroms.union(dev_chroms).union(test_chroms))
        # Partition intervals by splits
        train_intervals = [iv for iv in all_intervals if iv.chrom in train_chroms]
        dev_intervals = [iv for iv in all_intervals if iv.chrom in dev_chroms]
        test_intervals = [iv for iv in all_intervals if iv.chrom in test_chroms]

        print(f"  intervals: total={len(all_intervals)} train={len(train_intervals)} dev={len(dev_intervals)} test={len(test_intervals)}")

        # Everything negatives must avoid: full original peaks, centered peak windows, and any extra BEDs
        all_chroms = train_chroms | dev_chroms | test_chroms
        excl = collect_peak_intervals([bed_path] + list(args.exclude_beds), all_chroms)
        for iv in all_intervals:
            excl[iv.chrom].append((int(iv.start), int(iv.end)))
        merged_excl = merge_exclusions(excl, pad=args.neg_peak_pad)

        # Build examples and save
        for split_name, ivs, split_chroms in [("train", train_intervals, train_chroms),
                                              ("dev", dev_intervals, dev_chroms),
                                              ("test", test_intervals, test_chroms)]:
            out_path = os.path.join(args.out_dir, f"{split_name}.npz")
            if len(ivs) == 0:
                print(f"  [WARN] No intervals for {split_name}, writing empty arrays to {out_path}")
                np.savez_compressed(out_path, X=np.zeros((0, args.seq_len, 10), dtype=np.float32), y=np.zeros((0,), dtype=np.int8),
                                    **empty_coords(args.seq_len))
                continue
            print(f"  Building {split_name} examples ({len(ivs)} intervals) ...")
            X, y, coords = build_examples_for_intervals(
                ivs, fasta, vcf, args.bedmethyl_dir, args.seq_len, split_chroms, merged_excl, rng,
                dinuc_tol=args.dinuc_tol, dinuc_k=args.dinuc_k, strict_dinuc=args.strict_dinuc,
                max_n_frac=args.max_neg_n_frac, gc_content_tol=args.gc_content_tol, cpg_tol=args.cpg_tol,
                max_pair_cpg_diff=args.max_pair_cpg_diff, max_pair_gc_diff=args.max_pair_gc_diff,
                neg_stride=args.neg_stride)
            print(f"  -> Saving {X.shape[0]} examples to {out_path}")
            np.savez_compressed(out_path, X=X, y=y, **coords)
    fasta.close()
    vcf.close()
    print("[✓] Done!")

if __name__ == "__main__":
    main()