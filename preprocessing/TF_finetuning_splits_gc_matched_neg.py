"""
Create positive (tf peaks) and negative examples as (L,10) probabilistic matrices for each tf bed file. Produces per-tf train/dev/test .npz files with equal number of positive and negative sequences in each split. The negative examples are GC-content matched (based on reference genome) sequences of equal length to the positive sequences that:
  - do not overlap the full extent of any peak in the tf bed file (nor any centered positive window),
  - do not overlap any blacklisted region (e.g., ENCODE hg38-blacklist.v2),
  - contain no N (or other non-ACGT) bases in the reference,
  - are sampled without replacement (no genomic window is used as a negative more than once),
  - have a GC fraction within --max_gc_diff of their positive. If no unused window is within
    that tolerance, the positive is dropped too, so every kept positive has a matched negative.

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

# -----------------------------
# Globals
# -----------------------------
VOCAB = {"[UNK]":0,"[CLS]":1,"[SEP]":2,"[PAD]":3,"[MASK]":4,
         "A":5,"C":6,"G":7,"T":8,"M":9}

BASE_TO_IDX = {"A":5, "C":6, "G":7, "T":8}
A_IDX, C_IDX, G_IDX, T_IDX, M_IDX = 5,6,7,8,9
UNK_IDX = 0

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

def gc_fraction(seq):
    seq = seq.upper()
    if len(seq) == 0:
        return 0.0
    gc = sum(1 for b in seq if b in ("G", "C"))
    atgc = sum(1 for b in seq if b in ("A", "C", "G", "T"))
    return gc / max(atgc, 1)

def load_bed_intervals(bed_path, valid_chroms=None):
    """
    Read a BED / BED.gz file into {chrom: [(start, end), ...]} (0-based, end-exclusive).
    Used for the raw tf peaks and for the blacklist.
    """
    out = defaultdict(list)
    for f in pybedtools.BedTool(bed_path):
        if valid_chroms is not None and f.chrom not in valid_chroms:
            continue
        out[f.chrom].append((int(f.start), int(f.end)))
    return out

def intervals_to_dict(intervals):
    """Convert a list of pybedtools.Interval into {chrom: [(start, end), ...]}."""
    out = defaultdict(list)
    for iv in intervals:
        out[iv.chrom].append((int(iv.start), int(iv.end)))
    return out

def build_exclusion_intervals(*interval_dicts):
    """
    Merge any number of {chrom: [(start, end), ...]} dicts into one sorted exclusion dict.
    Overlapping intervals are fine: build_valid_windows tracks the running max end.
    """
    excl = defaultdict(list)
    for d in interval_dicts:
        for chrom, ivs in d.items():
            excl[chrom].extend(ivs)
    for chrom in excl:
        excl[chrom].sort()
    return excl

def build_valid_windows(chrom, chrom_len, exclusion_intervals, seq_len):
    windows = []
    excluded = exclusion_intervals.get(chrom, [])
    prev_end = 0

    for s, e in excluded:
        if s - prev_end >= seq_len:
            for x in range(prev_end, s - seq_len + 1, seq_len):
                windows.append((x, x + seq_len))
        prev_end = max(prev_end, e)

    if chrom_len - prev_end >= seq_len:
        for x in range(prev_end, chrom_len - seq_len + 1, seq_len):
            windows.append((x, x + seq_len))
    
    print(f"[DEBUG] {chrom}: {len(windows)} windows outside peaks/blacklist")

    return windows

ACGT_SET = frozenset("ACGT")

def build_gc_bins(fasta, chrom, windows, precision=3):
    """
    Bin candidate negative windows by reference GC fraction.
    - Windows containing any non-ACGT base (N, IUPAC codes) are dropped.
    - Each bin is shuffled once here, so popping from the end of a bin later
      is uniform random sampling WITHOUT replacement.
    """
    bins = defaultdict(list)
    n_dropped = 0
    for start, end in windows:
        seq = fasta.fetch(chrom, start, end).upper()
        if len(seq) != end - start or not set(seq) <= ACGT_SET:
            n_dropped += 1
            continue
        gc = gc_fraction(seq)
        key = round(gc, precision)
        bins[key].append((start, seq))

    for key in bins:
        order = np.random.permutation(len(bins[key]))
        bins[key] = [bins[key][i] for i in order]

    n_kept = sum(len(v) for v in bins.values())
    print(f"[DEBUG] {chrom}: {n_kept} N-free candidate windows ({n_dropped} dropped for non-ACGT bases)")
    return bins

def sample_gc_nearest_bin(gc_bins, gc_keys, target_gc, max_gc_diff):
    """
    Pop one window from the non-empty GC bin closest to target_gc (sampling without replacement).
    When a bin is exhausted it is deleted and its key removed from gc_keys, so later draws
    fall back to the next-closest GC bin, but never to one more than max_gc_diff away.

    Returns (start, seq, chosen_gc_key, updated_gc_keys). start/seq/key are None if no unused
    window lies within max_gc_diff of target_gc; in that case nothing is consumed.
    """
    if len(gc_keys) == 0:
        return None, None, None, gc_keys

    idx = int(np.searchsorted(gc_keys, target_gc))

    candidate_idxs = []
    if idx < len(gc_keys):
        candidate_idxs.append(idx)
    if idx > 0:
        candidate_idxs.append(idx - 1)

    # choose the closest non-empty GC bin
    best_idx = min(candidate_idxs, key=lambda i: abs(gc_keys[i] - target_gc))
    best_key = gc_keys[best_idx]

    # Hard cap: the nearest non-empty bin is too far, so no acceptable window exists.
    # Small epsilon absorbs float error from rounding both values to 3 decimals.
    if abs(best_key - target_gc) > max_gc_diff + 1e-9:
        return None, None, None, gc_keys

    start, seq = gc_bins[best_key].pop()
    if len(gc_bins[best_key]) == 0:
        del gc_bins[best_key]
        gc_keys = np.delete(gc_keys, best_idx)

    return start, seq, best_key, gc_keys

def build_examples_for_intervals(intervals, fasta, vcf, meth_dir, seq_len, exclusion_intervals, max_gc_diff):
    """
    exclusion_intervals: {chrom: sorted [(start, end), ...]} of regions negatives must not overlap
                         (raw peaks + centered positive windows + blacklist).
    max_gc_diff: largest allowed |GC(negative) - GC(positive)|. Positives with no negative
                 within this tolerance are dropped so the classes stay balanced and matched.
    """
    pos_matrices = []
    neg_matrices = []

    meth_cache = {}

    # Per-chromosome GC window cache: chrom -> (bins, sorted keys). Bins shrink as windows are used.
    window_cache = {}
    n_gc_fallback = 0   # negatives drawn from a different GC bin than the positive's
    n_unmatched = 0     # positives dropped: no unused negative within max_gc_diff
    max_seen_diff = 0.0

    for iv in intervals:
        chrom = iv.chrom
        start = int(iv.start)
        end = int(iv.end)

        try:
            pos_seq = fasta.fetch(chrom, start, end).upper()
        except Exception:
            continue
        if len(pos_seq) != seq_len:
            continue

        # methyl cache
        if chrom not in meth_cache:
            meth_cache[chrom] = load_bedmethyl_for_chrom(chrom, meth_dir)
        meth_dict = meth_cache[chrom]

        # positive
        arr_acgt = overlay_vcf_on_seq(pos_seq, chrom, start, vcf)
        pos_prob = integrate_methylation(arr_acgt, pos_seq, chrom, start, meth_dict)
        if pos_prob.shape != (seq_len, 10):
            continue

        # build GC bins once per chromosome
        if chrom not in window_cache:
            chrom_len = fasta.get_reference_length(chrom)
            windows = build_valid_windows(
                chrom,
                chrom_len,
                exclusion_intervals,
                seq_len
            )
            # window_cache[chrom] = build_gc_bins(fasta, chrom, windows)
            bins = build_gc_bins(fasta, chrom, windows)
            keys = np.array(sorted(bins.keys()))
            window_cache[chrom] = (bins, keys)

        # GC-matched negative, sampled without replacement
        target_gc = round(gc_fraction(pos_seq), 3)
        gc_bins, gc_keys = window_cache[chrom]
        neg_start, neg_seq, chosen_key, gc_keys = sample_gc_nearest_bin(
            gc_bins,
            gc_keys,
            target_gc,
            max_gc_diff
        )
        window_cache[chrom] = (gc_bins, gc_keys)  # keep exhausted bins removed

        if neg_seq is None:
            # no negative within tolerance: drop this positive too
            n_unmatched += 1
            continue
        gc_diff = abs(chosen_key - target_gc)
        if gc_diff > 1e-9:
            n_gc_fallback += 1
        max_seen_diff = max(max_seen_diff, gc_diff)

        arr_acgt_neg = overlay_vcf_on_seq(neg_seq, chrom, neg_start, vcf)
        neg_prob = integrate_methylation(arr_acgt_neg, neg_seq, chrom, neg_start, meth_dict)
        if neg_prob.shape != (seq_len, 10):
            continue
        # append as a matched pair
        pos_matrices.append(pos_prob)
        neg_matrices.append(neg_prob)

    print(f"  [INFO] pairs kept: {len(neg_matrices)}; "
          f"{n_gc_fallback} negatives from a neighboring GC bin (max |GC diff| = {max_seen_diff:.4f}, cap {max_gc_diff}); "
          f"{n_unmatched} positives dropped (no unused negative within cap)")

    if len(pos_matrices) == 0 or len(neg_matrices) == 0:
        return (
            np.zeros((0, seq_len, 10), dtype=np.float32),
            np.zeros((0,), dtype=np.int8),
            None
        )

    pos_X = np.stack(pos_matrices)
    neg_X = np.stack(neg_matrices)

    X = np.concatenate([pos_X, neg_X], axis=0)
    y = np.concatenate([
        np.ones(len(pos_X), dtype=np.int8),
        np.zeros(len(neg_X), dtype=np.int8)
    ])

    return X, y, None

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Create tf train/dev/test sets from peak BEDs with methylation+VCF overlay")
    parser.add_argument("--fasta", required=True, help="Reference genome FASTA (indexed)")
    parser.add_argument("--vcf", required=True, help="VCF (bgzipped + indexed) with variants for sample (GM12878)")
    parser.add_argument("--bedmethyl_dir", required=True, help="Directory containing merged_chr{i}_cpg.bed files")
    parser.add_argument("--bed_list", required=True, nargs="+", help="One or more tf peak bed.gz files (paths). Output will be created for each independently.")
    parser.add_argument("--blacklist", required=True, help="Blacklist BED(.gz) (e.g., hg38-blacklist.v2.bed.gz). Negatives never overlap these regions.")
    parser.add_argument("--seq_len", type=int, default=101, help="Sequence (and matrix) length (odd is recommended). Default 101.")
    parser.add_argument("--test_chroms", nargs="+", default=["chr5", "chr10", "chr14", "chr18", "chr20", "chr22"],
                        help="Chromosomes for test split")
    parser.add_argument("--dev_chroms", nargs="+", default=["chr6", "chr21"],
                        help="Chromosomes for validation (dev) split")
    parser.add_argument("--out_dir", required=True, help="Output directory for .npz files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for negatives generation")
    parser.add_argument("--max_gc_diff", type=float, default=0.01,
                        help="Max allowed |GC fraction difference| between a positive and its negative. "
                             "Positives with no unused negative within this tolerance are dropped. Default 0.01.")
    args = parser.parse_args()

    np.random.seed(args.seed)

    valid_chroms = {f"chr{i}" for i in range(1, 23)}  # autosomes only
    test_chroms = set(args.test_chroms)
    dev_chroms = set(args.dev_chroms)
    train_chroms = valid_chroms - test_chroms - dev_chroms

    os.makedirs(args.out_dir, exist_ok=True)

    # Open fasta and vcf once
    fasta = pysam.FastaFile(args.fasta)
    vcf = pysam.VariantFile(args.vcf)

    # Blacklist is shared across all tfs: load once
    blacklist = load_bed_intervals(args.blacklist, valid_chroms)
    print(f"[INFO] Loaded {sum(len(v) for v in blacklist.values())} blacklist regions from {args.blacklist}")

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

        # Regions negatives must avoid:
        #   - full extent of every raw peak (not just the centered seq_len window)
        #   - the centered positive windows (these can extend past peaks shorter than seq_len)
        #   - blacklisted regions
        # Splits are by chromosome and exclusion is per chromosome, so one dict serves all splits.
        raw_peaks = load_bed_intervals(bed_path, valid_chroms)
        exclusion_intervals = build_exclusion_intervals(
            raw_peaks,
            intervals_to_dict(all_intervals),
            blacklist
        )

        # Build examples and save
        for split_name, ivs in [("train", train_intervals), ("dev", dev_intervals), ("test", test_intervals)]:
            out_path = os.path.join(args.out_dir, f"{split_name}.npz")
            if len(ivs) == 0:
                print(f"  [WARN] No intervals for {split_name}, writing empty arrays to {out_path}")
                np.savez_compressed(out_path, X=np.zeros((0, args.seq_len, 10), dtype=np.float32), y=np.zeros((0,), dtype=np.int8))
                continue
            print(f"  Building {split_name} examples ({len(ivs)} intervals) ...")
            X, y, coords = build_examples_for_intervals(ivs, fasta, vcf, args.bedmethyl_dir, args.seq_len,
                                                        exclusion_intervals, args.max_gc_diff)
            print(f"  -> Saving {X.shape[0]} examples to {out_path}")
            np.savez_compressed(out_path, X=X, y=y)
    fasta.close()
    vcf.close()
    print("[✓] Done!")

if __name__ == "__main__":
    main()