"""
3_sample_probability_matrices.py
Retrieves the one-hot/probabilistic matrices (shape: [512, 10]) from train_data.npy / test_data.npy 
for each sampled genomic interval.
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch


def build_genomic_memmap_index(sm_root_dir: str, split: str = "train"):
    """
    Constructs an in-memory index mapping (chrom, genomic_start, genomic_end)
    to memmap row offsets in train_data.npy or test_data.npy.
    """
    total_chroms = [f'chr{i}' for i in range(1, 23)]
    test_chroms = ["chr5", "chr10", "chr20"]
    
    if split == "train":
        split_chroms = [c for c in total_chroms if c not in test_chroms]
        ranges_path = os.path.join(sm_root_dir, "train_ranges.npy")
    else:
        split_chroms = test_chroms
        ranges_path = os.path.join(sm_root_dir, "test_ranges.npy")

    memmap_ranges = np.load(ranges_path)
    
    segment_index = []
    range_ptr = 0

    for cname in split_chroms:
        chr_path = os.path.join(sm_root_dir, "chr_data", cname)
        chr_npy_list = [f for f in os.listdir(chr_path) if f.endswith('.npy')]
        chr_npy_list.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        
        for nf in chr_npy_list:
            part_data = np.load(os.path.join(chr_path, nf), allow_pickle=True).item()
            g_start, g_end = part_data["sp_range"]
            
            m_start, m_end = memmap_ranges[range_ptr]
            
            segment_index.append({
                'chrom': cname,
                'g_start': g_start,
                'g_end': g_end,
                'm_start': m_start,
                'm_end': m_end
            })
            range_ptr += 1

    return pd.DataFrame(segment_index)


def extract_probability_matrices(intervals_csv: str, sm_root_dir: str, output_npy: str, seq_length: int = 510):
    intervals_df = pd.read_csv(intervals_csv)
    
    print("Building memmap index for train and test splits...")
    train_index = build_genomic_memmap_index(sm_root_dir, split="train")
    test_index = build_genomic_memmap_index(sm_root_dir, split="test")

    # Open memmapped continuous matrices
    train_data_path = os.path.join(sm_root_dir, "train_data.npy")
    test_data_path = os.path.join(sm_root_dir, "test_data.npy")

    train_mmap = np.memmap(train_data_path, dtype=np.float32, mode='r').reshape(-1, 10)
    test_mmap = np.memmap(test_data_path, dtype=np.float32, mode='r').reshape(-1, 10)

    vocab_size = 10
    total_samples = len(intervals_df)
    
    valid_matrices = []
    valid_labels = []
    skipped_count = 0

    print(f"Extracting up to {total_samples} probability matrices...")

    for i, row in intervals_df.iterrows():
        chrom, start, end, label = row['chrom'], row['start'], row['end'], row['label']

        # 1. Search in train_index first
        match = train_index[(train_index['chrom'] == chrom) & 
                            (train_index['g_start'] <= start) & 
                            (train_index['g_end'] >= end)]
        
        mmap_source = train_mmap
        if match.empty:
            # 2. Search in test_index
            match = test_index[(test_index['chrom'] == chrom) & 
                               (test_index['g_start'] <= start) & 
                               (test_index['g_end'] >= end)]
            mmap_source = test_mmap

        if match.empty:
            # Skip this interval as it falls into an 'N' gap
            skipped_count += 1
            continue

        seg = match.iloc[0]
        offset = start - seg['g_start']
        m_slice_start = seg['m_start'] + offset
        m_slice_end = m_slice_start + seq_length

        # Fetch 510 bp matrix slice
        matrix_slice = mmap_source[m_slice_start:m_slice_end]

        # Construct 512 length tensor with [CLS] at idx 0 and [SEP] at idx -1
        full_sample = np.zeros((seq_length + 2, vocab_size), dtype=np.float32)
        full_sample[0, 1] = 1.0   # [CLS] token
        full_sample[-1, 2] = 1.0  # [SEP] token
        full_sample[1:-1] = matrix_slice

        valid_matrices.append(full_sample)
        valid_labels.append(label)

    # Convert to numpy arrays
    final_matrices = np.array(valid_matrices, dtype=np.float32)
    final_labels = np.array(valid_labels)

    print(f"Extraction complete!")
    print(f"Successfully extracted: {len(final_matrices)}")
    print(f"Skipped (due to 'N' gaps): {skipped_count}")

    # Save sampled matrices and labels
    np.savez_compressed(output_npy, matrices=final_matrices, labels=final_labels)
    print(f"Saved to {output_npy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--intervals", type=str, required=True, help="Path to sampled_intervals.csv")
    parser.add_argument("--sm_root", type=str, required=True, help="Path to preprocessed sm_510 directory")
    parser.add_argument("--output", type=str, default="sampled_matrices.npz")
    args = parser.parse_args()

    extract_probability_matrices(args.intervals, args.sm_root, args.output)