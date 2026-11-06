"""
2_sample_random_intervals.py
Samples N random 510 bp intervals from each genomic element type (CDS, 5UTR, 3UTR, 
Intron, Intergenic).
"""

import argparse
import pandas as pd
import numpy as np


def sample_intervals(bed_path: str, num_samples_per_element: int, seq_len: int, output_csv: str, seed: int = 42):
    np.random.seed(seed)
    df = pd.read_csv(bed_path, sep='\t', names=['chrom', 'start', 'end', 'label'])
    
    # Keep intervals that are large enough to fit a seq_len segment
    df['length'] = df['end'] - df['start']
    df = df[df['length'] >= seq_len].copy()

    sampled_records = []

    for label, group in df.groupby('label'):
        print(f"Sampling {num_samples_per_element} intervals for feature: {label}...")
        
        # Weight segments by their available sampling range
        group['valid_starts'] = group['length'] - seq_len + 1
        weights = group['valid_starts'] / group['valid_starts'].sum()
        
        # Sample interval rows with replacement proportional to valid sampling space
        chosen_indices = np.random.choice(group.index, size=num_samples_per_element, replace=True, p=weights)
        
        for idx in chosen_indices:
            row = group.loc[idx]
            # Random offset start position
            max_offset = row['end'] - row['start'] - seq_len
            offset = np.random.randint(0, max_offset + 1) if max_offset > 0 else 0
            
            sub_start = row['start'] + offset
            sub_end = sub_start + seq_len
            
            sampled_records.append({
                'chrom': row['chrom'],
                'start': sub_start,
                'end': sub_end,
                'label': label
            })

    out_df = pd.DataFrame(sampled_records)
    out_df.to_csv(output_csv, index=False)
    print(f"Saved {len(out_df)} sampled intervals to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bed", type=str, required=True, help="Path to disjoint annotations BED")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples per genomic element")
    parser.add_argument("--seq_len", type=int, default=510, help="Genomic sequence length (510 bp)")
    parser.add_argument("--output", type=str, default="sampled_intervals.csv")
    args = parser.parse_args()

    sample_intervals(args.bed, args.num_samples, args.seq_len, args.output)