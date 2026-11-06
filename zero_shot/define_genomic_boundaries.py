"""
1_define_genomic_boundaries.py
Extracts and creates disjoint genomic boundaries for 5 elements on autosomal chromosomes (chr1-chr22):
CDS, 5' UTR, 3' UTR, Intron, and Intergenic.
CDS includes the start codon, but not the stop codon.
"""

import os
import argparse
import pandas as pd
import pybedtools

def create_disjoint_boundaries(gtf_path: str, chrom_sizes_path: str, output_bed: str):
    print("Loading GTF, filtering for autosomes, and standardizing chromosome names...")
    
    # Accept both 'chr1' (GENCODE) and '1' (Ensembl)
    valid_chroms = {f"chr{i}" for i in range(1, 23)}.union({str(i) for i in range(1, 23)})
    target_chroms_with_prefix = {f"chr{i}" for i in range(1, 23)}
    
    # Load GTF using pybedtools
    gtf = pybedtools.BedTool(gtf_path)
    
    # Helper function to ensure all chromosomes have the 'chr' prefix
    def standardize_chrom_name(feature):
        if not feature.chrom.startswith('chr'):
            feature.chrom = f"chr{feature.chrom}"
        return feature
    
    # Filter for valid autosomes, standardize the names, and save to a temporary file
    gtf_auto = gtf.filter(lambda x: x.chrom in valid_chroms).each(standardize_chrom_name).saveas()

    # 1. Extract raw features
    cds = gtf_auto.filter(lambda x: x.fields[2] == "CDS").sort().merge()
    utr5 = gtf_auto.filter(lambda x: x.fields[2] in ["five_prime_utr", "5UTR"]).sort().merge()
    utr3 = gtf_auto.filter(lambda x: x.fields[2] in ["three_prime_utr", "3UTR"]).sort().merge()
    exons = gtf_auto.filter(lambda x: x.fields[2] == "exon").sort().merge()
    genes = gtf_auto.filter(lambda x: x.fields[2] == "gene").sort().merge()

    # 2. Derive Introns: (Exon boundaries inside Gene boundaries)
    # Introns = Gene regions minus Exon regions
    introns = genes.subtract(exons).sort().merge()

    # 3. Derive Intergenic: Entire Genome minus Gene regions
    # Convert 2-column chrom.sizes (chrom, size) into 3-column BED format (chrom, 0, size)
    chrom_sizes_df = pd.read_csv(chrom_sizes_path, sep='\t', header=None, names=['chrom', 'end'])
    chrom_sizes_df['start'] = 0
    
    # Reorder columns to standard BED format: chrom, start, end
    chrom_sizes_df = chrom_sizes_df[['chrom', 'start', 'end']]
    
    # Filter the genome sizes to only the autosomes we are processing
    chrom_sizes_df = chrom_sizes_df[chrom_sizes_df['chrom'].isin(target_chroms_with_prefix)]
    
    # Load the formatted dataframe into pybedtools
    genome_bed = pybedtools.BedTool.from_dataframe(chrom_sizes_df)
    
    # Subtract gene regions from the whole genome to get intergenic regions
    intergenic = genome_bed.subtract(genes).sort().merge()

    print("Enforcing strict disjoint hierarchy: CDS > 5'UTR > 3'UTR > Intron > Intergenic...")
    
    # Disjoint CDS
    cds_disjoint = cds
    
    # Disjoint 5' UTR = 5' UTR - CDS
    utr5_disjoint = utr5.subtract(cds_disjoint).sort().merge()
    
    # Disjoint 3' UTR = 3' UTR - (CDS + 5' UTR)
    occupied_coding = cds_disjoint.cat(utr5_disjoint).sort().merge()
    utr3_disjoint = utr3.subtract(occupied_coding).sort().merge()
    
    # Disjoint Intron = Intron - (CDS + 5' UTR + 3' UTR)
    occupied_exonic = occupied_coding.cat(utr3_disjoint).sort().merge()
    introns_disjoint = introns.subtract(occupied_exonic).sort().merge()
    
    # Disjoint Intergenic = Intergenic - (all genes / exonic / intronic)
    all_genes = genes.sort().merge()
    intergenic_disjoint = intergenic.subtract(all_genes).sort().merge()

    # Combine into a single master dataframe and save
    print("Writing disjoint annotations to output BED file...")
    
    feature_beds = [
        (cds_disjoint, "CDS"),
        (utr5_disjoint, "5UTR"),
        (utr3_disjoint, "3UTR"),
        (introns_disjoint, "Intron"),
        (intergenic_disjoint, "Intergenic")
    ]
    
    records = []
    for bed, label in feature_beds:
        df = bed.to_dataframe()
        if not df.empty:
            df = df.iloc[:, :3]
            df.columns = ["chrom", "start", "end"]
            df["label"] = label
            records.append(df)
            
    master_df = pd.concat(records, ignore_index=True)
    
    # Sort genome-wise
    master_df['chrom_num'] = master_df['chrom'].str.replace('chr', '').astype(int)
    master_df = master_df.sort_values(by=['chrom_num', 'start']).drop(columns=['chrom_num'])
    
    master_df.to_csv(output_bed, sep='\t', index=False, header=False)
    print(f"Successfully saved master disjoint genomic boundaries to {output_bed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gtf", type=str, required=True, help="Path to reference GTF file (e.g. GRCh38)")
    parser.add_argument("--chrom_sizes", type=str, required=True, help="Path to chrom.sizes file for GRCh38")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    create_disjoint_boundaries(args.gtf, args.chrom_sizes, args.output)