#!/usr/bin/env python3
import os
import glob
import subprocess
import argparse
import pandas as pd

ID_SCORE = 90
COVERAGE = 90
EVALUE   = 1e-5

def get_genome_name(genome_path: str) -> str:
    return os.path.basename(genome_path).split(".")[0]

def create_db(genome_fasta: str, blastdb_prefix: str, db_type: str = 'nucl'):
    cmd = ["makeblastdb", "-in", genome_fasta, "-dbtype", db_type, "-out", blastdb_prefix]
    subprocess.run(cmd, check=True)
    print(f"[makeblastdb] created: {blastdb_prefix}")

def blast_search(genes_fasta: str, blastdb_prefix: str, raw_out_path: str, filtered_csv_path: str,
                 out_format: str = "6 qseqid sseqid length pident qcovhsp evalue bitscore sstart send sstrand gapopen"):
    cmd = ["blastn", "-query", genes_fasta, "-db", blastdb_prefix, "-outfmt", out_format, "-out", raw_out_path]
    subprocess.run(cmd, check=True)
    print(f"[blastn] raw results: {raw_out_path}")

    blast_df = pd.read_csv(
        raw_out_path,
        sep="\t",
        header=None,
        names=[
            "query", "subject", "length", "identity", "query_coverage",
            "evalue", "bit_score", "subject_start", "subject_end", "strand", "gaps"
        ],
    )

    # filter and keep best hit per query (max bit_score)
    filtered = blast_df[
        (blast_df["identity"] > ID_SCORE) &
        (blast_df["query_coverage"] >= COVERAGE) &
        (blast_df["evalue"] < EVALUE)
    ]
    if not filtered.empty:
        best_hit = filtered.loc[filtered.groupby("query")["bit_score"].idxmax()]
    else:
        best_hit = pd.DataFrame(columns=blast_df.columns)

    best_hit.to_csv(filtered_csv_path, index=False)
    print(f"[filter] filtered results: {filtered_csv_path}")

    # optional: report missing queries
    all_queries = set(blast_df["query"].unique())
    passed_queries = set(best_hit["query"].unique())
    missing = all_queries - passed_queries
    if missing:
        print(f"[filter] {len(missing)} genes had no high-confidence hit")

def count_genes(results_csv: str):
    if not os.path.exists(results_csv) or os.path.getsize(results_csv) == 0:
        print("[count] no results file or file is empty.")
        return
    result_data = pd.read_csv(results_csv)
    if result_data.empty:
        print("[count] results CSV is empty after filtering.")
        return

    scores = result_data['identity'].tolist()
    print(f"[count] min identity: {min(scores):.2f}")
    print(f"[count] unique genes found: {result_data['query'].nunique()}")
    print(f"[count] total rows kept: {len(result_data)}")

def run_blast_all(assembled_glob: str = "assembled/*.fasta"):
    os.makedirs("blast_db", exist_ok=True)
    os.makedirs("blast_raw", exist_ok=True)
    os.makedirs("results",   exist_ok=True)

    genomes = sorted(glob.glob(assembled_glob))
    if not genomes:
        print(f"[run] No genomes found at {assembled_glob}")
        return

    for genome in genomes:
        strain = get_genome_name(genome)
        genes_fasta = f"filtered_genes/{strain}_genes.fasta"

        if not os.path.exists(genes_fasta):
            print(f"[skip] Missing genes FASTA for {strain}: {genes_fasta}")
            continue

        blastdb_prefix = f"blast_db/{strain}_db"
        raw_out_path   = f"blast_raw/{strain}_raw.txt"
        filtered_csv   = f"results/{strain}_results.csv"

        print(f"\n[run] Strain: {strain}")
        print(f"[run] Genome: {genome}")
        print(f"[run] Genes : {genes_fasta}")

        # build DB and run BLAST for this strain
        create_db(genome, blastdb_prefix)
        blast_search(genes_fasta, blastdb_prefix, raw_out_path, filtered_csv)
        count_genes(filtered_csv)

def main():
    parser = argparse.ArgumentParser(
        description="Batch BLAST genes against all genomes in assembled/*.fasta."
    )
    parser.add_argument(
        "--pattern",
        default="assembled/*.fasta",
        help="Glob pattern for genomes (default: assembled/*.fasta)",
    )
    args = parser.parse_args()
    run_blast_all(args.pattern)

if __name__ == "__main__":
    main()