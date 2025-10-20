#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rarefaction Analysis Script for Microbiome Data

This script performs rarefaction on a feature (e.g., OTU, ASV) table to
generate data for rarefaction curves. It is designed for high-performance,
memory-efficient analysis of large datasets.

Features:
-   Calculates multiple, user-specified alpha diversity metrics.
-   Uses multiprocessing to parallelize analysis across all available CPU cores.
-   Employs a low-memory, "append-to-file" strategy to handle massive datasets
    that would otherwise exceed available RAM.
-   Accepts feature tables where samples are oriented as either rows or columns.
-   Validates alpha diversity metric names against scikit-bio's library.
-   Provides a user-friendly command-line interface with detailed help.

Usage:
    python this_script_name.py -i <input_table.csv> -o <output.csv> -d "1000,5000,10000" -m "observed_otus,shannon" --sample-as-rows

"""

__author__ = "Gemini & ShaomingXiao"
__version__ = "1.0.0"
__creation_date__ = "2025-08-15"

import pandas as pd
import numpy as np
import skbio
import multiprocessing as mp
from functools import partial
import argparse
import time
from typing import List, Tuple
import sys

# --- Pre-computation of Valid Metrics ---
VALID_METRICS = {"ace", "chao1", "margalef", "menhinick", "osd", "observed_otus",
                 "brillouin_d", "enspie", "fisher_alpha", "kempton_taylor_q", "simpson", "shannon", "renyi",
                 "pielou_e", "simpson_e", "mcintosh_e", "heip_e"}

def parse_and_validate_metrics(metric_string: str) -> List[str]:
    """
    Custom argparse type function to parse and validate a comma-separated
    string of alpha diversity metrics.
    """
    requested_metrics = {m.strip() for m in metric_string.split(',')}
    invalid_metrics = requested_metrics - VALID_METRICS
    
    if invalid_metrics:
        error_msg = f"Invalid metric(s): {', '.join(invalid_metrics)}.\n"
        error_msg += f"Please choose from the following non-phylogenetic metrics:\n"
        error_msg += f"{', '.join(sorted(list(VALID_METRICS)))}"
        raise argparse.ArgumentTypeError(error_msg)
        
    return sorted(list(requested_metrics))


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the rarefaction script."""
    parser = argparse.ArgumentParser(
        description="Run massive-scale rarefaction with built-in metric validation."
    )
    # --- I/O and Layout Arguments ---
    parser.add_argument(
        "-i", "--input_file", type=str, required=True,
        help="Path to the input feature table (TSV)."
    )
    parser.add_argument(
        "-o", "--output_file", type=str, required=True,
        help="Path to save the detailed output (TSV)."
    )
    parser.add_argument(
        "--sample-as-rows", action="store_true",
        help="Specify this flag if samples are in rows. Default assumes samples are in columns."
    )
    
    # --- Rarefaction Parameters ---
    parser.add_argument(
        "-d", "--depths", type=str, required=True,
        help="Comma-separated list of rarefaction depths."
    )
    parser.add_argument(
        "-m", "--metrics", type=parse_and_validate_metrics, default="observed_otus",
        help="Comma-separated list of alpha diversity metrics. The five most common are: "
             "'observed_otus', 'shannon', 'simpson', 'chao1', and 'ace'."
    )
    parser.add_argument(
        "-r", "--repeats", type=int, default=10,
        help="Number of repeated subsamplings for each depth."
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=42,
        help="Random seed for reproducibility."
    )
    
    # --- Performance ---
    parser.add_argument(
        "-p", "--processes", type=int, default=mp.cpu_count(),
        help="Number of CPU processes to use."
    )
    return parser.parse_args()

def safe_append_worker(
    task: Tuple[int, int],
    feature_table: pd.DataFrame,
    metrics: List[str],
    base_seed: int,
    output_filename: str,
    lock: mp.Lock
) -> None:
    """
    Worker that calculates diversity and safely appends the result to a shared file.
    """
    depth, iteration_index = task
    run_seed = base_seed + iteration_index
    np.random.seed(run_seed)
    
    valid_samples = feature_table[feature_table.sum(axis=1) >= depth]
    if valid_samples.empty:
        return

    rarefied_table = valid_samples.apply(
        lambda row: skbio.stats.subsample_counts(row.astype(int), n=depth, replace=False),
        axis=1, result_type='expand'
    )
    rarefied_table.columns = valid_samples.columns

    # create temporary dataframe holding the alpha diversity metrics
    alpha_div_df = pd.DataFrame(columns=metrics, index=rarefied_table.index)

    # loop through metrics
    for metric in metrics:
        alpha_div_df[metric] = skbio.diversity.alpha_diversity(
        metric=metric, counts=rarefied_table.values, ids=rarefied_table.index
    )
    
    alpha_div_df['depth'] = depth
    alpha_div_df['iteration'] = iteration_index + 1
    alpha_div_df = alpha_div_df.reset_index().rename(columns={'index': 'sample_id'})
    
    with lock:
        alpha_div_df.to_csv(output_filename,
                            mode='a',
                            header=False,
                            index=False,
                            sep="\t")
    
    return

def main() -> None:
    """Main function to orchestrate the rarefaction."""
    args = parse_arguments()

    print(f"Loading feature table from {args.input_file}...")
    try:
        df = pd.read_csv(args.input_file, sep="\t",index_col=0)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.gtiinput_file}")
        sys.exit(1)
        
    if not args.sample_as_rows:
        print("Default behavior: Assuming samples are in columns. Transposing data...")
        df = df.T
    else:
        print("`--sample-as-rows` flag detected. Assuming samples are in rows.")
    
    print(f"Data loaded: {df.shape[0]} samples and {df.shape[1]} features.")

    try:
        depth_list: List[int] = sorted([int(d) for d in args.depths.split(',')])
    except ValueError:
        print("Error: Depths must be a comma-separated list of integers.")
        sys.exit(1)

    header_cols = ['sample_id'] + args.metrics + ['depth', 'iteration']
    pd.DataFrame(columns=header_cols).to_csv(args.output_file, index=False, sep="\t")
    
    tasks = []
    iteration_counter = 0
    for depth in depth_list:
        for _ in range(args.repeats):
            tasks.append((depth, iteration_counter))
            iteration_counter += 1
            
    total_tasks = len(tasks)
    print(f"Created {total_tasks} individual tasks to run across {args.processes} processes.")
    start_time = time.time()

    manager = mp.Manager()
    lock = manager.Lock()
    
    worker_func = partial(
        safe_append_worker,
        feature_table=df,
        metrics=args.metrics,
        base_seed=args.seed,
        output_filename=args.output_file,
        lock=lock
    )

    with mp.Pool(processes=args.processes) as pool:
        pool.map(worker_func, tasks)

    print("All tasks completed. Results have been written directly to disk.")
    end_time = time.time()
    print(f"Processing finished in {end_time - start_time:.2f} seconds.")
    print(f"Output saved to {args.output_file}")

if __name__ == '__main__':
    main()

