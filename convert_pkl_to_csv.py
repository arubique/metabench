#!/usr/bin/env python3
"""
Convert .pkl file to CSV format matching the format used in create_arc_sample_data.py

This script loads a pickle file with the same structure as created by create_arc_sample_data.py
and converts it to CSV files in the exact same format.

Usage:
    python convert_pkl_to_csv.py <pkl_file> [output_dir] [seed]

Example:
    python convert_pkl_to_csv.py data/arc-preproc-split-seed=0.pkl data/csv 0
"""

import pandas as pd
import pickle
import argparse
import os
from pathlib import Path

def load_pkl_file(filepath):
    """
    Load pickle file and return the data dictionary

    Parameters:
    -----------
    filepath : str
        Path to the pickle file

    Returns:
    --------
    dict : Data dictionary with the same structure as create_arc_sample_data.py
    """
    print(f"📖 Loading pickle file: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Pickle file not found: {filepath}")

    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    print(f"✅ Successfully loaded data with dimensions:")
    print(f"   Train models: {len(data['data.train'])}")
    print(f"   Test models: {len(data['data.test'])}")
    print(f"   Items: {len(data['items']) if data['items'] is not None else 'None'}")
    print(f"   Max points: {data['max.points.orig']}")

    return data

def save_pkl_to_csv(data, seed, pkl_filename, output_dir="data/csv", benchmark="arc"):
    """
    Save pickle data to CSV files in the same format as create_arc_sample_data.py

    Parameters:
    -----------
    data : dict
        Data dictionary loaded from pickle file
    seed : int
        Seed number for file naming
    pkl_filename : str
        Original pickle filename (used to create subdirectory)
    output_dir : str
        Base output directory for CSV files
    """
    # Create subdirectory based on pkl filename (without extension)
    pkl_basename = Path(pkl_filename).stem
    csv_subdir = Path(output_dir) / pkl_basename

    # Create output directory if it doesn't exist
    csv_subdir.mkdir(parents=True, exist_ok=True)

    print(f"💾 Saving CSV files to {csv_subdir}/")

    # Save data matrices (with index to preserve row names)
    data['data.train'].to_csv(csv_subdir / f"{benchmark}-train-seed={seed}.csv", index=True)
    print(f"   - {benchmark}-train-seed={seed}.csv")
    data['data.test'].to_csv(csv_subdir / f"{benchmark}-test-seed={seed}.csv", index=True)
    print(f"   - {benchmark}-test-seed={seed}.csv")

    # Save items metadata (if it exists)
    if data['items'] is not None:
        data['items'].to_csv(csv_subdir / f"{benchmark}-items-seed={seed}.csv", index=False)
        print(f"   - {benchmark}-items-seed={seed}.csv")

    else:
        print(f"   - {benchmark}-items-seed={seed}.csv (skipped - items is None)")
    # Save scores in the same format as create_arc_sample_data.py
    # Handle both DataFrame and Series formats for scores
    if isinstance(data['scores.train'], pd.DataFrame):
        train_scores = data['scores.train'].iloc[:, 0]  # Get first column
        train_names = data['scores.train'].index
    else:
        train_scores = data['scores.train']
        train_names = data['scores.train'].index

    if isinstance(data['scores.test'], pd.DataFrame):
        test_scores = data['scores.test'].iloc[:, 0]  # Get first column
        test_names = data['scores.test'].index
    else:
        test_scores = data['scores.test']
        test_names = data['scores.test'].index

    scores_df = pd.DataFrame({
        'model': list(train_names) + list(test_names),
        'score': list(train_scores) + list(test_scores),
        'set': ['train'] * len(train_scores) + ['test'] * len(test_scores)
    })
    scores_df.to_csv(csv_subdir / f"{benchmark}-scores-seed={seed}.csv", index=False)
    print(f"   - {benchmark}-scores-seed={seed}.csv")
    # Save max points
    max_points_df = pd.DataFrame({'max_points': [data['max.points.orig']]})
    max_points_df.to_csv(csv_subdir / f"{benchmark}-max_points-seed={seed}.csv", index=False)

    print(f"✅ Successfully saved all CSV files to {csv_subdir}/:")
    print(f"   - {benchmark}-train-seed={seed}.csv")
    print(f"   - {benchmark}-test-seed={seed}.csv")
    if data['items'] is not None:
        print(f"   - {benchmark}-items-seed={seed}.csv")
    print(f"   - {benchmark}-scores-seed={seed}.csv")
    print(f"   - {benchmark}-max_points-seed={seed}.csv")

def print_data_summary(data, seed, benchmark):
    """
    Print summary statistics of the loaded data
    """
    print(f"\n📊 Data Summary for {benchmark} (seed={seed}):")
    print(f"   Models (train): {len(data['data.train'])}")
    print(f"   Models (test): {len(data['data.test'])}")
    print(f"   Items: {len(data['items']) if data['items'] is not None else 'None'}")
    print(f"   Max points: {data['max.points.orig']}")

    # Handle different score formats
    if isinstance(data['scores.train'], pd.DataFrame):
        train_scores = data['scores.train'].iloc[:, 0]
        test_scores = data['scores.test'].iloc[:, 0]
    else:
        train_scores = data['scores.train']
        test_scores = data['scores.test']

    print(f"   Train score range: {train_scores.min()} - {train_scores.max()}")
    print(f"   Test score range: {test_scores.min()} - {test_scores.max()}")

    if data['items'] is not None:
        print(f"   Items excluded: {data['items']['exclude'].sum()}")
        print(f"   Item difficulty range: {data['items']['diff'].min():.3f} - {data['items']['diff'].max():.3f}")
        print(f"   Item discrimination range: {data['items']['disc'].min():.3f} - {data['items']['disc'].max():.3f}")
    else:
        print(f"   Items metadata: Not available")

def main():
    parser = argparse.ArgumentParser(description="Convert .pkl file to CSV format")
    # parser.add_argument("pkl_file", help="Path to the pickle file to convert")
    parser.add_argument("output_dir", nargs="?", default="data/csv", help="Output directory for CSV files (default: csv)")
    parser.add_argument("seed", nargs="?", type=int, default=0, help="Seed number for file naming (default: 0)")
    parser.add_argument("benchmark", nargs="?", default="arc", help="Benchmark name (default: arc)")

    args = parser.parse_args()
    pkl_file = f"data/{args.benchmark}-preproc-split-seed={args.seed}.pkl"

    print(f"🔄 Converting {pkl_file} to CSV format...")
    print(f"   Seed: {args.seed}")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Benchmark: {args.benchmark}")
    try:
        # Load pickle file
        data = load_pkl_file(pkl_file)

        # Print summary
        print_data_summary(data, args.seed, args.benchmark)

        # Save to CSV
        save_pkl_to_csv(data, args.seed, pkl_file, args.output_dir, args.benchmark)

        # Get the subdirectory path for final message
        pkl_basename = Path(pkl_file).stem
        csv_subdir = Path(args.output_dir) / pkl_basename

        print(f"\n✅ Conversion complete!")
        print(f"   CSV files saved to: {csv_subdir}/")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
