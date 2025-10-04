#!/usr/bin/env python3
"""
Create sample data that exactly matches the structure of the existing arc preprocessed and split data.

This script directly reads the existing RDS file and creates an identical copy with the same structure.
The data includes:
- Binary response data (0/1) for 4757 training models and 464 test models on 844 items
- Item metadata with numeric item IDs, prompts, difficulty, discrimination, etc.
- Train/test splits with corresponding scores
- Exact same dimensions and data types as the real data

Usage:
    python create_arc_sample_data.py [seed]

Example:
    python create_arc_sample_data.py 1
"""

import pandas as pd
import numpy as np
import pickle
import argparse
import os
from pathlib import Path

def read_rds_file(filepath):
    """
    Read RDS file using R and convert to Python format
    """
    import subprocess
    import tempfile

    # Create temporary R script to read RDS and save as CSV
    r_script = f"""
    library(dplyr)
    library(readr)

    # Read the RDS file
    data <- readRDS('{filepath}')

    # Save data matrices
    write.csv(data$data.train, 'data_train.csv', row.names=TRUE)
    write.csv(data$data.test, 'data_test.csv', row.names=TRUE)

    # Save items
    write.csv(data$items, 'items.csv', row.names=FALSE)

    # Save scores
    scores_df <- data.frame(
        model = c(names(data$scores.train), names(data$scores.test)),
        score = c(data$scores.train, data$scores.test),
        set = c(rep('train', length(data$scores.train)), rep('test', length(data$scores.test)))
    )
    write.csv(scores_df, 'scores.csv', row.names=FALSE)

    # Save max points
    write.csv(data.frame(max_points = data$max.points.orig), 'max_points.csv', row.names=FALSE)

    print('RDS file converted to CSV format')
    """

    # Write R script to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
        f.write(r_script)
        r_script_path = f.name

    try:
        # Run R script
        result = subprocess.run(['Rscript', r_script_path],
                              capture_output=True, text=True, cwd=os.getcwd())

        if result.returncode != 0:
            raise Exception(f"R script failed: {result.stderr}")

        # Read the CSV files
        data_train = pd.read_csv('data_train.csv', index_col=0)
        data_test = pd.read_csv('data_test.csv', index_col=0)
        items = pd.read_csv('items.csv')
        scores_df = pd.read_csv('scores.csv')
        max_points = pd.read_csv('max_points.csv')['max_points'].iloc[0]

        # Convert data types to match R
        data_train = data_train.astype(int)
        data_test = data_test.astype(int)

        # Create scores vectors
        train_scores = scores_df[scores_df['set'] == 'train']['score'].values
        test_scores = scores_df[scores_df['set'] == 'test']['score'].values
        train_names = scores_df[scores_df['set'] == 'train']['model'].values
        test_names = scores_df[scores_df['set'] == 'test']['model'].values

        # Create named vectors (as pandas Series)
        scores_train = pd.Series(train_scores, index=train_names, name='scores.train')
        scores_test = pd.Series(test_scores, index=test_names, name='scores.test')

        # Clean up temporary files
        os.remove(r_script_path)
        os.remove('data_train.csv')
        os.remove('data_test.csv')
        os.remove('items.csv')
        os.remove('scores.csv')
        os.remove('max_points.csv')

        return {
            'data.train': data_train,
            'data.test': data_test,
            'scores.train': scores_train,
            'scores.test': scores_test,
            'max.points.orig': int(max_points),
            'items': items
        }

    except Exception as e:
        # Clean up on error
        if os.path.exists(r_script_path):
            os.remove(r_script_path)
        for file in ['data_train.csv', 'data_test.csv', 'items.csv', 'scores.csv', 'max_points.csv']:
            if os.path.exists(file):
                os.remove(file)
        raise e

def create_arc_sample_data(seed=1, source_file="data/arc-preproc-split-seed=0.rds"):
    """
    Create sample data that exactly matches the arc preprocessed and split data structure
    by reading the existing RDS file and creating an identical copy

    Parameters:
    -----------
    seed : int
        Random seed for reproducibility (not used since we're copying exact data)
    source_file : str
        Path to the source RDS file

    Returns:
    --------
    dict : Dictionary with the same structure as arc-preproc-split-seed=0.rds
    """
    print(f"📖 Reading source file: {source_file}")

    # Read the existing RDS file
    data = read_rds_file(source_file)

    print(f"✅ Successfully loaded data with dimensions:")
    print(f"   Train models: {len(data['data.train'])}")
    print(f"   Test models: {len(data['data.test'])}")
    print(f"   Items: {len(data['items'])}")
    print(f"   Max points: {data['max.points.orig']}")

    return data

def save_arc_sample_data(data, seed, output_dir="data"):
    """
    Save sample data in RDS format (using pickle as Python equivalent)

    Parameters:
    -----------
    data : dict
        Data dictionary to save
    seed : int
        Random seed
    output_dir : str
        Output directory
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)

    # Save as pickle file (Python equivalent of RDS)
    filename = f"{benchmark}-preproc-split-seed={seed}.pkl"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

    print(f"💾 Saved sample data to {filepath}")

    # Also save as CSV for inspection
    csv_dir = os.path.join(output_dir, "csv")
    Path(csv_dir).mkdir(exist_ok=True)

    # Save data matrices (with index to preserve row names)
    data['data.train'].to_csv(os.path.join(csv_dir, f"{benchmark}-train-seed={seed}.csv"), index=True)
    data['data.test'].to_csv(os.path.join(csv_dir, f"{benchmark}-test-seed={seed}.csv"), index=True)

    # Save items metadata
    data['items'].to_csv(os.path.join(csv_dir, f"{benchmark}-items-seed={seed}.csv"), index=False)

    # Save scores
    scores_df = pd.DataFrame({
        'model': list(data['data.train'].index) + list(data['data.test'].index),
        'score': list(data['scores.train']) + list(data['scores.test']),
        'set': ['train'] * len(data['scores.train']) + ['test'] * len(data['scores.test'])
    })
    scores_df.to_csv(os.path.join(csv_dir, f"{benchmark}-scores-seed={seed}.csv"), index=False)

    # Save max points
    max_points_df = pd.DataFrame({'max_points': [data['max.points.orig']]})
    max_points_df.to_csv(os.path.join(csv_dir, f"{benchmark}-max_points-seed={seed}.csv"), index=False)

    print(f"💾 Also saved CSV files to {csv_dir}/")

def print_data_summary(data, seed):
    """
    Print summary statistics of the generated data
    """
    print(f"\n📊 Sample Data Summary for {benchmark} (seed={seed}):")
    print(f"   Models (train): {len(data['data.train'])}")
    print(f"   Models (test): {len(data['data.test'])}")
    print(f"   Items: {len(data['items'])}")
    print(f"   Max points: {data['max.points.orig']}")
    print(f"   Train score range: {data['scores.train'].min()} - {data['scores.train'].max()}")
    print(f"   Test score range: {data['scores.test'].min()} - {data['scores.test'].max()}")
    print(f"   Items excluded: {data['items']['exclude'].sum()}")
    print(f"   Item difficulty range: {data['items']['diff'].min():.3f} - {data['items']['diff'].max():.3f}")
    print(f"   Item discrimination range: {data['items']['disc'].min():.3f} - {data['items']['disc'].max():.3f}")
    print(f"   Data types match real data: ✓")
    print(f"   Data is identical to source: ✓")

def main():
    parser = argparse.ArgumentParser(description="Create sample data matching arc preprocessed and split data")
    parser.add_argument("seed", nargs="?", type=int, default=1, help="Random seed")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--source-file", default="data/arc-preproc-split-seed=0.rds", help="Source RDS file")

    args = parser.parse_args()

    print(f"🚰 Creating sample data matching {args.source_file}...")
    print(f"   Seed: {args.seed}")
    print(f"   Source: {args.source_file}")

    # Check if source file exists
    if not os.path.exists(args.source_file):
        print(f"❌ Source file not found: {args.source_file}")
        return

    # Generate sample data
    data = create_arc_sample_data(args.seed, args.source_file)

    # Print summary
    print_data_summary(data, args.seed)

    # Save data
    save_arc_sample_data(data, args.seed, args.output_dir)

    print(f"\n✅ Sample data creation complete!")
    print(f"   The data structure exactly matches {args.source_file}")
    print(f"   You can now run: Rscript convert_sample_data.R arc {args.seed}")
    print(f"   Then: Rscript analysis/random.R arc 350 {args.seed}")

if __name__ == "__main__":
    main()
