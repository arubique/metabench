#!/usr/bin/env python3
"""
Read the raw data files that are being preprocessed in preprocess.R
"""

import pandas as pd
import os
from pathlib import Path

def read_raw_data(benchmark="arc"):
    """
    Read the raw data files for a given benchmark.

    Parameters:
    -----------
    benchmark : str
        Benchmark name (arc, gsm8k, hellaswag, mmlu, truthfulqa, winogrande)

    Returns:
    --------
    dict : Dictionary containing raw data and prompts
    """

    print(f"Reading raw data for {benchmark}...")

    if benchmark == "mmlu":
        return read_mmlu_raw_data()
    else:
        return read_standard_raw_data(benchmark)

def read_standard_raw_data(benchmark):
    """
    Read raw data for standard benchmarks (non-MMLU).
    """

    # File paths
    data_file = f"data/{benchmark}.csv"
    prompts_file = f"data/{benchmark}_prompts.csv"

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    if not os.path.exists(prompts_file):
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

    print(f"Reading data from: {data_file}")
    print(f"Reading prompts from: {prompts_file}")

    # Read the raw data
    df = pd.read_csv(data_file)
    prompts = pd.read_csv(prompts_file)

    print(f"Raw data shape: {df.shape}")
    print(f"Prompts shape: {prompts.shape}")

    # Show the structure
    print(f"\nRaw data columns: {list(df.columns)}")
    print(f"Prompts columns: {list(prompts.columns)}")

    print(f"\nRaw data preview:")
    print(df.head())

    print(f"\nPrompts preview:")
    print(prompts.head())

    # Check for duplicates in prompts
    duplicates = prompts.duplicated(subset=['prompt']).sum()
    if duplicates > 0:
        print(f"\n⚠️  Found {duplicates} duplicate prompts")

    # Transform data to match the R processing
    # The R code uses df2data which pivots the data
    print(f"\nTransforming data to match R processing...")

    # Convert to the format expected by df2data
    # The raw data should have columns: source, item, correct
    if 'source' in df.columns and 'item' in df.columns and 'correct' in df.columns:
        # Pivot the data to get subjects as rows and items as columns
        data_matrix = df.pivot(index='source', columns='item', values='correct')
        data_matrix = data_matrix.fillna(0)  # Fill missing values with 0

        print(f"Transformed data matrix shape: {data_matrix.shape}")
        print(f"Data matrix preview:")
        print(data_matrix.head())

        # Calculate scores
        scores = data_matrix.sum(axis=1)
        print(f"\nScore statistics:")
        print(f"Mean score: {scores.mean():.2f}")
        print(f"Std score: {scores.std():.2f}")
        print(f"Min score: {scores.min()}")
        print(f"Max score: {scores.max()}")

        return {
            'raw_data': df,
            'prompts': prompts,
            'data_matrix': data_matrix,
            'scores': scores,
            'max_points': len(data_matrix.columns)
        }
    else:
        print(f"❌ Expected columns 'source', 'item', 'correct' not found")
        print(f"Available columns: {list(df.columns)}")
        return {
            'raw_data': df,
            'prompts': prompts,
            'data_matrix': None,
            'scores': None,
            'max_points': None
        }

def read_mmlu_raw_data():
    """
    Read raw data for MMLU benchmark (special case with multiple files).
    """

    print("Reading MMLU raw data (multiple files)...")

    # Find all MMLU files
    data_dir = "data"
    mmlu_data_files = [f for f in os.listdir(data_dir) if f.startswith('mmlu_') and f.endswith('.csv') and 'prompts' not in f]
    mmlu_prompt_files = [f for f in os.listdir(data_dir) if f.startswith('mmlu_') and f.endswith('_prompts.csv')]

    print(f"Found {len(mmlu_data_files)} MMLU data files:")
    for f in sorted(mmlu_data_files):
        print(f"  - {f}")

    print(f"Found {len(mmlu_prompt_files)} MMLU prompt files:")
    for f in sorted(mmlu_prompt_files):
        print(f"  - {f}")

    # Read all data files
    data_list = []
    prompt_list = []

    for data_file in sorted(mmlu_data_files):
        benchmark = data_file.replace('mmlu_', '').replace('.csv', '')
        filepath = os.path.join(data_dir, data_file)

        print(f"\nReading {data_file}...")
        df = pd.read_csv(filepath)

        # Transform to match R processing
        if 'source' in df.columns and 'item' in df.columns and 'correct' in df.columns:
            # Add benchmark prefix to item names
            df['item'] = f"{benchmark}.{df['item']}"

            # Pivot the data
            data_matrix = df.pivot(index='source', columns='item', values='correct')
            data_matrix = data_matrix.fillna(0)

            data_list.append(data_matrix)
            print(f"  Shape: {data_matrix.shape}")

    # Read all prompt files
    for prompt_file in sorted(mmlu_prompt_files):
        benchmark = prompt_file.replace('mmlu_', '').replace('_prompts.csv', '')
        filepath = os.path.join(data_dir, prompt_file)

        print(f"\nReading {prompt_file}...")
        prompts = pd.read_csv(filepath)

        # Add benchmark prefix to item names
        prompts['item'] = f"{benchmark}.{prompts['item']}"

        prompt_list.append(prompts)
        print(f"  Shape: {prompts.shape}")

    # Merge all data matrices
    if data_list:
        combined_data = pd.concat(data_list, axis=1)
        combined_prompts = pd.concat(prompt_list, ignore_index=True)

        print(f"\nCombined data shape: {combined_data.shape}")
        print(f"Combined prompts shape: {combined_prompts.shape}")

        # Calculate scores
        scores = combined_data.sum(axis=1)
        print(f"\nScore statistics:")
        print(f"Mean score: {scores.mean():.2f}")
        print(f"Std score: {scores.std():.2f}")
        print(f"Min score: {scores.min()}")
        print(f"Max score: {scores.max()}")

        return {
            'raw_data_files': mmlu_data_files,
            'prompt_files': mmlu_prompt_files,
            'data_matrix': combined_data,
            'prompts': combined_prompts,
            'scores': scores,
            'max_points': len(combined_data.columns)
        }
    else:
        print("❌ No valid MMLU data files found")
        return None

def analyze_raw_data(benchmark="arc"):
    """
    Analyze the raw data to understand its structure and quality.
    """

    print(f"Analyzing raw data for {benchmark}...")

    data = read_raw_data(benchmark)

    if data is None or data['data_matrix'] is None:
        print("❌ Could not read data")
        return None

    data_matrix = data['data_matrix']
    prompts = data['prompts']
    scores = data['scores']

    print(f"\n{'='*50}")
    print(f"RAW DATA ANALYSIS FOR {benchmark.upper()}")
    print(f"{'='*50}")

    print(f"\n1. DATA STRUCTURE:")
    print(f"   Subjects (models): {data_matrix.shape[0]}")
    print(f"   Items (questions): {data_matrix.shape[1]}")
    print(f"   Total responses: {data_matrix.size}")
    print(f"   Missing values: {data_matrix.isna().sum().sum()}")

    print(f"\n2. SCORE DISTRIBUTION:")
    print(f"   Mean score: {scores.mean():.2f}")
    print(f"   Median score: {scores.median():.2f}")
    print(f"   Std deviation: {scores.std():.2f}")
    print(f"   Min score: {scores.min()}")
    print(f"   Max score: {scores.max()}")
    print(f"   Score range: {scores.max() - scores.min()}")

    # Percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"   Percentiles: {percentiles}")
    for p in percentiles:
        print(f"     {p}th percentile: {scores.quantile(p/100):.2f}")

    print(f"\n3. ITEM ANALYSIS:")
    item_correct = data_matrix.sum(axis=0)
    item_difficulty = item_correct / len(data_matrix)

    print(f"   Mean item difficulty: {item_difficulty.mean():.3f}")
    print(f"   Min item difficulty: {item_difficulty.min():.3f}")
    print(f"   Max item difficulty: {item_difficulty.max():.3f}")
    print(f"   Items with 0% correct: {(item_difficulty == 0).sum()}")
    print(f"   Items with 100% correct: {(item_difficulty == 1).sum()}")

    print(f"\n4. SUBJECT ANALYSIS:")
    print(f"   Subjects with 0% correct: {(scores == 0).sum()}")
    print(f"   Subjects with 100% correct: {(scores == data_matrix.shape[1]).sum()}")

    # Check for outliers (similar to R preprocessing)
    threshold = scores.quantile(0.001)
    outliers = scores <= threshold
    print(f"   Outliers (≤{threshold:.2f}): {outliers.sum()}")

    return data

def main():
    """Example usage."""

    benchmarks = ["arc", "gsm8k", "hellaswag", "mmlu", "truthfulqa", "winogrande"]

    for benchmark in benchmarks:
        try:
            print(f"\n{'='*60}")
            print(f"ANALYZING {benchmark.upper()}")
            print(f"{'='*60}")

            data = analyze_raw_data(benchmark)

            if data and data['data_matrix'] is not None:
                print(f"✅ Successfully analyzed {benchmark}")
            else:
                print(f"❌ Failed to analyze {benchmark}")

        except Exception as e:
            print(f"❌ Error analyzing {benchmark}: {e}")
            continue

if __name__ == "__main__":
    main()
