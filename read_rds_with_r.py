#!/usr/bin/env python3
"""
Read RDS files by calling R directly and converting to CSV/JSON.
"""

import os
import subprocess
import pandas as pd
import json
import tempfile

def read_rds_with_r(benchmark="arc", seed=0, skip_reduced=False):
    """
    Read RDS file by calling R directly and converting to Python-readable format.
    """

    # Construct the file path
    suffix = "-v2" if skip_reduced else ""
    filename = f"{benchmark}-sub-350-seed={seed}{suffix}.rds"
    filepath = os.path.join("data", filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    print(f"Reading data from: {filepath}")

    # Create temporary files for output
    with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as r_script:
        r_script_path = r_script.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as json_file:
        json_path = json_file.name

    try:
        # Create R script to read RDS and convert to JSON
        r_script_content = f'''
# Load the RDS file
data <- readRDS("{filepath}")

# Print structure for debugging
cat("Structure of data:\\n")
str(data)
cat("\\n\\n")

# Print names if it's a list
if (is.list(data)) {{
    cat("Names of data:\\n")
    print(names(data))
    cat("\\n\\n")
}}

# Convert to JSON
library(jsonlite)
json_data <- toJSON(data, auto_unbox = TRUE, digits = 10)
write(json_data, "{json_path}")

# Also save individual components as CSV for easier access
if (is.list(data) && !is.null(names(data))) {{
    if ("data.train" %in% names(data)) {{
        write.csv(data$data.train, "{json_path}.train.csv", row.names = TRUE)
    }}
    if ("data.test" %in% names(data)) {{
        write.csv(data$data.test, "{json_path}.test.csv", row.names = TRUE)
    }}
    if ("scores.train" %in% names(data)) {{
        write.csv(data.frame(scores.train = data$scores.train), "{json_path}.scores_train.csv", row.names = TRUE)
    }}
    if ("scores.test" %in% names(data)) {{
        write.csv(data.frame(scores.test = data$scores.test), "{json_path}.scores_test.csv", row.names = TRUE)
    }}
}}
'''

        # Write R script
        with open(r_script_path, 'w') as f:
            f.write(r_script_content)

        # Run R script
        print("Running R script to read RDS file...")
        result = subprocess.run(['Rscript', r_script_path],
                              capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            print(f"R script failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return None

        print("R script output:")
        print(result.stdout)

        # Read the JSON file
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                json_data = json.load(f)

            print(f"✅ Successfully loaded data from RDS")
            return json_data
        else:
            print(f"❌ JSON file not created: {json_path}")
            return None

    except subprocess.TimeoutExpired:
        print("❌ R script timed out")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None
    finally:
        # Clean up temporary files
        try:
            os.unlink(r_script_path)
            if os.path.exists(json_path):
                os.unlink(json_path)
        except:
            pass

def read_rds_as_csv(benchmark="arc", seed=0, skip_reduced=False):
    """
    Read RDS file and convert individual components to CSV files.
    """

    # Construct the file path
    suffix = "-v2" if skip_reduced else ""
    filename = f"{benchmark}-sub-350-seed={seed}{suffix}.rds"
    filepath = os.path.join("data", filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    print(f"Reading data from: {filepath}")

    # Create temporary R script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as r_script:
        r_script_path = r_script.name

    try:
        # Create R script to read RDS and save as CSV
        r_script_content = f'''
# Load the RDS file
data <- readRDS("{filepath}")

# Print structure
cat("Structure:\\n")
str(data)
cat("\\nNames:\\n")
print(names(data))

# Save components as CSV
if (is.list(data) && !is.null(names(data))) {{
    output_dir <- "temp_rds_output"
    dir.create(output_dir, showWarnings = FALSE)

    if ("data.train" %in% names(data)) {{
        write.csv(data$data.train, file.path(output_dir, "data_train.csv"), row.names = TRUE)
        cat("Saved data.train\\n")
    }}
    if ("data.test" %in% names(data)) {{
        write.csv(data$data.test, file.path(output_dir, "data_test.csv"), row.names = TRUE)
        cat("Saved data.test\\n")
    }}
    if ("scores.train" %in% names(data)) {{
        write.csv(data.frame(scores.train = data$scores.train), file.path(output_dir, "scores_train.csv"), row.names = TRUE)
        cat("Saved scores.train\\n")
    }}
    if ("scores.test" %in% names(data)) {{
        write.csv(data.frame(scores.test = data$scores.test), file.path(output_dir, "scores_test.csv"), row.names = TRUE)
        cat("Saved scores.test\\n")
    }}
    if ("max.points.orig" %in% names(data)) {{
        write.csv(data.frame(max.points.orig = data$max.points.orig), file.path(output_dir, "max_points_orig.csv"), row.names = FALSE)
        cat("Saved max.points.orig\\n")
    }}
}}
'''

        # Write R script
        with open(r_script_path, 'w') as f:
            f.write(r_script_content)

        # Run R script
        print("Running R script to convert RDS to CSV...")
        result = subprocess.run(['Rscript', r_script_path],
                              capture_output=True, text=True, timeout=30)

        print("R script output:")
        print(result.stdout)
        if result.stderr:
            print("R script errors:")
            print(result.stderr)

        # Read the CSV files
        output_dir = "temp_rds_output"
        if os.path.exists(output_dir):
            data_dict = {}

            # Read training data
            train_csv = os.path.join(output_dir, "data_train.csv")
            if os.path.exists(train_csv):
                data_dict['data_train'] = pd.read_csv(train_csv, index_col=0)
                print(f"✅ Loaded training data: {data_dict['data_train'].shape}")

            # Read test data
            test_csv = os.path.join(output_dir, "data_test.csv")
            if os.path.exists(test_csv):
                data_dict['data_test'] = pd.read_csv(test_csv, index_col=0)
                print(f"✅ Loaded test data: {data_dict['data_test'].shape}")

            # Read scores
            scores_train_csv = os.path.join(output_dir, "scores_train.csv")
            if os.path.exists(scores_train_csv):
                scores_df = pd.read_csv(scores_train_csv, index_col=0)
                data_dict['scores_train'] = scores_df['scores.train']
                print(f"✅ Loaded training scores: {len(data_dict['scores_train'])}")

            scores_test_csv = os.path.join(output_dir, "scores_test.csv")
            if os.path.exists(scores_test_csv):
                scores_df = pd.read_csv(scores_test_csv, index_col=0)
                data_dict['scores_test'] = scores_df['scores.test']
                print(f"✅ Loaded test scores: {len(data_dict['scores_test'])}")

            # Read max points
            max_points_csv = os.path.join(output_dir, "max_points_orig.csv")
            if os.path.exists(max_points_csv):
                max_points_df = pd.read_csv(max_points_csv)
                data_dict['max_points_orig'] = max_points_df['max.points.orig'].iloc[0]
                print(f"✅ Loaded max points: {data_dict['max_points_orig']}")

            return data_dict
        else:
            print("❌ Output directory not created")
            return None

    except Exception as e:
        print(f"❌ Error: {e}")
        return None
    finally:
        # Clean up temporary files
        try:
            os.unlink(r_script_path)
        except:
            pass

def main():
    """Example usage."""

    print("Method 1: Reading as CSV...")
    data_dict = read_rds_as_csv(benchmark="arc", seed=0, skip_reduced=False)

    if data_dict and len(data_dict) > 0:
        print("\n=== Data Summary ===")
        for key, value in data_dict.items():
            if hasattr(value, 'shape'):
                print(f"{key}: {value.shape}")
            elif hasattr(value, '__len__'):
                print(f"{key}: {len(value)} items")
            else:
                print(f"{key}: {value}")

        print("\n=== Training Data Preview ===")
        if 'data_train' in data_dict:
            print(data_dict['data_train'].head())

        print("\n=== Training Scores Preview ===")
        if 'scores_train' in data_dict:
            print(data_dict['scores_train'].head())

        return data_dict
    else:
        print("❌ Failed to read data")
        return None

if __name__ == "__main__":
    main()
