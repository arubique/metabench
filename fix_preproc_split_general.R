#!/usr/bin/env Rscript

# General-purpose script to fix items with no variance in preproc-split data
# This addresses the error: "The following items have only one response category and cannot be estimated"
# This should be applied BEFORE subsampling to prevent the issue from occurring
#
# Usage: Rscript fix_preproc_split_general.R {benchmark} [seed]
# Example: Rscript fix_preproc_split_general.R arc_disco_ood 0

# Load required libraries
library(here)

# Set working directory
here::i_am("fix_preproc_split_general.R")

# Function to fix preproc-split data by removing items with no variance
fix_preproc_split_no_variance <- function(input_file, output_file) {
  cat("Loading preproc-split data from:", input_file, "\n")
  data <- readRDS(input_file)

  # Get training and test data
  train_data <- data$data.train
  test_data <- data$data.test

  cat("Original training data dimensions:", dim(train_data), "\n")
  cat("Original test data dimensions:", dim(test_data), "\n")

  # Find items with no variance in training data
  item_variance <- apply(train_data, 2, sd)
  no_variance_items <- which(item_variance == 0)

  cat("Items with no variance:", length(no_variance_items), "\n")
  if (length(no_variance_items) > 0) {
    cat("Problematic item indices (first 20):", paste(head(no_variance_items, 20), collapse = ", "), "\n")
    if (length(no_variance_items) > 20) {
      cat("... and", length(no_variance_items) - 20, "more items\n")
    }

    # Remove these items from both training and test data
    train_data_fixed <- train_data[, -no_variance_items, drop = FALSE]
    test_data_fixed <- test_data[, -no_variance_items, drop = FALSE]

    cat("Fixed training data dimensions:", dim(train_data_fixed), "\n")
    cat("Fixed test data dimensions:", dim(test_data_fixed), "\n")

    # Update the data object
    data$data.train <- train_data_fixed
    data$data.test <- test_data_fixed

    # Update max.points.orig to reflect the new number of items
    data$max.points.orig <- ncol(train_data_fixed)

    cat("Removed", length(no_variance_items), "items with no variance\n")
    cat("New total items:", ncol(train_data_fixed), "\n")

    # Show some statistics about the removed items
    removed_train_means <- colMeans(train_data[, no_variance_items, drop = FALSE])
    all_zeros <- sum(removed_train_means == 0)
    all_ones <- sum(removed_train_means == 1)
    cat("Removed items breakdown:\n")
    cat("  - Items with all 0s:", all_zeros, "\n")
    cat("  - Items with all 1s:", all_ones, "\n")

  } else {
    cat("No items with no variance found. Data is already clean.\n")
  }

  # Save the fixed data
  cat("Saving fixed preproc-split data to:", output_file, "\n")
  saveRDS(data, output_file)

  return(data)
}

# Main execution
args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 1) {
  benchmark <- args[1]
  seed <- ifelse(length(args) >= 2, args[2], "0")

  # Construct file paths
  input_file <- paste0("data/", benchmark, "-preproc-split-seed=", seed, ".rds")
  output_file <- paste0("data/", benchmark, "-preproc-split-seed=", seed, "_fixed.rds")

} else {
  cat("Usage: Rscript fix_preproc_split_general.R {benchmark} [seed]\n")
  cat("Example: Rscript fix_preproc_split_general.R arc_disco_ood 0\n")
  cat("Example: Rscript fix_preproc_split_general.R mmlu_disco_ood 0\n")
  stop("Please provide at least the benchmark name as an argument.")
}

cat("Benchmark:", benchmark, "\n")
cat("Seed:", seed, "\n")
cat("Input file:", input_file, "\n")
cat("Output file:", output_file, "\n")

# Check if input file exists
if (!file.exists(input_file)) {
  stop("Input file does not exist: ", input_file)
}

# Fix the data
fixed_data <- fix_preproc_split_no_variance(input_file, output_file)

cat("✅ Preproc-split data fixed successfully!\n")
cat("You can now use the fixed data file:", output_file, "\n")
cat("This should prevent the no-variance error when subsampling and running IRT analysis.\n")
cat("\nNext steps:\n")
cat("1. Use the fixed preproc-split file for subsampling\n")
cat("2. Run your crossvalidate.R script with the subsampled data\n")
cat("3. The IRT model should now fit without the no-variance error\n")
