# Convert Python-generated sample data to RDS format for R analysis
# usage: Rscript convert_sample_data.R {benchmark} {seed}

# =============================================================================
# Load required libraries
library(dplyr)
library(readr)

# =============================================================================
# Parse arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript convert_sample_data.R {benchmark} {seed}")
}
benchmark <- args[1]
seed <- as.numeric(args[2])

# =============================================================================
# Helper function to convert Python data to R format
convert_python_data <- function(benchmark, seed) {
  # Load Python data
  python_data_path <- glue::glue("data/{benchmark}-preproc-split-seed={seed}.pkl")

  if (!file.exists(python_data_path)) {
    stop(glue::glue("Python data file not found: {python_data_path}"))
  }

  # Use reticulate to load Python pickle file
  py <- import("pickle")
  with(py$open(python_data_path, "rb"), {
    python_data <- pickle$load(f)
  })

  # Convert to R format
  r_data <- list(
    data.train = as.data.frame(python_data$data.train),
    data.test = as.data.frame(python_data$data.test),
    scores.train = as.numeric(python_data$scores.train),
    scores.test = as.numeric(python_data$scores.test),
    max.points.orig = as.numeric(python_data$max.points.orig),
    items = as.data.frame(python_data$items)
  )

  # Ensure proper data types
  r_data$data.train <- as.data.frame(lapply(r_data$data.train, as.numeric))
  r_data$data.test <- as.data.frame(lapply(r_data$data.test, as.numeric))

  # Convert items dataframe
  r_data$items$item <- as.character(r_data$items$item)
  r_data$items$prompt <- as.character(r_data$items$prompt)
  r_data$items$diff <- as.numeric(r_data$items$diff)
  r_data$items$disc <- as.numeric(r_data$items$disc)
  r_data$items$sd <- as.numeric(r_data$items$sd)
  r_data$items$exclude <- as.logical(r_data$items$exclude)

  return(r_data)
}

# =============================================================================
# Alternative function using CSV files (if reticulate is not available)
convert_csv_data <- function(benchmark, seed) {
  csv_dir <- glue::glue("data/csv")

  # Load data matrices (with row names)
  # Read the CSV file and handle the row names properly
  data_train_raw <- read_csv(glue::glue("{csv_dir}/{benchmark}-train-seed={seed}.csv"),
                            col_types = cols(.default = "c"))
  data_test_raw <- read_csv(glue::glue("{csv_dir}/{benchmark}-test-seed={seed}.csv"),
                           col_types = cols(.default = "c"))

  # Extract row names from first column and convert data to numeric
  train_row_names <- data_train_raw[[1]]
  test_row_names <- data_test_raw[[1]]

  # Convert data columns to numeric (skip first column which contains row names)
  data_train <- data_train_raw[, -1] |>
    mutate(across(everything(), as.numeric)) |>
    as.data.frame()
  data_test <- data_test_raw[, -1] |>
    mutate(across(everything(), as.numeric)) |>
    as.data.frame()

  # Set row names
  rownames(data_train) <- train_row_names
  rownames(data_test) <- test_row_names

  # Load items metadata (keep as tibble to match original)
  items <- read_csv(glue::glue("{csv_dir}/{benchmark}-items-seed={seed}.csv"))

  # Remove readr-specific attributes to match original structure
  attr(items, "spec") <- NULL
  attr(items, "problems") <- NULL

  # Set class to match original (remove spec_tbl_df)
  class(items) <- c("tbl_df", "tbl", "data.frame")

  # Add names to sd, diff, disc columns to match original structure
  items$sd <- setNames(items$sd, as.character(items$item))
  items$diff <- setNames(items$diff, as.character(items$item))
  items$disc <- setNames(items$disc, as.character(items$item))

  # Load scores
  scores_df <- read_csv(glue::glue("{csv_dir}/{benchmark}-scores-seed={seed}.csv"))
  scores_train <- setNames(scores_df$score[scores_df$set == "train"],
                          scores_df$model[scores_df$set == "train"])
  scores_test <- setNames(scores_df$score[scores_df$set == "test"],
                         scores_df$model[scores_df$set == "test"])

  # Read max points from separate file
  max_points_orig <- as.integer(read_csv(glue::glue("{csv_dir}/{benchmark}-max_points-seed={seed}.csv"))$max_points[1])

  # Create R data structure
  r_data <- list(
    data.train = data_train,
    data.test = data_test,
    scores.train = scores_train,
    scores.test = scores_test,
    max.points.orig = max_points_orig,
    items = items
  )

  return(r_data)
}

# =============================================================================
# Main conversion
cat(glue::glue("🔄 Converting sample data for {benchmark} (seed={seed})...\n"))

# Use CSV conversion (reticulate not available)
cat("📁 Using CSV format conversion...\n")
r_data <- convert_csv_data(benchmark, seed)
cat("✅ Successfully converted from CSV format\n")

# Save as RDS file
output_path <- glue::glue("data/{benchmark}-preproc-split-seed={seed}.rds")
saveRDS(r_data, output_path)

cat(glue::glue("💾 Saved RDS file to {output_path}\n"))
cat(glue::glue("📊 Data summary:\n"))
cat(glue::glue("   Train models: {nrow(r_data$data.train)}\n"))
cat(glue::glue("   Test models: {nrow(r_data$data.test)}\n"))
cat(glue::glue("   Items: {ncol(r_data$data.train)}\n"))
cat(glue::glue("   Max points: {r_data$max.points.orig}\n"))

cat(glue::glue("\n✅ Conversion complete! You can now run:\n"))
cat(glue::glue("   Rscript random.R {benchmark} 350 {seed}\n"))
