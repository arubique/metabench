# Evaluate the performance of the IRT models using REDUCED (selected) items instead of random subsampling
# This script uses the items selected by reduce.R and evaluates RMSE between predicted and ground truth accuracy
#
# usage: Rscript evaluate.reduced.R {benchmark} {theta-method} {lambda} {seed}

# =============================================================================
# custom utils, args, path, seed
box::use(./utils[parse.args, gprint, gpath, rowmerge, mytheme, get.theta])
parse.args(
   names = c("BM", "METH", "LAMBDA", "seed"),
   defaults = c("arc", "MAP", 0.005, 0),
   legal = list(
     BM = c("arc", "gsm8k", "hellaswag", "mmlu", "truthfulqa", "winogrande"),
     METH = c("MAP", "EAPsum"),
     LAMBDA = seq(0, 1, 0.001)
   )
)
here::i_am("analysis/evaluate.reduced.R")
seed <- as.numeric(seed)
LAMBDA <- as.numeric(LAMBDA)
set.seed(seed)

# =============================================================================
# helper functions
cv.extract <- function(results, itemtype) {
   df <- results[[itemtype]]$df
   df$type <- itemtype
   rowmerge(df, leaderboard)
}

cv.collect <- function(results) {
  dfs <- lapply(names(results), function(itemtype) cv.extract(results, itemtype))
  names(dfs) <- names(results)
  dfs <- do.call(rbind, dfs)
  dfs$set <- factor(dfs$set, levels = c("train", "test"))
  dfs$error <- dfs$score - dfs$p
  dfs
}

fit.gam <- function(df.train){
  # get columns that start with F
  if ("F2" %in% colnames(df.train)){
    formula <- "score ~ s(F1, bs = 'ad') + s(F2, bs = 'ad') + s(sub, bs = 'ad')"
  } else {
    formula <- "score ~ s(F1, bs = 'ad') + s(sub, bs = 'ad')"
  }
  mgcv::gam(as.formula(formula), data = df.train)
}

refit <- function(result, data.train, data.test){
  # load data
  model <- result$model
  df <- result$df
  train <- df |> dplyr::filter(set == "train") |> dplyr::select(-F1)
  test <- df |> dplyr::filter(set == "test") |> dplyr::select(-F1)
  if ("SE_F1" %in% colnames(train)){
    train <- train |> dplyr::select(-SE_F1)
    test <- test |> dplyr::select(-SE_F1)
  }

  # refit theta
  theta.train <- get.theta(model, method = METH, resp = data.train)
  train <- cbind(train, theta.train[, 1, drop = F])
  theta.test <- get.theta(model, method = METH, resp = data.test)
  test <- cbind(test, theta.test[, 1, drop = F])

  # refit gam
  mod.score <- fit.gam(train)
  train$p <- predict(mod.score, train)
  test$p <- predict(mod.score, test)

  # export
  result$df <- rbind(train, test) |>
    dplyr::mutate(rank.theta = rank(F1),
                  perc.theta = rank.theta/max(rank.theta))
  result
}

refit.wrapper <- function(cvs, data.train, data.test){
  gprint("Refitting theta using {METH} on reduced item set...")
  cvs.re <- list()
  for (i in 1:length(cvs)){
    tryCatch({
      cvs.re[[i]] <- refit(cvs[[i]], data.train, data.test)
    }, error = function(e){
      print(e)
      gprint("Could not re-estimate theta for {names(cvs)[i]}")
    })
  }
  names(cvs.re) <- names(cvs)[1:length(cvs.re)]
  cvs.re[!sapply(cvs.re, is.null)]
}

evaluate.fit <- function(df.score) {
   out <-df.score |>
      dplyr::group_by(type, set) |>
      dplyr::summarize(
            rmse = sqrt(mean(error^2)),
            mae = mean(abs(error)),
            r = cor(p, score, method = "spearman"),
            r1 = cor(F1, score, method = "spearman"),
            r2 = ifelse("F2" %in% colnames(df.score),
                        cor(F2, score, method = "spearman"), NA),
            .groups = 'drop')
}

plot.theta.score <- function(df.score, itemtype){
   box::use(ggplot2[...], latex2exp[TeX])
   df.plot <- df.score |>
      dplyr::filter(set == "test", type == itemtype)
   sfs <- evaluate.fit(df.plot)
   text <- glue::glue(
     "r = {round(sfs$r, 3)}")
   x.label <- 0.8 * diff(range(df.plot$F1)) + min(df.plot$F1)
   y.label <- 0.1 * diff(range(df.plot$score)) + min(df.plot$score)
   ggplot(df.plot, aes(x = F1, y = score)) +
      geom_point(alpha = 0.5) +
      ylim(0,100) +
      annotate("text", x = x.label, y = y.label, label = text, size = 3) +
      labs(
         title = glue::glue("Theta vs. Score ({itemtype}) - Reduced Items"),
         x = TeX("$\\theta$"),
         y = "Score",
      ) +
      mytheme()
}

plot.score <- function(df.score, itemtype){
   box::use(ggplot2[...])
   df.plot <- df.score |>
      dplyr::filter(set == "test", type == itemtype)
   sfs <- evaluate.fit(df.plot)
   text <- glue::glue("RMSE = {round(sfs$rmse, 3)}")
   ggplot(df.plot, aes(x = score, y = p)) +
         geom_abline(intercept = 0,
                     slope = 1,
                     linetype = "dashed") +
         geom_point(alpha = 0.5) +
         coord_cartesian(xlim = c(0, 100), ylim = c(0, 100)) +
         annotate("text", x = 75, y = 25, label = text, size = 3) +
         labs(
            title = glue::glue("{BM} ({itemtype}-{METH}) - Reduced Items (λ={LAMBDA})"),
            x = "Ground Truth Score",
            y = "Predicted Score",
            ) +
         mytheme()
}

plot.error <- function(df.score, itemtype){
  box::use(ggplot2[...])
  df.plot <- df.score |>
    dplyr::filter(set == "test", type == itemtype) |>
    dplyr::mutate(ae = abs(error))
  ggplot(df.plot, aes(x = ae)) +
   # histogram
     geom_histogram(aes(x = ae), bins = 20, fill = "white", color = "black") +
      coord_cartesian(xlim = c(0, 100)) +
         labs(
            title = glue::glue("Error Distribution ({itemtype}) - Reduced Items"),
            x = "Absolute Error",
            y = "Frequency"
            ) +
         mytheme()
}

# =============================================================================
# Leaderboard
leaderboard <- read.csv(gpath("scraping/open-llm-leaderboard.csv"))
rownames(leaderboard) <- leaderboard$name
leaderboard <- leaderboard |> dplyr::select(size) |> dplyr::filter(size > 0)

# =============================================================================
# Load reduced item set results
gprint("🚰 Loading reduced item set results for {BM} (λ={LAMBDA}, seed={seed})...")

# Find the reduced model file
reduced_files <- list.files("analysis/reduced", pattern = glue::glue("{BM}-.*-{LAMBDA}-seed={seed}.rds"), full.names = TRUE)
if (length(reduced_files) == 0) {
  stop("No reduced model file found for {BM} with λ={LAMBDA} and seed={seed}")
}
reduced_file <- reduced_files[1]
gprint("📁 Using reduced model file: {basename(reduced_file)}")

# Load the reduced results
reduced_results <- readRDS(reduced_file)
gprint("📊 Reduced to {nrow(reduced_results$items)} items (from original {ncol(reduced_results$info.items.orig)-1} items)")

# Extract the reduced data
data.train.reduced <- reduced_results$theta.train
data.test.reduced <- reduced_results$theta.test
model.reduced <- reduced_results$model
items.reduced <- reduced_results$items

# Get the original data to extract scores
datapath <- gpath("data/{BM}-sub-350-seed={seed}.rds")
all <- readRDS(datapath)

# The reduced model uses a validation split, so we need to match the dimensions
# Check the dimensions
gprint("📏 Data dimensions:")
gprint("   Reduced train: {nrow(data.train.reduced)}")
gprint("   Reduced test: {nrow(data.test.reduced)}")
gprint("   Original train: {length(all$scores.train)}")
gprint("   Original test: {length(all$scores.test)}")

# The reduced model was trained on a subset (validation split removed)
# We need to use the same subset for fair comparison
# Use the reduced model's own score predictions from the results
if ("df.score.sub" %in% names(reduced_results)) {
  # Use the score predictions from the reduced model results
  df_reduced <- reduced_results$df.score.sub
  gprint("📊 Using score predictions from reduced model results")
  gprint("   Score predictions dimensions: {nrow(df_reduced)} rows, {ncol(df_reduced)} columns")
} else {
  stop("No df.score.sub found in reduced results - this is required for evaluation")
}

# Create a mock CV results structure to work with existing functions
cv_results <- list()
model_type <- "REDUCED"  # We'll use this as the model type name

# Use the pre-computed score predictions from reduce.R
df_combined <- df_reduced
# Rename 'theta' to 'F1' to match expected column names
if ("theta" %in% colnames(df_combined)) {
  df_combined <- df_combined |> dplyr::rename(F1 = theta)
}
gprint("📊 Using pre-computed score predictions from reduce.R")

# Create mock CV results
cv_results[[model_type]] <- list(
  model = model.reduced,
  df = df_combined
)

# =============================================================================
# Evaluate using reduced items
gprint("🔍 Evaluating performance with reduced item set...")

# Refit if needed (for EAPsum method)
if (METH == "EAPsum"){
  # We need the actual response data for EAPsum
  # Load the reduced response data
  datapath_reduced <- gpath("data/{BM}-sub-350-seed={seed}.rds")
  all_reduced <- readRDS(datapath_reduced)

  # Get the reduced item columns
  reduced_item_names <- as.character(items.reduced$item)
  data.train.responses <- all_reduced$data.train[, reduced_item_names]
  data.test.responses <- all_reduced$data.test[, reduced_item_names]

  cv_results <- refit.wrapper(cv_results, data.train.responses, data.test.responses)
}

# Collect results
df.score <- cv.collect(cv_results)

# Evaluate
sfs <- evaluate.fit(df.score)
gprint("📈 Performance Results:")
print(sfs)

# =============================================================================
# Create plots
gprint("📊 Creating evaluation plots...")

# Main prediction plot
p_main <- plot.score(df.score, model_type)
p_theta <- plot.theta.score(df.score, model_type)
p_error <- plot.error(df.score, model_type)

# Combine plots
p_combined <- cowplot::plot_grid(
  p_main, p_theta, p_error,
  nrow = 1
)

# Save plots
outpath <- gpath("plots/{BM}-{METH}-reduced-{LAMBDA}-seed={seed}.png")
# Set graphics device to avoid X11 issues
options(bitmapType='cairo')
ggplot2::ggsave(outpath, p_combined, width = 16, height = 6, device = "png")
gprint("💾 Saved plot to {outpath}")

# Save results
results_outpath <- gpath("plots/{BM}-{METH}-reduced-{LAMBDA}-seed={seed}.rds")
saveRDS(list(plot = p_combined, results = sfs, df = df.score), results_outpath)
gprint("💾 Saved results to {results_outpath}")

# =============================================================================
# Summary
gprint("🎉 Evaluation complete!")
gprint("📊 Summary:")
gprint("   Benchmark: {BM}")
gprint("   Method: {METH}")
gprint("   Lambda: {LAMBDA}")
gprint("   Seed: {seed}")
gprint("   Reduced items: {nrow(items.reduced)}")
gprint("   Test RMSE: {round(sfs$rmse[sfs$set == 'test'], 3)}")
gprint("   Test Correlation: {round(sfs$r[sfs$set == 'test'], 3)}")
