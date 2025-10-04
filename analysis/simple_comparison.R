# Simple comparison of RMSE results
# Shows the difference between random subsampling and selected items

# =============================================================================
# custom utils
box::use(./utils[gprint])
here::i_am("analysis/simple_comparison.R")

# =============================================================================
# Results from our evaluations
gprint("📊 RMSE COMPARISON: Random Subsampling vs Selected Items")
gprint("========================================================")
gprint("")
gprint("🎲 Random Subsampling (350 items, 4PL model):")
gprint("   Test RMSE: 0.984")
gprint("")
gprint("🎯 Selected Items (108 items, λ=0.005, 4PL model):")
gprint("   Test RMSE: 2.334")
gprint("")
gprint("📈 ANALYSIS:")
gprint("   ❌ Selected items perform WORSE by 1.350 RMSE points")
gprint("   📉 Relative degradation: 137.2%")
gprint("")
gprint("🤔 POSSIBLE EXPLANATIONS:")
gprint("   1. The reduced item set (108 items) may be too small")
gprint("   2. The lambda penalty (0.005) may be too high, selecting too few items")
gprint("   3. The item selection criteria may need tuning")
gprint("   4. Different model configurations between evaluations")
gprint("")
gprint("💡 RECOMMENDATIONS:")
gprint("   1. Try different lambda values (e.g., 0.001, 0.01)")
gprint("   2. Try different model types (2PL, 3PL)")
gprint("   3. Compare with the same number of items (350)")
gprint("   4. Check if the reduced model was properly trained")
gprint("")
gprint("========================================================")
