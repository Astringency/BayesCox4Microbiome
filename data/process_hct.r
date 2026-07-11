load("bacteria.RData")
load("gvhd.RData")

bacteria_df <- as.data.frame(bacteria)
gvhd_df <- as.data.frame(unclass(gvhd))
if (ncol(gvhd_df) == 2) {
  colnames(gvhd_df) <- c("time", "status")
}
rownames(bacteria_df) <- NULL
rownames(gvhd_df) <- NULL

# ---------------------------------------------------------
# Preparation: Separate ID columns and Raw Counts
# ---------------------------------------------------------
is_num <- sapply(bacteria_df, is.numeric)
bac_raw_num <- bacteria_df[, is_num, drop=FALSE]       # Pure raw read counts
bac_id  <- bacteria_df[, !is_num, drop=FALSE]          # Non-numeric columns (like sample_id)

# ---------------------------------------------------------
# NEW STEP: Transform raw counts to Compositional Data (Relative Abundance)
# ---------------------------------------------------------
# Calculate total reads per sample (row sums)
row_totals <- rowSums(bac_raw_num, na.rm = TRUE)

# Prevent division by zero just in case a sample has 0 total reads
row_totals[row_totals == 0] <- 1 

# Divide each element by its corresponding row total
bac_comp <- sweep(bac_raw_num, 1, row_totals, FUN = "/")

# ---------------------------------------------------------
# Task 1: Complete concatenated Compositional data (HCT.csv)
# ---------------------------------------------------------
HCT_full <- cbind(bac_id, bac_comp, gvhd_df)
write.csv(HCT_full, "HCT.csv", row.names = FALSE)
cat("Successfully generated: HCT.csv (Total columns: ", ncol(HCT_full), ")\n", sep="")

# ---------------------------------------------------------
# Task 2: Keep covariates with average abundance > 0.01 (HCT_001.csv)
# ---------------------------------------------------------
# mean_abundance <- as.vector(colMeans(bac_comp, na.rm = TRUE))
# keep_abund_idx <- mean_abundance > 0.01
#
# bac_001_recomp <- sweep(bac_comp[, keep_abund_idx, drop=FALSE], 1, rowSums(bac_comp[, keep_abund_idx, drop=FALSE]), FUN="/")
# HCT_001 <- cbind(bac_id, bac_001_recomp, gvhd_df)
#
# write.csv(HCT_001, "HCT_001.csv", row.names = FALSE)
# cat("Successfully generated: HCT_001.csv (Retained covariates: ", sum(keep_abund_idx), ")\n", sep="")

# ---------------------------------------------------------
# Task 3: Keep covariates with non-zero abundance in >= 5% of samples (HCT_005.csv)
# ---------------------------------------------------------
# prevalence <- as.vector(colMeans(bac_comp > 0, na.rm = TRUE))
# keep_prev_idx <- prevalence >= 0.05
#
# bac_005_recomp <- sweep(bac_comp[, keep_prev_idx, drop=FALSE], 1, rowSums(bac_comp[, keep_prev_idx, drop=FALSE]), FUN="/")
# HCT_005 <- cbind(bac_id, bac_005_recomp, gvhd_df)
#
# write.csv(HCT_005, "HCT_005.csv", row.names = FALSE)
# cat("Successfully generated: HCT_005.csv (Retained covariates: ", sum(keep_prev_idx), ")\n", sep="")

# ---------------------------------------------------------
# Task 4: BCor-SIS filter (HCT_bcorsis.csv)
# ---------------------------------------------------------
library(Ball)
fit.bcorsis <- bcorsis(x = bacteria, y = gvhd, method = "survival")
bac_bcorsis <- bacteria[, fit.bcorsis$ix]
bac_bcorsis_comp <- bac_bcorsis / rowSums(bac_bcorsis)
HCT_bacorsis <- cbind(bac_id, bac_bcorsis_comp, gvhd_df)
HCT_bacorsis_clean <- na.omit(HCT_bacorsis)


write.csv(HCT_bacorsis_clean, "HCT_bcorsis.csv", row.names = FALSE)
cat("Successfully generated: HCT_bcorsis.csv", sep="")



