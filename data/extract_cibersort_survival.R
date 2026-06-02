#!/usr/bin/env Rscript

# ============================================================
# Extract CIBERSORT compositional covariates + survival variables
# from the 8 TCGA-TIMER2 survival CSV files.
#
# Input:
#   data/processed/TCGA_<CANCER>_TIMER2_survival.csv
#
# Output:
#   data/cibersort/TCGA_<CANCER>_CIBERSORT_survival.csv
#   data/cibersort/TCGA_all8_CIBERSORT_survival.csv
#   data/cibersort/cibersort_sum_check_by_sample.csv
#   data/cibersort/cibersort_sum_check_summary.csv
#
# Run:
#   Rscript extract_cibersort_survival_check_sum.R
# ============================================================

options(stringsAsFactors = FALSE)

# ---------- 0. Settings ----------
projects <- c("BRCA", "BLCA", "COAD", "HNSC", "LIHC", "LUAD", "OV", "STAD")

input_dir <- "TCGA/processed"
output_dir <- "TCGA/cibersort"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# In TCGA-CDR, os/pfi/dfi/dss are event indicators:
#   1 = event occurred
#   0 = censored
# If you want explicit censoring indicators, this script also creates:
#   os_censor, pfi_censor, dfi_censor, dss_censor
# where:
#   1 = censored
#   0 = event occurred

surv_pairs <- list(
  os  = c(time = "os_time",  status = "os"),
  pfi = c(time = "pfi_time", status = "pfi"),
  dfi = c(time = "dfi_time", status = "dfi"),
  dss = c(time = "dss_time", status = "dss")
)

# Numerical tolerance for checking whether row sums are equal to 1.
tol <- 1e-6

# Whether to also save a closed/renormalized version of CIBERSORT covariates.
# Useful if row sums are close but not exactly 1 due to rounding.
save_closed_version <- TRUE

# ---------- 1. Packages ----------
install_if_missing <- function(pkgs) {
  missing <- pkgs[!vapply(pkgs, requireNamespace, FUN.VALUE = logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    install.packages(missing, repos = "https://cloud.r-project.org")
  }
}

install_if_missing(c("readr", "dplyr", "tibble"))
library(readr)
library(dplyr)
library(tibble)

# ---------- 2. Helper functions ----------
get_cibersort_cols <- function(dat) {
  # TIMER2 usually uses method names as suffixes after simple name cleaning,
  # e.g. b_cell_naive_cibersort, macrophage_m1_cibersort.
  grep("_cibersort$", names(dat), value = TRUE, ignore.case = TRUE)
}

get_surv_cols <- function(dat) {
  cols <- c()
  for (nm in names(surv_pairs)) {
    pair <- surv_pairs[[nm]]
    if (all(pair %in% names(dat))) {
      cols <- c(cols, pair["time"], pair["status"])
    }
  }
  unique(cols)
}

add_censor_indicators <- function(dat) {
  for (nm in names(surv_pairs)) {
    status_col <- surv_pairs[[nm]]["status"]
    censor_col <- paste0(nm, "_censor")

    if (status_col %in% names(dat)) {
      # status: 1 = event, 0 = censored
      # censor: 1 = censored, 0 = event
      dat[[censor_col]] <- ifelse(
        is.na(dat[[status_col]]),
        NA_integer_,
        ifelse(dat[[status_col]] == 0, 1L, 0L)
      )
    }
  }
  dat
}

check_cibersort_sum <- function(dat, cib_cols, dataset_name, tol = 1e-6) {
  X <- dat[, cib_cols, drop = FALSE]

  # Force numeric in case readr guesses some columns oddly.
  X[] <- lapply(X, function(z) suppressWarnings(as.numeric(z)))

  n_missing_parts <- rowSums(is.na(X))
  row_sum <- rowSums(X, na.rm = FALSE)

  tibble(
    dataset = dataset_name,
    patient = if ("patient" %in% names(dat)) dat$patient else NA_character_,
    sample_id = if ("sample_id" %in% names(dat)) dat$sample_id else NA_character_,
    n_cibersort_parts = length(cib_cols),
    n_missing_cibersort_parts = n_missing_parts,
    cibersort_sum = row_sum,
    abs_diff_from_1 = abs(row_sum - 1),
    is_complete_cibersort = n_missing_parts == 0,
    is_sum_1_tol = !is.na(row_sum) & abs(row_sum - 1) <= tol,
    is_sum_1_1e_4 = !is.na(row_sum) & abs(row_sum - 1) <= 1e-4,
    is_sum_1_1e_2 = !is.na(row_sum) & abs(row_sum - 1) <= 1e-2
  )
}

close_composition <- function(dat, cib_cols) {
  X <- dat[, cib_cols, drop = FALSE]
  X[] <- lapply(X, function(z) suppressWarnings(as.numeric(z)))

  # Replace exact zeros before log-ratio analysis if needed.
  # Here we only close the composition; zero replacement should be done
  # separately before CLR/ILR modeling.
  rs <- rowSums(X, na.rm = FALSE)
  X_closed <- sweep(X, 1, rs, FUN = "/")

  out <- dat
  out[, cib_cols] <- X_closed
  out
}

summarize_sum_check <- function(check_df) {
  check_df %>%
    group_by(dataset) %>%
    summarise(
      n_samples = n(),
      n_complete_cibersort = sum(is_complete_cibersort, na.rm = TRUE),
      n_sum_1_tol = sum(is_sum_1_tol, na.rm = TRUE),
      pct_sum_1_tol = round(100 * mean(is_sum_1_tol, na.rm = TRUE), 2),
      n_sum_1_1e_4 = sum(is_sum_1_1e_4, na.rm = TRUE),
      pct_sum_1_1e_4 = round(100 * mean(is_sum_1_1e_4, na.rm = TRUE), 2),
      n_sum_1_1e_2 = sum(is_sum_1_1e_2, na.rm = TRUE),
      pct_sum_1_1e_2 = round(100 * mean(is_sum_1_1e_2, na.rm = TRUE), 2),
      min_sum = min(cibersort_sum, na.rm = TRUE),
      q1_sum = quantile(cibersort_sum, 0.25, na.rm = TRUE),
      median_sum = median(cibersort_sum, na.rm = TRUE),
      mean_sum = mean(cibersort_sum, na.rm = TRUE),
      q3_sum = quantile(cibersort_sum, 0.75, na.rm = TRUE),
      max_sum = max(cibersort_sum, na.rm = TRUE),
      max_abs_diff_from_1 = max(abs_diff_from_1, na.rm = TRUE),
      .groups = "drop"
    )
}

# ---------- 3. Main extraction ----------
all_cibersort_data <- list()
all_sum_checks <- list()

for (p in projects) {
  dataset_name <- paste0("TCGA-", p)
  in_file <- file.path(input_dir, paste0("TCGA_", p, "_TIMER2_survival.csv"))

  if (!file.exists(in_file)) {
    warning("Input file does not exist, skip: ", in_file)
    next
  }

  message("Reading: ", in_file)
  dat <- readr::read_csv(in_file, show_col_types = FALSE, guess_max = 100000)

  cib_cols <- get_cibersort_cols(dat)
  if (length(cib_cols) == 0) {
    warning("No CIBERSORT columns found in: ", in_file)
    next
  }

  surv_cols <- get_surv_cols(dat)
  if (length(surv_cols) == 0) {
    warning("No survival endpoint columns found in: ", in_file)
    next
  }

  id_cols <- intersect(c("patient", "cancer_type", "sample_id", "sample_type"), names(dat))

  out <- dat %>%
    select(any_of(c(id_cols, surv_cols, cib_cols))) %>%
    add_censor_indicators() %>%
    mutate(dataset = dataset_name, .before = 1)

  # Save extracted CIBERSORT + survival data.
  out_file <- file.path(output_dir, paste0("TCGA_", p, "_CIBERSORT_survival.csv"))
  readr::write_csv(out, out_file, na = "")

  # Check whether the CIBERSORT parts sum to 1 for each sample.
  sum_check <- check_cibersort_sum(out, cib_cols, dataset_name, tol = tol)
  check_file <- file.path(output_dir, paste0("TCGA_", p, "_CIBERSORT_sum_check_by_sample.csv"))
  readr::write_csv(sum_check, check_file, na = "")

  # Optionally save a closed version where the CIBERSORT columns sum exactly to 1.
  if (save_closed_version) {
    out_closed <- close_composition(out, cib_cols)
    closed_file <- file.path(output_dir, paste0("TCGA_", p, "_CIBERSORT_survival_closed.csv"))
    readr::write_csv(out_closed, closed_file, na = "")
  }

  all_cibersort_data[[p]] <- out
  all_sum_checks[[p]] <- sum_check

  message("  CIBERSORT columns: ", length(cib_cols))
  message("  Saved: ", out_file)
}

# ---------- 4. Save combined outputs ----------
combined_dat <- bind_rows(all_cibersort_data)
combined_check <- bind_rows(all_sum_checks)
summary_check <- summarize_sum_check(combined_check)

readr::write_csv(
  combined_dat,
  file.path(output_dir, "TCGA_all8_CIBERSORT_survival.csv"),
  na = ""
)

readr::write_csv(
  combined_check,
  file.path(output_dir, "cibersort_sum_check_by_sample.csv"),
  na = ""
)

readr::write_csv(
  summary_check,
  file.path(output_dir, "cibersort_sum_check_summary.csv"),
  na = ""
)

# Save CIBERSORT variable names separately.
cibersort_vars <- grep("_cibersort$", names(combined_dat), value = TRUE, ignore.case = TRUE)
readr::write_csv(
  tibble(cibersort_variable = cibersort_vars),
  file.path(output_dir, "cibersort_variable_names.csv"),
  na = ""
)

message("\nDone. CIBERSORT files saved under: ", normalizePath(output_dir))
message("\nSummary of whether CIBERSORT row sums equal 1:")
print(summary_check)
