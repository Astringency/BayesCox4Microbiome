#!/usr/bin/env Rscript

# ============================================================
# Download and merge TCGA-CDR survival endpoints with TIMER2
# immune-composition covariates, then save CSV files.
#
# Output:
#   data/raw/TCGA-CDR-SupplementalTableS1.xlsx
#   data/raw/timer2_infiltration_estimation_for_tcga.csv.gz
#   data/processed/TCGA_<CANCER>_TIMER2_survival.csv
#   data/processed/TCGA_all8_TIMER2_survival.csv
#   data/processed/download_summary.csv
#
# Run:
#   Rscript download_tcga_timer_survival.R
# ============================================================

options(stringsAsFactors = FALSE)

# ---------- 0. User settings ----------
projects <- c("BRCA", "BLCA", "COAD", "HNSC", "LIHC", "LUAD", "OV", "STAD")

base_dir <- "data"
raw_dir <- file.path(base_dir, "raw")
processed_dir <- file.path(base_dir, "processed")
dir.create(raw_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(processed_dir, recursive = TRUE, showWarnings = FALSE)

timer_url <- "https://compbio.cn/timer2/infiltration_estimation_for_tcga.csv.gz"
gdc_publication_url <- "https://gdc.cancer.gov/about-data/publications/PanCan-Clinical-2018"

timer_file <- file.path(raw_dir, "timer2_infiltration_estimation_for_tcga.csv.gz")
cdr_file <- file.path(raw_dir, "TCGA-CDR-SupplementalTableS1.xlsx")

# If automatic scraping of the GDC page fails, set this environment variable manually:
# Sys.setenv(TCGA_CDR_XLSX_URL = "https://api.gdc.cancer.gov/data/<file_uuid>")
cdr_url_env <- Sys.getenv("TCGA_CDR_XLSX_URL", unset = "")

# ---------- 1. Install/load packages ----------
install_if_missing <- function(pkgs) {
  missing <- pkgs[!vapply(pkgs, requireNamespace, FUN.VALUE = logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    message("Installing missing packages: ", paste(missing, collapse = ", "))
    install.packages(missing, repos = "https://cloud.r-project.org")
  }
}

cran_pkgs <- c("curl", "readr", "readxl", "dplyr", "stringr", "xml2", "rvest", "tibble")
install_if_missing(cran_pkgs)

library(curl)
library(readr)
library(readxl)
library(dplyr)
library(stringr)
library(xml2)
library(rvest)
library(tibble)

# ---------- 2. Helper functions ----------
clean_names_simple <- function(x) {
  x <- iconv(x, from = "", to = "ASCII//TRANSLIT", sub = "")
  x <- tolower(x)
  x <- gsub("[^a-z0-9]+", "_", x)
  x <- gsub("^_+|_+$", "", x)
  x <- make.unique(x, sep = "_")
  x
}

download_if_missing <- function(url, dest) {
  if (!file.exists(dest) || file.info(dest)$size == 0) {
    message("Downloading: ", url)
    curl::curl_download(url, destfile = dest, mode = "wb", quiet = FALSE)
  } else {
    message("File already exists, skip download: ", dest)
  }
  invisible(dest)
}

get_cdr_download_url <- function(publication_url, env_url = "") {
  if (nzchar(env_url)) {
    message("Using TCGA_CDR_XLSX_URL from environment.")
    return(env_url)
  }

  message("Finding TCGA-CDR XLSX link from GDC publication page...")
  page <- xml2::read_html(publication_url)
  links <- rvest::html_elements(page, "a")
  link_text <- rvest::html_text2(links)
  href <- rvest::html_attr(links, "href")

  idx <- which(
    grepl("TCGA-CDR-SupplementalTableS1.xlsx", link_text, fixed = TRUE) |
      grepl("TCGA-CDR-SupplementalTableS1.xlsx", href, fixed = TRUE)
  )

  if (length(idx) == 0) {
    stop(
      "Could not find TCGA-CDR-SupplementalTableS1.xlsx on the GDC page. ",
      "Open the GDC publication page manually and set TCGA_CDR_XLSX_URL to the XLSX download URL."
    )
  }

  xml2::url_absolute(href[idx[1]], publication_url)
}

detect_tcga_sample_col <- function(df) {
  score <- vapply(df, function(x) {
    x <- as.character(x)
    mean(grepl("^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", x), na.rm = TRUE)
  }, numeric(1))

  if (all(is.na(score)) || max(score, na.rm = TRUE) <= 0) {
    stop("Could not detect a TIMER2 column containing TCGA sample barcodes.")
  }

  names(score)[which.max(score)]
}

coalesce_existing <- function(df, candidates) {
  candidates <- candidates[candidates %in% names(df)]
  if (length(candidates) == 0) {
    return(rep(NA, nrow(df)))
  }

  out <- df[[candidates[1]]]
  if (length(candidates) > 1) {
    for (nm in candidates[-1]) {
      out <- dplyr::coalesce(out, df[[nm]])
    }
  }
  out
}

as_event01 <- function(x) {
  if (is.numeric(x)) return(as.integer(x))
  x_chr <- tolower(trimws(as.character(x)))
  dplyr::case_when(
    x_chr %in% c("1", "dead", "deceased", "yes", "true", "event") ~ 1L,
    x_chr %in% c("0", "alive", "no", "false", "censored") ~ 0L,
    TRUE ~ suppressWarnings(as.integer(x_chr))
  )
}

pick_endpoint_cols <- function(df) {
  endpoint_patterns <- c(
    "^os$", "^os_time$", "^dss$", "^dss_time$",
    "^dfi$", "^dfi_time$", "^pfi$", "^pfi_time$"
  )
  grep(paste(endpoint_patterns, collapse = "|"), names(df), value = TRUE)
}

# ---------- 3. Download source files ----------
cdr_url <- get_cdr_download_url(gdc_publication_url, cdr_url_env)
download_if_missing(cdr_url, cdr_file)
download_if_missing(timer_url, timer_file)

# ---------- 4. Read TCGA-CDR survival endpoints ----------
sheets <- readxl::excel_sheets(cdr_file)
cdr_sheet <- sheets[grepl("TCGA-CDR", sheets, ignore.case = TRUE)][1]
if (is.na(cdr_sheet)) cdr_sheet <- sheets[1]

message("Reading TCGA-CDR sheet: ", cdr_sheet)
cdr_raw <- readxl::read_xlsx(cdr_file, sheet = cdr_sheet)
names(cdr_raw) <- clean_names_simple(names(cdr_raw))

patient_id_candidates <- c(
  "bcr_patient_barcode", "patient", "patient_id",
  "submitter_id", "case_submitter_id", "sample"
)

type_candidates <- c("type", "cancer_type", "project", "project_id", "study")

cdr <- cdr_raw %>%
  mutate(
    patient = coalesce_existing(., patient_id_candidates),
    cancer_type = coalesce_existing(., type_candidates)
  ) %>%
  mutate(
    patient = substr(as.character(patient), 1, 12),
    cancer_type = gsub("^TCGA[-_]", "", toupper(as.character(cancer_type)))
  )

if (all(is.na(cdr$patient))) {
  stop("Could not identify patient IDs in TCGA-CDR. Check names(cdr_raw).")
}
if (all(is.na(cdr$cancer_type))) {
  stop("Could not identify cancer type in TCGA-CDR. Check names(cdr_raw).")
}

# Standardize endpoint event columns where present.
for (ev in c("os", "dss", "dfi", "pfi")) {
  if (ev %in% names(cdr)) cdr[[ev]] <- as_event01(cdr[[ev]])
}

# ---------- 5. Read TIMER2 immune composition estimates ----------
message("Reading TIMER2 infiltration matrix...")
timer_raw <- readr::read_csv(timer_file, show_col_types = FALSE, guess_max = 100000)

sample_col_raw <- detect_tcga_sample_col(timer_raw)
message("Detected TIMER2 sample column: ", sample_col_raw)

timer <- timer_raw %>%
  rename(sample_id = all_of(sample_col_raw))

names(timer) <- clean_names_simple(names(timer))

timer <- timer %>%
  mutate(
    sample_id = as.character(sample_id),
    patient = substr(sample_id, 1, 12),
    sample_type = ifelse(nchar(sample_id) >= 15, substr(sample_id, 14, 15), NA_character_)
  ) %>%
  filter(grepl("^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", patient)) %>%
  # Keep primary tumor samples when full TCGA sample barcode is available.
  filter(is.na(sample_type) | sample_type == "01")

# Make one row per patient. If multiple tumor aliquots exist, keep the first row.
timer_patient <- timer %>%
  arrange(patient, sample_id) %>%
  group_by(patient) %>%
  slice(1) %>%
  ungroup()

# Try to tag numeric TIMER2 variables.
id_like_cols <- c("sample_id", "patient", "sample_type")
non_component_patterns <- c("cancer", "type", "project", "barcode", "sample", "patient")
numeric_cols <- names(timer_patient)[vapply(timer_patient, is.numeric, logical(1))]
component_cols <- setdiff(
  numeric_cols,
  grep(paste(non_component_patterns, collapse = "|"), numeric_cols, value = TRUE, ignore.case = TRUE)
)

# Recommended component families for compositional Cox modeling.
recommended_component_cols <- grep(
  "cibersort|quantiseq|epic",
  component_cols,
  value = TRUE,
  ignore.case = TRUE
)

message("Numeric TIMER2 estimate columns detected: ", length(component_cols))
message("Recommended CIBERSORT/quanTIseq/EPIC-like columns detected: ", length(recommended_component_cols))

# ---------- 6. Merge and write CSV files ----------
endpoint_cols <- pick_endpoint_cols(cdr)

basic_clinical_candidates <- c(
  "age_at_initial_pathologic_diagnosis", "age_at_diagnosis", "age",
  "gender", "sex", "race", "ethnicity",
  "ajcc_pathologic_stage", "pathologic_stage", "clinical_stage",
  "tumor_stage", "stage"
)
basic_clinical_cols <- intersect(basic_clinical_candidates, names(cdr))

keep_cdr_cols <- unique(c(
  "patient", "cancer_type",
  endpoint_cols,
  basic_clinical_cols
))

cdr_keep <- cdr %>%
  select(any_of(keep_cdr_cols)) %>%
  distinct(patient, .keep_all = TRUE)

all_outputs <- list()
summary_rows <- list()

for (p in projects) {
  message("Processing TCGA-", p)

  one <- cdr_keep %>%
    filter(cancer_type == p) %>%
    inner_join(timer_patient, by = "patient", suffix = c("_cdr", "_timer2"))

  # Put identifiers and survival variables first.
  front_cols <- intersect(
    c("patient", "cancer_type", "sample_id", "sample_type",
      "os_time", "os", "pfi_time", "pfi", "dfi_time", "dfi", "dss_time", "dss"),
    names(one)
  )
  one <- one %>% select(any_of(front_cols), everything())

  out_file <- file.path(processed_dir, paste0("TCGA_", p, "_TIMER2_survival.csv"))
  readr::write_csv(one, out_file, na = "")

  all_outputs[[p]] <- one %>% mutate(dataset = paste0("TCGA-", p), .before = 1)

  summary_rows[[p]] <- tibble(
    dataset = paste0("TCGA-", p),
    n_rows = nrow(one),
    os_events = if ("os" %in% names(one)) sum(one$os == 1, na.rm = TRUE) else NA_integer_,
    pfi_events = if ("pfi" %in% names(one)) sum(one$pfi == 1, na.rm = TRUE) else NA_integer_,
    dfi_events = if ("dfi" %in% names(one)) sum(one$dfi == 1, na.rm = TRUE) else NA_integer_,
    dss_events = if ("dss" %in% names(one)) sum(one$dss == 1, na.rm = TRUE) else NA_integer_,
    output_file = out_file
  )
}

all_df <- bind_rows(all_outputs)
summary_df <- bind_rows(summary_rows)

readr::write_csv(all_df, file.path(processed_dir, "TCGA_all8_TIMER2_survival.csv"), na = "")
readr::write_csv(summary_df, file.path(processed_dir, "download_summary.csv"), na = "")

# Also save detected component columns for downstream CLR/ILR transformation.
component_info <- tibble(
  column = component_cols,
  recommended_for_coda = column %in% recommended_component_cols
)
readr::write_csv(component_info, file.path(processed_dir, "timer2_component_columns.csv"), na = "")

message("\nDone. Files saved under: ", normalizePath(processed_dir))
print(summary_df)
