#!/usr/bin/env Rscript

###############################################################################
# Publication-quality MCMC diagnostic plots for BayesCox simulation runs.
#
# Generates, for each discovered scenario and method:
#   Figure 1a: trace + density + ACF, one nonzero coefficient per row
#   Figure 1b: R-hat + bulk ESS + tail ESS for all coefficients in one fit
#   Figure 3: Coefficient-sum distribution across available replicates
#   Figure 4: Coefficient-sum distribution across posterior samples for one replicate
#
# Usage:
#   Rscript scripts/plot_simulation.R --root results/simulation/prox
#   Rscript scripts/plot_simulation.R --roots results/simulation/prox,results/simulation/soft_center
###############################################################################

.libPaths(c("~/R/library", .libPaths()))

args <- commandArgs(trailingOnly = TRUE)

arg_value <- function(name, default = NULL) {
  key <- paste0("--", name)
  eq_prefix <- paste0(key, "=")
  eq_idx <- which(startsWith(args, eq_prefix))
  if (length(eq_idx) > 0) {
    return(substring(args[[eq_idx[[1]]]], nchar(eq_prefix) + 1L))
  }
  idx <- match(key, args)
  if (is.na(idx) || idx == length(args)) {
    return(default)
  }
  args[[idx + 1]]
}

arg_flag <- function(name, default = FALSE) {
  key <- paste0("--", name)
  if (key %in% args) {
    return(TRUE)
  }
  value <- arg_value(name, NULL)
  if (is.null(value)) {
    return(default)
  }
  tolower(value) %in% c("1", "true", "yes", "y")
}

split_csv <- function(value) {
  if (is.null(value) || !nzchar(value)) {
    return(character())
  }
  trimws(strsplit(value, ",", fixed = TRUE)[[1]])
}

usage <- function() {
  cat(
    "Usage:\n",
    "  Rscript scripts/plot_simulation.R --root ROOT [options]\n",
    "  Rscript scripts/plot_simulation.R --roots ROOT1,ROOT2 [options]\n\n",
    "Options:\n",
    "  --root ROOT              One simulation result root.\n",
    "  --roots ROOT1,ROOT2      Comma-separated simulation result roots.\n",
    "  --output-dir DIR         Override output directory for a single root.\n",
    "  --rep N|all              Replicate to plot (default 1), or every saved fit.\n",
    "  --manifest CSV           Audited diagnostics_by_run.csv, including split batches.\n",
    "  --suite NAME             Optional manifest suite filter (main/supplementary).\n",
    "  --cells A,B              Optional exact manifest cell IDs.\n",
    "  --shard I/N              Optional disjoint cell shard for batch rendering.\n",
    "  --sum-reps N             Maximum replicate count for Figure 3 coefficient sums. Default: 50.\n",
    "  --methods m1,m2          Optional method filter.\n",
    "  --scenarios s1,s2        Optional scenario directory/name filter.\n",
    "  --acf-max-lag N          Maximum ACF lag. Default: 50.\n",
    "  R-hat reference lines are fixed at 1.01 and 1.1; ESS at 400.\n",
    "  --dry-run                Print discovered work without writing plots.\n",
    "  --help                   Show this message.\n",
    sep = ""
  )
}

if (arg_flag("help")) {
  usage()
  quit(status = 0)
}

needed <- c("ggplot2", "data.table", "patchwork")
missing_pkgs <- needed[!vapply(needed, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  stop("Missing R packages: ", paste(missing_pkgs, collapse = ", "), call. = FALSE)
}

suppressPackageStartupMessages({
  library(ggplot2)
  library(data.table)
  library(patchwork)
})

DEFAULT_ROOT <- file.path("results", "simulation")
ROOTS <- split_csv(arg_value("roots", arg_value("root", DEFAULT_ROOT)))
if (length(ROOTS) == 0) {
  ROOTS <- DEFAULT_ROOT
}

REP_ARG <- arg_value("rep", "1")
ALL_REPS <- identical(REP_ARG,"all")
FOCUS_REP <- if(ALL_REPS) 1L else as.integer(REP_ARG)
MANIFEST <- arg_value("manifest", NULL)
SUM_REPS <- as.integer(arg_value("sum-reps", "50"))
ACF_MAX_LAG <- as.integer(arg_value("acf-max-lag", "50"))
METHOD_FILTER <- split_csv(arg_value("methods", ""))
SCENARIO_FILTER <- split_csv(arg_value("scenarios", ""))
OUTPUT_DIR_OVERRIDE <- arg_value("output-dir", NULL)
DRY_RUN <- arg_flag("dry-run")

if (is.na(FOCUS_REP) || FOCUS_REP < 1) {
  stop("--rep must be a positive integer.", call. = FALSE)
}
if (is.na(SUM_REPS) || SUM_REPS < 1) {
  stop("--sum-reps must be a positive integer.", call. = FALSE)
}
if (is.na(ACF_MAX_LAG) || ACF_MAX_LAG < 1) {
  stop("--acf-max-lag must be a positive integer.", call. = FALSE)
}
if (is.null(MANIFEST) && !is.null(OUTPUT_DIR_OVERRIDE) && length(ROOTS) != 1) {
  stop("--output-dir can only be used with one --root.", call. = FALSE)
}

scenario_dir <- function(n, p, nz, baseline) {
  sprintf("n%d_p%d_nonzero%d_%s", n, p, nz, baseline)
}

scenario_label <- function(n, p, nz, baseline) {
  sprintf("n%d_p%d_nz%d_%s", n, p, nz, baseline)
}

parse_scenario_dir <- function(path) {
  name <- basename(path)
  m <- regexec("^n([0-9]+)_p([0-9]+)_nonzero([0-9]+)_(.+)$", name)
  parts <- regmatches(name, m)[[1]]
  if (length(parts) == 0) {
    return(NULL)
  }
  data.table(
    scenario_dir = name,
    n = as.integer(parts[[2]]),
    p = as.integer(parts[[3]]),
    nonzero = as.integer(parts[[4]]),
    baseline = parts[[5]],
    scenario_label = scenario_label(
      as.integer(parts[[2]]),
      as.integer(parts[[3]]),
      as.integer(parts[[4]]),
      parts[[5]]
    ),
    scenario_path = normalizePath(path, mustWork = FALSE)
  )
}

parse_run_dir <- function(path) {
  name <- basename(path)
  m <- regexec("^rep([0-9]+)_(.+)_([^_]+)_n([0-9]+)_p([0-9]+)_nz([0-9]+)$", name)
  parts <- regmatches(name, m)[[1]]
  if (length(parts) == 0) {
    return(NULL)
  }
  data.table(
    run_dir_name = name,
    rep = as.integer(parts[[2]]),
    method = parts[[3]],
    baseline = parts[[4]],
    n = as.integer(parts[[5]]),
    p = as.integer(parts[[6]]),
    nonzero = as.integer(parts[[7]]),
    run_path = normalizePath(path, mustWork = FALSE)
  )
}

has_scenario_dirs <- function(root) {
  if (!dir.exists(root)) {
    return(FALSE)
  }
  entries <- list.dirs(root, full.names = FALSE, recursive = FALSE)
  any(grepl("^n[0-9]+_p[0-9]+_nonzero[0-9]+_.+", entries))
}

expand_roots <- function(root) {
  if (!dir.exists(root)) {
    warning("Root does not exist: ", root, call. = FALSE)
    return(character())
  }
  if (has_scenario_dirs(root)) {
    return(normalizePath(root, mustWork = FALSE))
  }
  children <- list.dirs(root, full.names = TRUE, recursive = FALSE)
  children <- children[vapply(children, has_scenario_dirs, logical(1))]
  if (length(children) > 0) {
    return(normalizePath(children, mustWork = FALSE))
  }
  normalizePath(root, mustWork = FALSE)
}

discover_work <- function(data_root) {
  scenario_paths <- list.dirs(data_root, full.names = TRUE, recursive = FALSE)
  scenario_info <- rbindlist(lapply(scenario_paths, parse_scenario_dir), fill = TRUE)
  if (nrow(scenario_info) == 0) {
    return(data.table())
  }

  if (length(SCENARIO_FILTER) > 0) {
    scenario_info <- scenario_info[
      scenario_dir %in% SCENARIO_FILTER | scenario_label %in% SCENARIO_FILTER
    ]
  }
  if (nrow(scenario_info) == 0) {
    return(data.table())
  }

  run_info <- rbindlist(lapply(seq_len(nrow(scenario_info)), function(i) {
    sc <- scenario_info[i]
    runs_dir <- file.path(sc$scenario_path, "runs")
    run_paths <- if (dir.exists(runs_dir)) {
      list.dirs(runs_dir, full.names = TRUE, recursive = FALSE)
    } else {
      character()
    }
    parsed <- rbindlist(lapply(run_paths, parse_run_dir), fill = TRUE)
    if (nrow(parsed) == 0) {
      return(data.table())
    }
    parsed <- parsed[
      rep == FOCUS_REP &
        n == sc$n &
        p == sc$p &
        nonzero == sc$nonzero &
        baseline == sc$baseline
    ]
    if (nrow(parsed) == 0) {
      return(data.table())
    }
    unique(parsed[, .(
      scenario_dir = sc$scenario_dir,
      scenario_label = sc$scenario_label,
      scenario_path = sc$scenario_path,
      n = sc$n,
      p = sc$p,
      nonzero = sc$nonzero,
      baseline = sc$baseline,
      method
    )])
  }), fill = TRUE)

  if (nrow(run_info) == 0) {
    return(data.table())
  }
  run_info <- run_info[!is.na(method)]

  if (length(METHOD_FILTER) > 0) {
    run_info <- run_info[method %in% METHOD_FILTER]
  }
  setorder(run_info, n, p, nonzero, baseline, method)
  unique(run_info)
}

run_label <- function(method, baseline, n, p, nz) {
  sprintf("%s_%s_n%d_p%d_nz%d", method, baseline, n, p, nz)
}

run_dir <- function(data_root, sc_dir, method, baseline, n, p, nz, rep_id) {
  file.path(
    data_root,
    sc_dir,
    "runs",
    sprintf("rep%d_%s", rep_id, run_label(method, baseline, n, p, nz))
  )
}

load_diagnostics <- function(data_root, sc_dir, method, baseline, n, p, nz, rep_id) {
  run <- run_dir(data_root, sc_dir, method, baseline, n, p, nz, rep_id)
  candidates <- file.path(run,c("rank_diagnostics.csv","diagnose/diagnostics_summary.csv"))
  diag_path <- candidates[file.exists(candidates)][1]
  if(is.na(diag_path)) stop("No saved coefficient diagnostics: ",run)
  if (!file.exists(diag_path)) {
    stop(sprintf("Diagnostics file not found: %s", diag_path))
  }

  diag <- fread(diag_path, select = c("parameters", "rhat", "ess_bulk", "ess_tail"))
  diag <- diag[grepl("^beta\\[", parameters)]
  diag[, p_dim := p]
  diag
}

normalize_posterior <- function(post, run) {
  if(!"variable" %in% names(post) && "parameters" %in% names(post))
    setnames(post,"parameters","variable")
  truth_file <- file.path(run,"coefficient_audit.csv")
  if(!"true_beta" %in% names(post) && file.exists(truth_file)) {
    truth <- fread(truth_file)
    stopifnot(all(c("taxon","true_beta") %in% names(truth)))
    truth[,variable:=paste0("beta[",taxon,"]")]
    stopifnot(!anyDuplicated(truth$variable))
    post[,true_beta:=truth$true_beta[match(variable,truth$variable)]]
  }
  post
}

load_posterior <- function(data_root, sc_dir, method, baseline, n, p, nz, rep_id) {
  post_path <- file.path(
    run_dir(data_root, sc_dir, method, baseline, n, p, nz, rep_id),
    "posterior_summary.csv"
  )
  if (!file.exists(post_path)) {
    stop(sprintf("Posterior file not found: %s", post_path))
  }
  normalize_posterior(fread(post_path),dirname(post_path))
}

load_samples <- function(data_root, sc_dir, method, baseline, n, p, nz, rep_id, nonzero_idx) {
  samples_path <- file.path(
    run_dir(data_root, sc_dir, method, baseline, n, p, nz, rep_id),
    "samples_wide.csv"
  )
  if (!file.exists(samples_path)) {
    stop(sprintf("Samples file not found: %s", samples_path))
  }

  beta_cols <- paste0("beta[", nonzero_idx, "]")
  select_cols <- c("chain", "iter", intersect("draw",names(fread(samples_path,nrows=0))), beta_cols)
  samples <- fread(samples_path, select = select_cols)
  melt(
    samples,
    id.vars = intersect(c("chain", "iter", "draw"),names(samples)),
    measure.vars = beta_cols,
    variable.name = "parameter",
    value.name = "value"
  )
}

compute_acf <- function(samples, max_lag = ACF_MAX_LAG) {
  params <- unique(samples$parameter)
  chains <- unique(samples$chain)
  acf_list <- vector("list", length(params) * length(chains))
  idx <- 1

  for (ch in chains) {
    for (par_name in params) {
      vals <- samples[chain == ch & parameter == par_name][order(iter)]$value
      acf_obj <- acf(vals, lag.max = max_lag, plot = FALSE)
      acf_list[[idx]] <- data.frame(
        chain = ch,
        parameter = par_name,
        lag = as.integer(acf_obj$lag),
        acf = as.numeric(acf_obj$acf)
      )
      idx <- idx + 1
    }
  }

  rbindlist(acf_list)
}

beta_row_filter <- function(df) {
  name_col <- intersect(c("variable", "parameter", "parameters", "term", "name"), names(df))
  if (length(name_col) == 0) {
    rep(TRUE, nrow(df))
  } else {
    grepl("^beta\\[", as.character(df[[name_col[[1]]]]))
  }
}

numeric_vector <- function(x) {
  if (is.numeric(x)) {
    return(x)
  }
  suppressWarnings(as.numeric(as.character(x)))
}

sum_from_coef_table <- function(path) {
  if (!file.exists(path)) {
    return(NA_real_)
  }
  coef_data <- fread(path)
  if (!("mean" %in% names(coef_data))) {
    return(NA_real_)
  }
  coef_data <- coef_data[beta_row_filter(coef_data)]
  if (nrow(coef_data) == 0) {
    return(NA_real_)
  }
  sum(numeric_vector(coef_data$mean), na.rm = TRUE)
}

sum_from_posterior <- function(path) {
  if (!file.exists(path)) {
    return(NA_real_)
  }
  post <- fread(path)
  if (!("mean" %in% names(post))) {
    return(NA_real_)
  }
  post <- post[beta_row_filter(post)]
  if (nrow(post) == 0) {
    return(NA_real_)
  }
  sum(numeric_vector(post$mean), na.rm = TRUE)
}

load_replicate_sums <- function(data_root, sc_dir, method, baseline, n, p, nz, max_reps = SUM_REPS) {
  values <- rep(NA_real_, max_reps)
  rlbl <- run_label(method, baseline, n, p, nz)
  tables_dir <- file.path(data_root, sc_dir, "tables")

  for (rep_id in seq_len(max_reps)) {
    coef_file <- file.path(tables_dir, sprintf("coef_rep%d_%s.csv", rep_id, rlbl))
    value <- sum_from_coef_table(coef_file)
    if (is.na(value)) {
      post_file <- file.path(
        run_dir(data_root, sc_dir, method, baseline, n, p, nz, rep_id),
        "posterior_summary.csv"
      )
      value <- sum_from_posterior(post_file)
    }
    values[[rep_id]] <- value
  }

  values[!is.na(values)]
}

load_sample_sums <- function(data_root, sc_dir, method, baseline, n, p, nz, rep_id = FOCUS_REP) {
  samples_path <- file.path(
    run_dir(data_root, sc_dir, method, baseline, n, p, nz, rep_id),
    "samples_wide.csv"
  )
  if (!file.exists(samples_path)) {
    stop(sprintf("Samples file not found: %s", samples_path))
  }

  beta_cols <- paste0("beta[", seq_len(p), "]")
  header <- names(fread(samples_path, nrows = 0))
  missing_cols <- setdiff(beta_cols, header)
  if (length(missing_cols) > 0) {
    stop(
      sprintf(
        "Samples file is missing beta columns: %s",
        paste(missing_cols, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  samples <- fread(samples_path, select = beta_cols)
  samples[, (beta_cols) := lapply(.SD, numeric_vector), .SDcols = beta_cols]
  sum_values <- rowSums(as.matrix(samples[, ..beta_cols]))
  sum_values[is.finite(sum_values)]
}

as_bool <- function(x) {
  if (is.logical(x)) {
    return(x)
  }
  if (is.numeric(x)) {
    return(!is.na(x) & x != 0)
  }
  tolower(trimws(as.character(x))) %in% c("true", "t", "1", "yes", "y")
}

parameter_index <- function(x) {
  m <- regexec("^beta\\[([0-9]+)\\]$", as.character(x))
  parts <- regmatches(as.character(x), m)
  vapply(parts, function(part) {
    if (length(part) == 0) {
      return(NA_integer_)
    }
    as.integer(part[[2]])
  }, integer(1))
}

posterior_nonzero <- function(posterior) {
  if (!("variable" %in% names(posterior))) {
    return(posterior[0])
  }
  is_beta <- grepl("^beta\\[", posterior$variable)
  if ("true_beta" %in% names(posterior)) {
    true_beta <- numeric_vector(posterior$true_beta)
    keep <- is_beta & is.finite(true_beta) & true_beta != 0
  } else if ("nonzero" %in% names(posterior)) {
    keep <- is_beta & as_bool(posterior$nonzero)
  } else {
    return(posterior[0])
  }
  out <- posterior[keep]
  out[, beta_index := parameter_index(variable)]
  setorder(out, beta_index)
  out[!is.na(beta_index)]
}

safe_parse_beta <- function(x) {
  parse(text = as.character(x))
}

SUBSCRIPT_DIGITS <- c(
  "0" = "\u2080",
  "1" = "\u2081",
  "2" = "\u2082",
  "3" = "\u2083",
  "4" = "\u2084",
  "5" = "\u2085",
  "6" = "\u2086",
  "7" = "\u2087",
  "8" = "\u2088",
  "9" = "\u2089"
)

subscript_number <- function(x) {
  vapply(as.character(x), function(value) {
    chars <- strsplit(value, "", fixed = TRUE)[[1]]
    paste0(SUBSCRIPT_DIGITS[chars], collapse = "")
  }, character(1))
}

beta_symbol_label <- function(index) {
  paste0("\u03b2", subscript_number(index))
}

PLOT_FONT_FAMILY <- "Times New Roman"
PLOT_TEXT_COLOUR <- "black"
PLOT_BASE_SIZE <- 12
PLOT_SMALL_BASE_SIZE <- 8.5

publication_theme <- function(base_size = PLOT_BASE_SIZE) {
  theme_bw(base_size = base_size, base_family = PLOT_FONT_FAMILY) +
    theme(
      text = element_text(family = PLOT_FONT_FAMILY, colour = PLOT_TEXT_COLOUR),
      axis.title = element_text(colour = PLOT_TEXT_COLOUR),
      axis.text = element_text(colour = PLOT_TEXT_COLOUR),
      legend.title = element_text(colour = PLOT_TEXT_COLOUR),
      legend.text = element_text(colour = PLOT_TEXT_COLOUR),
      strip.text = element_text(colour = PLOT_TEXT_COLOUR),
      plot.title = element_text(colour = PLOT_TEXT_COLOUR),
      plot.subtitle = element_text(colour = PLOT_TEXT_COLOUR),
      plot.caption = element_text(colour = PLOT_TEXT_COLOUR)
    )
}

script_arg <- grep("^--file=",commandArgs(),value=TRUE)[1]
script_dir <- dirname(normalizePath(sub("^--file=","",script_arg),mustWork=TRUE))
source(file.path(script_dir,"lib","plotting","simulation_figures.R"))

plot_coef_sum <- function(sum_values) {
  if (length(sum_values) < 2) {
    stop("Need at least two replicate coefficient sums.", call. = FALSE)
  }

  df <- data.frame(sum = sum_values)
  x_range <- range(df$sum, na.rm = TRUE)
  x_pad <- diff(x_range) * 0.08
  if (!is.finite(x_pad) || x_pad == 0) {
    x_pad <- 0.1
  }
  x_lim <- c(x_range[[1]] - x_pad, x_range[[2]] + x_pad)

  if (length(unique(df$sum)) == 1) {
    return(
      ggplot(df, aes(x = sum, y = 0)) +
        geom_vline(xintercept = 0, linetype = "dashed", color = "grey40", linewidth = 0.5) +
        geom_boxplot(
          aes(group = 1),
          orientation = "y",
          fill = "#E69F00",
          alpha = 0.7,
          width = 0.075,
          outlier.color = "#D55E00",
          outlier.size = 1.5,
          outlier.alpha = 0.8
        ) +
        annotate(
          "text",
          x = x_lim[[2]],
          y = 0.08,
          hjust = 1.1,
          size = 3.7,
          family = PLOT_FONT_FAMILY,
          colour = PLOT_TEXT_COLOUR,
          label = sprintf("Value = %.3f", unique(df$sum)[[1]])
        ) +
        labs(x = "Sum of coefficient estimates", y = NULL) +
        coord_cartesian(xlim = x_lim, ylim = c(-0.1, 0.12)) +
        publication_theme() +
        theme(panel.grid.minor = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank())
    )
  }

  dens <- density(df$sum)
  max_y <- max(dens$y)
  q_vec <- quantile(df$sum, probs = c(0.25, 0.5, 0.75), na.rm = TRUE)

  ggplot(df, aes(x = sum)) +
    geom_density(fill = "#56B4E9", alpha = 0.35, color = "#0072B2", linewidth = 1.0) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey40", linewidth = 0.5) +
    geom_boxplot(
      aes(y = -max_y * 0.075, group = 1),
      orientation = "y",
      fill = "#E69F00",
      alpha = 0.7,
      width = max_y * 0.06,
      outlier.color = "#D55E00",
      outlier.size = 1.5,
      outlier.alpha = 0.8
    ) +
    annotate(
      "text",
      x = x_lim[[2]],
      y = max_y * 1.0,
      hjust = 1.1,
      vjust = 1.5,
      size = 3.7,
      family = PLOT_FONT_FAMILY,
      colour = PLOT_TEXT_COLOUR,
      label = sprintf("Median = %.3f\nIQR = [%.3f, %.3f]", q_vec[[2]], q_vec[[1]], q_vec[[3]])
    ) +
    labs(x = "Sum of coefficient estimates", y = "Density") +
    coord_cartesian(xlim = x_lim, ylim = c(-max_y * 0.15, max_y * 1.15)) +
    publication_theme() +
    theme(panel.grid.minor = element_blank())
}

save_pdf_png <- function(plot, output_root, width, height, units = "mm", dpi = 300) {
  if(!capabilities("cairo")) stop("Cairo is required for embedded Times New Roman text and mathematics.")
  pdf_device <- function(...) grDevices::cairo_pdf(...,family=PLOT_FONT_FAMILY,
    symbolfamily=grDevices::cairoSymbolFont(PLOT_FONT_FAMILY,usePUA=FALSE))
  ggsave(paste0(output_root, ".pdf"), plot, width = width, height = height, units = units, device = pdf_device)
  # ggsave passes inches to a custom device; png otherwise interprets pixels.
  png_device <- function(...) grDevices::png(...,units="in",res=dpi,type="cairo",family=PLOT_FONT_FAMILY,
    symbolfamily=grDevices::cairoSymbolFont(PLOT_FONT_FAMILY,usePUA=FALSE))
  ggsave(paste0(output_root, ".png"), plot, width = width, height = height, units = units, dpi = dpi, device=png_device)
}

process_item <- function(data_root, output_dir, item) {
  method <- item$method
  sc_dir <- item$scenario_dir
  sc_label <- item$scenario_label
  baseline <- item$baseline
  n <- item$n
  p <- item$p
  nz <- item$nonzero

  diag <- load_diagnostics(data_root, sc_dir, method, baseline, n, p, nz, FOCUS_REP)
  post <- load_posterior(data_root, sc_dir, method, baseline, n, p, nz, FOCUS_REP)
  nz_params <- posterior_nonzero(post)
  nz_idx <- nz_params$beta_index
  if (length(nz_idx) == 0) {
    stop("No nonzero coefficients identified.", call. = FALSE)
  }

  samples <- load_samples(data_root, sc_dir, method, baseline, n, p, nz, FOCUS_REP, nz_idx)
  acf_data <- compute_acf(samples)
  sum_vals <- load_replicate_sums(data_root, sc_dir, method, baseline, n, p, nz)
  sample_sum_vals <- load_sample_sums(data_root, sc_dir, method, baseline, n, p, nz, FOCUS_REP)

  out_parent <- file.path(output_dir, sc_label, method)
  out_subdir <- file.path(out_parent, sprintf("rep%04d", FOCUS_REP))
  dir.create(out_subdir, showWarnings = FALSE, recursive = TRUE)

  fwrite(diag,file.path(out_subdir,"coefficient_diagnostics.csv"))
  fwrite(acf_data,file.path(out_subdir,"acf_values.csv"))
  fwrite(nz_params,file.path(out_subdir,"nonzero_truth.csv"))
  fig1 <- plot_rhat_ess(diag, p)
  save_pdf_png(fig1, file.path(out_subdir, "fig01_rhat_ess"), width = 280, height = 105)

  fig2 <- plot_trace_density_acf(samples, post, acf_data)
  if (!is.null(fig2)) {
    fig2_h <- 24 + length(nz_idx) * 40
    save_pdf_png(fig2, file.path(out_subdir, "fig01_trace_density_acf"), width = 280, height = fig2_h)
  }

  if (length(sum_vals) >= 2) {
    fig3 <- plot_coef_sum(sum_vals)
    save_pdf_png(fig3, file.path(out_parent, "fig03_coef_sum"), width = 155, height = 105)
  } else {
    warning(
      sprintf(
        "Skipping coefficient-sum plot for %s/%s: need at least 2 replicate sums, found %d.",
        sc_label,
        method,
        length(sum_vals)
      ),
      call. = FALSE
    )
  }

  if (length(sample_sum_vals) >= 2) {
    fig4 <- plot_coef_sum(sample_sum_vals)
    save_pdf_png(fig4, file.path(out_subdir, "fig04_coef_sum_samples"), width = 155, height = 105)
  } else {
    warning(
      sprintf(
        "Skipping posterior-sample coefficient-sum plot for %s/%s rep%d: need at least 2 sample sums, found %d.",
        sc_label,
        method,
        FOCUS_REP,
        length(sample_sum_vals)
      ),
      call. = FALSE
    )
  }
}

process_root <- function(data_root, output_dir = NULL) {
  data_root <- normalizePath(data_root, mustWork = FALSE)
  if (is.null(output_dir)) {
    output_dir <- file.path(data_root, "public_figs")
  }

  work <- discover_work(data_root)
  cat("==================================================\n")
  cat("BayesCox simulation figure generator\n")
  cat("Data root: ", data_root, "\n", sep = "")
  cat("Output:    ", output_dir, "\n", sep = "")
  cat("Rep:       ", FOCUS_REP, "\n", sep = "")
  cat("Items:     ", nrow(work), "\n", sep = "")
  cat("==================================================\n")

  if (nrow(work) == 0) {
    warning("No matching scenarios/methods found under: ", data_root, call. = FALSE)
    return(data.table(root = data_root, success = 0L, errors = 0L))
  }

  if (DRY_RUN) {
    print(work[, .(scenario_label, method, rep = FOCUS_REP)])
    return(data.table(root = data_root, success = 0L, errors = 0L))
  }

  total_success <- 0L
  total_error <- 0L
  for (sc_label in unique(work$scenario_label)) {
    sc_work <- work[scenario_label == sc_label]
    sc <- sc_work[1]
    cat(sprintf(
      "\nScenario: %s (n=%d, p=%d, nz=%d, %s)\n",
      sc_label,
      sc$n,
      sc$p,
      sc$nonzero,
      sc$baseline
    ))

    for (i in seq_len(nrow(sc_work))) {
      item <- sc_work[i]
      cat(sprintf("  %-18s ... ", item$method))
      result <- tryCatch(
        {
          process_item(data_root, output_dir, item)
          "OK"
        },
        error = function(e) paste("ERROR:", conditionMessage(e))
      )

      if (identical(result, "OK")) {
        cat("OK\n")
        total_success <- total_success + 1L
      } else {
        cat(result, "\n")
        total_error <- total_error + 1L
      }
    }
  }

  cat("\n==================================================\n")
  cat(sprintf("Done. Success: %d, Errors: %d\n", total_success, total_error))
  cat(sprintf("Output: %s\n", output_dir))
  cat("==================================================\n")
  data.table(root = data_root, success = total_success, errors = total_error)
}

if(!is.null(MANIFEST)) {
  source(file.path(script_dir,"lib","plotting","simulation_manifest.R"))
  process_manifest(MANIFEST, OUTPUT_DIR_OVERRIDE)
  quit(status=0)
}
if(ALL_REPS) stop("--rep all requires --manifest, so repeated fits cannot overwrite one another.")
expanded_roots <- unique(unlist(lapply(ROOTS, expand_roots), use.names = FALSE))
expanded_roots <- expanded_roots[nzchar(expanded_roots)]
if (length(expanded_roots) == 0) {
  stop("No usable simulation roots found.", call. = FALSE)
}
if (!is.null(OUTPUT_DIR_OVERRIDE) && length(expanded_roots) != 1) {
  stop("--output-dir can only be used when exactly one simulation root is selected.", call. = FALSE)
}

results <- rbindlist(lapply(expanded_roots, function(root) {
  process_root(root, OUTPUT_DIR_OVERRIDE)
}), fill = TRUE)

if (nrow(results) > 1) {
  cat("\nOverall summary:\n")
  print(results)
}

if (sum(results$errors, na.rm = TRUE) > 0) {
  quit(status = 1)
}
