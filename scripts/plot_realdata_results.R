#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
outdir <- NULL
if (length(args) == 1 && !grepl("^--", args[[1]])) {
  outdir <- args[[1]]
} else {
  i <- 1
  while (i <= length(args)) {
    if (args[[i]] == "--outdir" && i < length(args)) {
      outdir <- args[[i + 1]]
      i <- i + 2
    } else {
      i <- i + 1
    }
  }
}
if (is.null(outdir)) {
  stop("Usage: plot_realdata_results.R --outdir <summary-dir>", call. = FALSE)
}

ensure_package <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}
ensure_package("ggplot2")

figdir <- file.path(outdir, "figures")
dir.create(figdir, recursive = TRUE, showWarnings = FALSE)

font_family <- "Times New Roman"
base_plot_size <- 14
forest_base_size <- 13

read_table <- function(name) {
  path <- file.path(outdir, name)
  if (!file.exists(path)) {
    stop("Missing input table: ", path, call. = FALSE)
  }
  read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
}

num <- function(x) suppressWarnings(as.numeric(x))

safe_name <- function(x) gsub("[^A-Za-z0-9_.-]+", "_", x)

save_bundle <- function(plot, stem, width, height, dpi = 180) {
  output_path <- function(ext) {
    path <- file.path(figdir, paste0(stem, ext))
    if (file.exists(path)) {
      ok <- suppressWarnings(file.remove(path))
      if (!isTRUE(ok)) {
        path <- file.path(figdir, paste0(stem, "_new", ext))
        if (file.exists(path)) suppressWarnings(file.remove(path))
      }
    }
    path
  }
  png_path <- output_path(".png")
  pdf_path <- output_path(".pdf")
  eps_path <- output_path(".eps")
  png_device <- function(filename, width, height, units, res, ...) {
    grDevices::png(filename, width = width, height = height, units = units, res = res, type = "cairo", ...)
  }
  ggplot2::ggsave(png_path, plot = plot, width = width, height = height, units = "in", dpi = dpi, device = png_device, bg = "white")
  ggplot2::ggsave(pdf_path, plot = plot, width = width, height = height, units = "in", device = grDevices::cairo_pdf, bg = "white")
  ggplot2::ggsave(eps_path, plot = plot, width = width, height = height, units = "in", device = grDevices::cairo_ps, bg = "white")
}

prior_order <- c("hs", "rhs", "new-rhs", "cox-hs", "cox-rhs", "lasso", "sslasso")
baseline_order <- c("FLORAL_min", "FLORAL_1se", "coda4microbiome_lambda.min", "coda4microbiome_lambda.1se")
palette <- c(
  "FLORAL" = "#4E79A7",
  "coda4microbiome" = "#F28E2B",
  "hs" = "#276FBF",
  "rhs" = "#F28E2B",
  "new-rhs" = "#59A14F",
  "cox-hs" = "#AF7AA1",
  "cox-rhs" = "#E15759",
  "lasso" = "#4E79A7",
  "sslasso" = "#76B7B2",
  "FLORAL_min" = "#1F77B4",
  "FLORAL_1se" = "#6BAED6",
  "coda4microbiome_lambda.min" = "#D95F02",
  "coda4microbiome_lambda.1se" = "#FDB863"
)

plot_cv_cindex <- function() {
  df <- read_table("best_per_prior_and_baselines.csv")
  df$cv_cindex <- num(df$cv_cindex)
  df <- df[is.finite(df$cv_cindex), , drop = FALSE]
  df$category <- ifelse(df$method_family == "Bayes-soft-center", df$prior, df$method_family)
  category_levels <- c("FLORAL", "coda4microbiome", prior_order)
  category_levels <- category_levels[category_levels %in% df$category]
  df$category <- factor(df$category, levels = category_levels)
  df$dataset_id <- factor(df$dataset_id, levels = sort(unique(df$dataset_id)))
  df$shape_group <- ifelse(df$method_family == "Bayes-soft-center", "Bayes prior", "Baseline")

  p <- ggplot2::ggplot(df, ggplot2::aes(x = dataset_id, y = cv_cindex, color = category, shape = shape_group)) +
    ggplot2::geom_point(position = ggplot2::position_dodge(width = 0.78), size = 2.4, stroke = 0.4) +
    ggplot2::scale_color_manual(values = palette, drop = FALSE) +
    ggplot2::scale_shape_manual(values = c("Bayes prior" = 16, "Baseline" = 17)) +
    ggplot2::labs(x = NULL, y = "CV-Cindex", color = "Prior / method", shape = NULL) +
    ggplot2::theme_bw(base_size = base_plot_size, base_family = font_family) +
    ggplot2::theme(
      legend.position = "bottom",
      legend.box = "vertical",
      axis.text.x = ggplot2::element_text(angle = 35, hjust = 1, vjust = 1, family = font_family),
      text = ggplot2::element_text(family = font_family),
      panel.grid.minor = ggplot2::element_blank()
    )
  save_bundle(p, "cv_cindex_comparison", width = 14, height = 6)
}

plot_rhat <- function() {
  df <- read_table("bayes_rhat_by_parameter.csv")
  df$rhat <- num(df$rhat)
  df <- df[is.finite(df$rhat), , drop = FALSE]
  if (nrow(df) == 0) return(invisible(NULL))
  df$prior <- factor(df$prior, levels = prior_order[prior_order %in% df$prior])
  df$dataset_id <- factor(df$dataset_id, levels = sort(unique(df$dataset_id)))
  cap <- max(1.10, min(max(df$rhat, na.rm = TRUE), 1.25))
  df$rhat_plot <- pmin(df$rhat, cap)

  p <- ggplot2::ggplot(df, ggplot2::aes(x = dataset_id, y = rhat_plot, color = prior)) +
    ggplot2::geom_hline(yintercept = c(1.01, 1.05, 1.10), linetype = "dashed", color = "#B55A30", linewidth = 0.25) +
    ggplot2::geom_point(position = ggplot2::position_jitterdodge(jitter.width = 0.07, dodge.width = 0.72), size = 1.2) +
    ggplot2::scale_color_manual(values = palette, drop = FALSE) +
    ggplot2::labs(x = NULL, y = "Rhat", color = "Prior") +
    ggplot2::theme_bw(base_size = base_plot_size, base_family = font_family) +
    ggplot2::theme(
      legend.position = "bottom",
      axis.text.x = ggplot2::element_text(angle = 35, hjust = 1, vjust = 1, family = font_family),
      text = ggplot2::element_text(family = font_family),
      panel.grid.minor = ggplot2::element_blank()
    )
  save_bundle(p, "rhat_scatter", width = 14, height = 6)
}

plot_forest_one <- function(dataset_id, df_all) {
  df <- df_all[df_all$dataset_id == dataset_id, , drop = FALSE]
  if (nrow(df) == 0) return(invisible(NULL))
  df$estimate <- num(df$estimate)
  df$lower <- num(df$lower)
  df$upper <- num(df$upper)
  df <- df[is.finite(df$estimate), , drop = FALSE]
  if (nrow(df) == 0) return(invisible(NULL))
  method_levels <- c(paste0("Bayes_", prior_order), baseline_order)
  method_levels <- method_levels[method_levels %in% df$method]
  df$method <- factor(df$method, levels = method_levels)
  df$method_label <- ifelse(grepl("^Bayes_", as.character(df$method)), sub("^Bayes_", "", as.character(df$method)), as.character(df$method))
  df$method_label <- factor(df$method_label, levels = ifelse(grepl("^Bayes_", method_levels), sub("^Bayes_", "", method_levels), method_levels))
  method_index <- match(as.character(df$method), method_levels)
  n_methods <- max(1, length(method_levels))

  agg <- aggregate(abs(estimate) ~ variable, data = df, FUN = max, na.rm = TRUE)
  agg <- agg[order(agg[[2]], decreasing = TRUE), , drop = FALSE]
  variable_levels <- agg$variable
  df$y_base <- match(df$variable, variable_levels)
  df$y_plot <- df$y_base + if (n_methods == 1) 0 else (method_index - (n_methods + 1) / 2) * (0.72 / (n_methods - 1))
  df$selected_bool <- tolower(df$selected) == "true"
  df$is_bayes <- df$method_family == "Bayes-soft-center"
  df$plot_alpha <- ifelse(df$selected_bool, 0.95, ifelse(df$is_bayes, 0.42, 0.20))
  df$plot_size <- ifelse(df$selected_bool, 1.25, 0.75)

  bayes <- df[df$is_bayes & is.finite(df$lower) & is.finite(df$upper), , drop = FALSE]
  baselines <- df[!df$is_bayes, , drop = FALSE]
  row_bg <- data.frame(y = seq_along(variable_levels))
  row_bg <- row_bg[row_bg$y %% 2 == 0, , drop = FALSE]
  height <- max(8, 2.8 + 0.32 * length(variable_levels))
  y_text_size <- ifelse(length(variable_levels) > 100, 8, ifelse(length(variable_levels) > 50, 9, 11))

  p <- ggplot2::ggplot() +
    ggplot2::geom_rect(data = row_bg, ggplot2::aes(xmin = -Inf, xmax = Inf, ymin = y - 0.5, ymax = y + 0.5), fill = "#F7F7F7", inherit.aes = FALSE) +
    ggplot2::geom_vline(xintercept = 0, linetype = "dashed", color = "#555555", linewidth = 0.3) +
    ggplot2::geom_segment(
      data = bayes,
      ggplot2::aes(x = lower, xend = upper, y = y_plot, yend = y_plot, color = method_label, alpha = plot_alpha),
      linewidth = 0.35
    ) +
    ggplot2::geom_point(
      data = df[df$is_bayes, , drop = FALSE],
      ggplot2::aes(x = estimate, y = y_plot, color = method_label, alpha = plot_alpha, size = plot_size),
      shape = 16
    ) +
    ggplot2::geom_point(
      data = baselines,
      ggplot2::aes(x = estimate, y = y_plot, color = method_label, alpha = plot_alpha, size = plot_size),
      shape = 18
    ) +
    ggplot2::scale_y_continuous(
      breaks = seq_along(variable_levels),
      labels = variable_levels,
      trans = "reverse",
      expand = ggplot2::expansion(mult = c(0.01, 0.01))
    ) +
    ggplot2::scale_color_manual(values = palette, drop = FALSE) +
    ggplot2::scale_alpha_identity() +
    ggplot2::scale_size_identity() +
    ggplot2::labs(x = "Coefficient", y = NULL, color = "Prior / method") +
    ggplot2::theme_bw(base_size = forest_base_size, base_family = font_family) +
    ggplot2::theme(
      legend.position = "bottom",
      text = ggplot2::element_text(family = font_family),
      axis.text.y = ggplot2::element_text(size = y_text_size, family = font_family),
      panel.grid.minor = ggplot2::element_blank(),
      panel.grid.major.y = ggplot2::element_blank()
    )
  save_bundle(p, paste0("forest_", safe_name(dataset_id)), width = 16, height = height)
}

plot_forests <- function() {
  df <- read_table("all_coefficients_for_forest.csv")
  for (dataset_id in sort(unique(df$dataset_id))) {
    plot_forest_one(dataset_id, df)
  }
}

plot_cv_cindex()
plot_rhat()
plot_forests()
cat("R plotting complete:", normalizePath(figdir), "\n")
