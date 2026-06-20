#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

arg_value <- function(name, default = NULL) {
  key <- paste0("--", name)
  idx <- match(key, args)
  if (is.na(idx) || idx == length(args)) {
    return(default)
  }
  args[[idx + 1]]
}

summary_dir <- arg_value("summary-dir", file.path("results", "summary", "realdata", "method_comparison"))
font_family <- Sys.getenv("REALDATA_PLOT_FONT", "Times New Roman")
read_scale <- function(env_name, default) {
  value <- suppressWarnings(as.numeric(Sys.getenv(env_name, as.character(default))))
  if (is.na(value) || value <= 0) {
    return(default)
  }
  value
}
forest_font_scale <- read_scale("REALDATA_FOREST_FONT_SCALE", 2.5)
rhat_font_scale <- read_scale("REALDATA_RHAT_FONT_SCALE", 2.5)
select_font_scale <- read_scale("REALDATA_SELECT_FONT_SCALE", 3.0)
select_layout_scale <- 1.5
select_point_size <- 3.3
select_legend_point_size <- 4.5
font_size <- function(x, scale) x * scale
legend_ncol <- function(n_items, max_cols = 4) {
  max(1, min(max_cols, n_items))
}
legend_extra_height <- function(n_items, ncol, scale, row_height = 0.24, padding = 0.35) {
  padding + ceiling(n_items / ncol) * row_height * scale
}
legend_ncol_for_width <- function(labels, width, text_size, max_cols = 4) {
  labels <- as.character(labels)
  n_items <- length(labels)
  if (n_items == 0) {
    return(1)
  }
  item_width <- 0.35 + max(nchar(labels), na.rm = TRUE) * text_size * 0.0065
  max_cols_by_width <- max(1, floor(width / item_width))
  legend_ncol(n_items, max_cols = min(max_cols, max_cols_by_width))
}

needed <- c("ggplot2", "dplyr", "readr")
missing_pkgs <- needed[!vapply(needed, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  stop("Missing R packages: ", paste(missing_pkgs, collapse = ", "), call. = FALSE)
}

library(ggplot2)
library(dplyr)
library(readr)

safe_name <- function(x) {
  x <- gsub("[^A-Za-z0-9_]+", "_", x)
  x <- gsub("_+", "_", x)
  x <- gsub("^_|_$", "", x)
  ifelse(nchar(x) == 0, "value", x)
}

ensure_dir <- function(path) {
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
  path
}

save_plot_bundle <- function(plot, pathroot, width, height) {
  tryCatch(
    ggsave(paste0(pathroot, ".pdf"), plot, width = width, height = height, units = "in", device = grDevices::cairo_pdf, limitsize = FALSE),
    error = function(e) {
      tryCatch(
        {
          grDevices::pdf(
            paste0(pathroot, ".pdf"),
            width = width,
            height = height,
            family = "Times"
          )
          on.exit(grDevices::dev.off(), add = TRUE)
          print(plot)
        },
        error = function(e2) {
          warning("Could not write PDF for ", pathroot, ": ", conditionMessage(e2), call. = FALSE)
        }
      )
    }
  )
  ggsave(paste0(pathroot, ".png"), plot, width = width, height = height, units = "in", dpi = 300, limitsize = FALSE)
  tryCatch(
    ggsave(paste0(pathroot, ".eps"), plot, width = width, height = height, units = "in", device = grDevices::cairo_ps, limitsize = FALSE),
    error = function(e) {
      grDevices::postscript(
        paste0(pathroot, ".eps"),
        width = width,
        height = height,
        horizontal = FALSE,
        onefile = FALSE,
        paper = "special",
        family = "Times"
      )
      print(plot)
      grDevices::dev.off()
    }
  )
}

prior_label <- function(x) {
  dplyr::recode(
    as.character(x),
    "hs" = "HS",
    "rhs" = "RHS",
    "new-rhs" = "New-RHS",
    "est-rhs" = "Est-RHS",
    "cox-hs" = "Cox-HS",
    "cox-rhs" = "Cox-RHS",
    "lasso" = "BLASSO",
    "sslasso" = "SSLASSO",
    "prox-lasso" = "Prox-LASSO",
    "prox-sslasso" = "Prox-SSLASSO",
    .default = as.character(x)
  )
}

constraint_label <- function(x) {
  dplyr::recode(
    as.character(x),
    "soft_center" = "soft-center",
    "soft-center" = "soft-center",
    "prox" = "prox",
    .default = as.character(x)
  )
}

baseline_method_label <- function(x) {
  dplyr::recode(
    as.character(x),
    "FLORAL_min" = "FLORAL(min)",
    "FLORAL_1se" = "FLORAL(1se)",
    "coda4microbiome_lambda.min" = "coda4microbiome(min)",
    "coda4microbiome_lambda.1se" = "coda4microbiome(1se)",
    .default = as.character(x)
  )
}

method_family_label <- function(x) {
  dplyr::recode(
    as.character(x),
    "Bayes-prox" = "prox",
    "Bayes-soft-center" = "soft-center",
    "FLORAL" = "FLORAL",
    "coda4microbiome" = "coda4microbiome",
    .default = as.character(x)
  )
}

method_label <- function(df) {
  ifelse(
    grepl("^Bayes", df$method_family),
    paste(constraint_label(df$constraint), prior_label(df$prior), sep = " / "),
    baseline_method_label(df$method)
  )
}

legend_label <- function(x) {
  gsub(" / ", "\n", as.character(x), fixed = TRUE)
}

method_codes <- function(n) {
  if (n <= length(LETTERS)) {
    return(LETTERS[seq_len(n)])
  }
  outer_codes <- expand.grid(LETTERS, LETTERS, KEEP.OUT.ATTRS = FALSE)
  c(LETTERS, paste0(outer_codes[[1]], outer_codes[[2]]))[seq_len(n)]
}

base_theme <- function(scale) {
  theme_bw(base_family = font_family, base_size = font_size(14, scale)) +
    theme(
      plot.title = element_blank(),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      legend.title = element_blank(),
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.box = "vertical",
      legend.justification = "center",
      legend.margin = margin(t = 3, r = 0, b = 0, l = 0),
      axis.text = element_text(color = "black"),
      strip.background = element_rect(fill = "grey95", color = "grey70"),
      strip.text = element_text(face = "bold")
    )
}

coefs_path <- file.path(summary_dir, "plot_data", "coefficients_long.csv")
rhat_path <- file.path(summary_dir, "plot_data", "rhat_long.csv")
select_path <- file.path(summary_dir, "plot_data", "selection_matrix.csv")

coefs <- readr::read_csv(coefs_path, show_col_types = FALSE)
rhats <- readr::read_csv(rhat_path, show_col_types = FALSE)
select_df <- readr::read_csv(select_path, show_col_types = FALSE)

coefs <- coefs %>%
  mutate(
    method_label = method_label(.),
    method_family_display = method_family_label(method_family),
    selected = as.logical(selected),
    interval_available = as.logical(interval_available),
    point_estimate = as.numeric(point_estimate),
    variable_index = as.integer(variable_index)
  )

select_df <- select_df %>%
  mutate(
    method_label = method_label(.),
    selected = as.logical(selected),
    variable_index = as.integer(variable_index)
  )

rhats <- rhats %>%
  mutate(
    method_label = method_label(.),
    constraint_display = constraint_label(constraint),
    variable_index = as.integer(variable_index),
    rhat = as.numeric(rhat)
  )

forest_dir <- ensure_dir(file.path(summary_dir, "figures", "forest"))
rhat_dir <- ensure_dir(file.path(summary_dir, "figures", "rhat"))
select_dir <- ensure_dir(file.path(summary_dir, "figures", "select"))

for (dataset in sort(unique(coefs$dataset_id))) {
  dat <- coefs %>%
    filter(dataset_id == dataset) %>%
    arrange(variable_index, method_family, method)
  if (nrow(dat) == 0) next

  var_levels <- dat %>%
    distinct(variable_index, variable) %>%
    arrange(variable_index) %>%
    pull(variable)
  method_levels <- unique(dat$method_label)

  dat <- dat %>%
    mutate(
      variable = factor(variable, levels = rev(var_levels)),
      method_label = factor(method_label, levels = method_levels)
    )

  n_var <- length(var_levels)
  n_method <- length(method_levels)
  height <- max(7, 2.5 + 0.17 * n_var * forest_font_scale / 1.5)
  width <- max(22, min(28, 7.5 + 0.22 * n_method))
  forest_legend_cols <- legend_ncol(n_method, max_cols = 5)
  forest_height <- height + legend_extra_height(n_method, forest_legend_cols, forest_font_scale, row_height = 0.34)
  dodge <- position_dodge(width = 0.75)

  p <- ggplot(dat, aes(x = point_estimate, y = variable, color = method_label, shape = method_family_display, group = method_label)) +
    geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.35, color = "grey35") +
    geom_errorbar(
      data = dat %>% filter(interval_available, !is.na(lower), !is.na(upper)),
      aes(xmin = lower, xmax = upper),
      orientation = "y",
      width = 0,
      linewidth = 0.25,
      alpha = 0.42,
      position = dodge,
      na.rm = TRUE
    ) +
    geom_point(size = 1.25, alpha = 0.9, position = dodge, na.rm = TRUE) +
    scale_color_discrete(labels = legend_label) +
    labs(x = "Coefficient", y = NULL) +
    guides(
      color = guide_legend(ncol = forest_legend_cols, byrow = TRUE, override.aes = list(size = 2.4, alpha = 1)),
      shape = guide_legend(nrow = 1, byrow = TRUE, override.aes = list(size = 2.4, alpha = 1))
    ) +
    base_theme(forest_font_scale) +
    theme(
      axis.text.y = element_text(size = font_size(ifelse(n_var > 80, 6.5, 8.5), forest_font_scale)),
      axis.text.x = element_text(size = font_size(11, forest_font_scale)),
      legend.text = element_text(size = font_size(9, forest_font_scale)),
      legend.key.height = unit(forest_font_scale * 0.34, "cm"),
      legend.key.width = unit(0.35, "cm"),
      legend.spacing.x = unit(0.12, "cm")
    )

  save_plot_bundle(p, file.path(forest_dir, paste0(safe_name(dataset), "_forest")), width, forest_height)
}

for (dataset in sort(unique(rhats$dataset_id))) {
  dat <- rhats %>%
    filter(dataset_id == dataset, !is.na(rhat)) %>%
    arrange(variable_index, constraint, prior)
  if (nrow(dat) == 0) next

  dat <- dat %>%
    mutate(method_label = factor(method_label, levels = unique(method_label)))

  y_max <- max(1.12, stats::quantile(dat$rhat, 0.99, na.rm = TRUE) * 1.03)
  rhat_n_method <- dplyr::n_distinct(dat$method_label)
  rhat_width <- max(22, min(28, 7.5 + 0.22 * rhat_n_method))
  rhat_legend_cols <- legend_ncol(rhat_n_method, max_cols = 5)
  rhat_height <- 7.2 + legend_extra_height(rhat_n_method, rhat_legend_cols, rhat_font_scale, row_height = 0.34)
  p <- ggplot(dat, aes(x = variable_index, y = rhat, color = method_label, shape = constraint_display)) +
    geom_hline(yintercept = 1.0, linetype = "dashed", color = "red", linewidth = 0.45) +
    geom_hline(yintercept = 1.1, linetype = "dashed", color = "red", linewidth = 0.45) +
    annotate("text", x = Inf, y = 1.0, label = "1.0", hjust = 1.05, vjust = -0.35, color = "red", size = font_size(3.4, rhat_font_scale), family = font_family) +
    annotate("text", x = Inf, y = 1.1, label = "1.1", hjust = 1.05, vjust = -0.35, color = "red", size = font_size(3.4, rhat_font_scale), family = font_family) +
    geom_point(size = 1.7, alpha = 0.82) +
    scale_color_discrete(labels = legend_label) +
    coord_cartesian(ylim = c(min(0.98, min(dat$rhat, na.rm = TRUE)), y_max)) +
    labs(x = "Variable index", y = "Rhat") +
    guides(
      color = guide_legend(ncol = rhat_legend_cols, byrow = TRUE, override.aes = list(size = 2.5, alpha = 1)),
      shape = guide_legend(nrow = 1, byrow = TRUE, override.aes = list(size = 2.5, alpha = 1))
    ) +
    base_theme(rhat_font_scale) +
    theme(
      legend.text = element_text(size = font_size(9, rhat_font_scale)),
      legend.key.width = unit(0.35, "cm"),
      legend.spacing.x = unit(0.12, "cm")
    )

  save_plot_bundle(p, file.path(rhat_dir, paste0(safe_name(dataset), "_rhat")), rhat_width, rhat_height)
}

for (dataset in sort(unique(select_df$dataset_id))) {
  dat <- select_df %>%
    filter(dataset_id == dataset) %>%
    arrange(variable_index, method_family, method)
  if (nrow(dat) == 0) next

  var_levels <- dat %>%
    distinct(variable_index, variable) %>%
    arrange(variable_index) %>%
    pull(variable)
  method_levels <- unique(dat$method_label)
  method_map <- dat %>%
    distinct(method_label, method_family) %>%
    mutate(
      method_index = dplyr::row_number(),
      method_code = method_codes(dplyr::n()),
      method_legend = paste0("(", method_code, ") ", method_label)
    )

  dat <- dat %>%
    left_join(method_map, by = c("method_label", "method_family")) %>%
    mutate(
      method_index = as.integer(method_index),
      method_legend = factor(method_legend, levels = method_map$method_legend)
    )

  n_var <- length(var_levels)
  n_method <- length(method_levels)
  plot_size <- max(8.5, min(30, max(4.8 + 0.12 * n_var, 6.5 + 0.24 * n_method)))
  y_label_width <- max(2.8, min(8.0, 1.0 + 0.055 * select_layout_scale * max(nchar(var_levels))))
  plot_width <- plot_size + y_label_width
  select_legend_cols <- legend_ncol_for_width(method_map$method_legend, plot_width, font_size(7.8, select_font_scale), max_cols = 2)
  select_height <- plot_size + legend_extra_height(n_method, select_legend_cols, select_font_scale, row_height = 0.26, padding = 0.45)
  legend_seed <- method_map %>%
    mutate(
      method_legend = factor(method_legend, levels = method_map$method_legend),
      variable_index = 1
    )

  p <- ggplot(dat, aes(x = method_index, y = variable_index)) +
    geom_blank(data = dat, aes(x = method_index, y = variable_index)) +
    geom_point(
      data = legend_seed,
      aes(x = method_index, y = variable_index, color = method_legend),
      inherit.aes = FALSE,
      shape = 16,
      size = select_point_size,
      alpha = 0,
      show.legend = TRUE
    ) +
    geom_point(
      data = dat %>% filter(selected),
      aes(color = method_legend),
      shape = 16,
      size = select_point_size,
      alpha = 0.9
    ) +
    scale_x_continuous(
      breaks = method_map$method_index,
      labels = method_map$method_code,
      limits = c(0.5, n_method + 0.5),
      expand = c(0, 0)
    ) +
    scale_y_reverse(
      breaks = seq_along(var_levels),
      labels = var_levels,
      limits = c(n_var + 0.5, 0.5),
      expand = c(0, 0)
    ) +
    scale_color_hue(
      limits = method_map$method_legend,
      breaks = method_map$method_legend,
      drop = FALSE
    ) +
    coord_fixed(ratio = n_method / n_var, clip = "off") +
    labs(x = "Method", y = NULL, color = "Method") +
    guides(color = guide_legend(ncol = select_legend_cols, byrow = TRUE, override.aes = list(shape = 16, size = select_legend_point_size, alpha = 1))) +
    base_theme(select_font_scale) +
    theme(
      panel.grid.major.x = element_line(color = "grey92", linewidth = 0.25),
      panel.grid.major.y = element_line(color = "grey92", linewidth = 0.25),
      axis.text.x = element_text(angle = 0, hjust = 0.5, vjust = 0.5, size = font_size(ifelse(n_method > 24, 7.5, 8.5), select_font_scale)),
      axis.text.y = element_text(size = font_size(ifelse(n_var > 100, 6.0, ifelse(n_var > 60, 7.0, 8.0)), select_font_scale)),
      legend.text = element_text(size = font_size(7.8, select_font_scale)),
      legend.title = element_blank(),
      legend.key.height = unit(select_font_scale * 0.28, "cm"),
      legend.key.width = unit(select_font_scale * 0.42, "cm")
    )

  save_plot_bundle(p, file.path(select_dir, paste0(safe_name(dataset), "_select")), plot_width, select_height)
}
