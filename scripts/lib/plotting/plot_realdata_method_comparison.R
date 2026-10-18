#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
initial_rplots_exists <- file.exists("Rplots.pdf")

arg_value <- function(name, default = NULL) {
  key <- paste0("--", name)
  idx <- match(key, args)
  if (is.na(idx) || idx == length(args)) {
    return(default)
  }
  args[[idx + 1]]
}

arg_flag <- function(name, default = FALSE) {
  value <- arg_value(name, NULL)
  if (is.null(value)) {
    return(default)
  }
  tolower(value) %in% c("1", "true", "yes", "y")
}

arg_list <- function(name, default = character()) {
  value <- arg_value(name, NULL)
  if (is.null(value) || !nzchar(value)) {
    return(default)
  }
  trimws(strsplit(value, ",", fixed = TRUE)[[1]])
}

is_all_filter <- function(values) {
  values <- trimws(values)
  length(values) == 0 || any(tolower(values) == "all")
}

normalize_prior_filter <- function(values) {
  values <- trimws(values)
  values <- gsub("_", "-", tolower(values), fixed = TRUE)
  unique(setdiff(values, c("", "all")))
}

summary_dir <- arg_value("summary-dir", file.path("results", "summary", "realdata", "method_comparison"))
t1d_metadata_path <- arg_value("t1d-metadata", file.path("data", "T1D_metadata.csv"))
combined_layouts <- arg_list("combined-layouts", "none")
combined_layouts <- setdiff(tolower(combined_layouts), c("", "none"))
valid_combined_layouts <- c("stacked", "aligned", "summary")
bad_combined_layouts <- setdiff(combined_layouts, valid_combined_layouts)
if (length(bad_combined_layouts) > 0) {
  stop("Unknown --combined-layouts value(s): ", paste(bad_combined_layouts, collapse = ", "), call. = FALSE)
}
combined_datasets <- arg_list("combined-datasets", "all")
combined_only <- arg_flag("combined-only", FALSE)
if (combined_only && length(combined_layouts) == 0) {
  stop("--combined-only requires --combined-layouts.", call. = FALSE)
}
prior_filter_arg <- arg_list("priors", "all")
baseline_family_filter_arg <- arg_list("baseline-families", "all")
prior_filter <- if (is_all_filter(prior_filter_arg)) {
  character()
} else {
  normalize_prior_filter(prior_filter_arg)
}

font_family <- Sys.getenv("REALDATA_PLOT_FONT", "Times New Roman")
read_scale <- function(env_name, default) {
  value <- suppressWarnings(as.numeric(Sys.getenv(env_name, as.character(default))))
  if (is.na(value) || value <= 0) {
    return(default)
  }
  value
}
forest_font_scale <- read_scale("REALDATA_FOREST_FONT_SCALE", 2.0)
rhat_font_scale <- read_scale("REALDATA_RHAT_FONT_SCALE", 2.0)
forest_width_scale <- read_scale("REALDATA_FOREST_WIDTH_SCALE", 0.7)
forest_height_scale <- read_scale("REALDATA_FOREST_HEIGHT_SCALE", 1.25)
rhat_width_scale <- read_scale("REALDATA_RHAT_WIDTH_SCALE", 0.7)
rhat_height_scale <- read_scale("REALDATA_RHAT_HEIGHT_SCALE", 1.25)
select_font_scale <- read_scale("REALDATA_SELECT_FONT_SCALE", 3.0)
combined_font_scale <- read_scale("REALDATA_COMBINED_FONT_SCALE", forest_font_scale)
combined_axis_text_scale <- read_scale("REALDATA_COMBINED_AXIS_TEXT_SCALE", 1.12)
combined_axis_title_scale <- read_scale("REALDATA_COMBINED_AXIS_TITLE_SCALE", 1.15)
combined_legend_text_scale <- read_scale("REALDATA_COMBINED_LEGEND_TEXT_SCALE", 1.15)
combined_strip_text_scale <- read_scale("REALDATA_COMBINED_STRIP_TEXT_SCALE", 1.12)
combined_height_scale <- read_scale("REALDATA_COMBINED_HEIGHT_SCALE", 1.12)
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

needed <- c("ggplot2", "dplyr", "readr", "gtable")
missing_pkgs <- needed[!vapply(needed, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  stop("Missing R packages: ", paste(missing_pkgs, collapse = ", "), call. = FALSE)
}

library(ggplot2)
library(dplyr)
library(readr)

draw_plot <- function(plot) {
  if (inherits(plot, "grob") || inherits(plot, "gTree") || inherits(plot, "gtable")) {
    grid::grid.draw(plot)
  } else {
    print(plot)
  }
}

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

save_grob_bundle <- function(plot, pathroot, width, height) {
  had_rplots <- file.exists("Rplots.pdf")
  on.exit(
    {
      while (grDevices::dev.cur() > 1) {
        grDevices::dev.off()
      }
      if (!had_rplots && file.exists("Rplots.pdf")) {
        unlink("Rplots.pdf")
      }
    },
    add = TRUE
  )

  draw_to_device <- function(open_device) {
    open_device()
    on.exit(grDevices::dev.off(), add = TRUE)
    grid::grid.newpage()
    draw_plot(plot)
  }

  tryCatch(
    {
      draw_to_device(function() {
        grDevices::cairo_pdf(paste0(pathroot, ".pdf"), width = width, height = height, bg = "white")
      })
    },
    error = function(e) {
      tryCatch(
        {
          draw_to_device(function() {
            grDevices::pdf(
              paste0(pathroot, ".pdf"),
              width = width,
              height = height,
              family = "Times",
              bg = "white"
            )
          })
        },
        error = function(e2) {
          warning("Could not write PDF for ", pathroot, ": ", conditionMessage(e2), call. = FALSE)
        }
      )
    }
  )

  draw_to_device(function() {
    grDevices::png(paste0(pathroot, ".png"), width = width, height = height, units = "in", res = 300, bg = "white")
  })

  tryCatch(
    {
      draw_to_device(function() {
        grDevices::cairo_ps(paste0(pathroot, ".eps"), width = width, height = height, onefile = FALSE, bg = "white")
      })
    },
    error = function(e) {
      draw_to_device(function() {
        grDevices::postscript(
          paste0(pathroot, ".eps"),
          width = width,
          height = height,
          horizontal = FALSE,
          onefile = FALSE,
          paper = "special",
          family = "Times",
          bg = "white"
        )
      })
    }
  )
}

save_plot_bundle <- function(plot, pathroot, width, height) {
  if (inherits(plot, "grob") || inherits(plot, "gTree") || inherits(plot, "gtable")) {
    save_grob_bundle(plot, pathroot, width, height)
    return(invisible(NULL))
  }

  tryCatch(
    ggsave(paste0(pathroot, ".pdf"), plot, width = width, height = height, units = "in", device = grDevices::cairo_pdf, limitsize = FALSE, bg = "white"),
    error = function(e) {
      tryCatch(
        {
          grDevices::pdf(
            paste0(pathroot, ".pdf"),
            width = width,
            height = height,
            family = "Times",
            bg = "white"
          )
          on.exit(grDevices::dev.off(), add = TRUE)
          draw_plot(plot)
        },
        error = function(e2) {
          warning("Could not write PDF for ", pathroot, ": ", conditionMessage(e2), call. = FALSE)
        }
      )
    }
  )
  ggsave(paste0(pathroot, ".png"), plot, width = width, height = height, units = "in", dpi = 300, limitsize = FALSE, bg = "white")
  tryCatch(
    ggsave(paste0(pathroot, ".eps"), plot, width = width, height = height, units = "in", device = grDevices::cairo_ps, limitsize = FALSE, bg = "white"),
    error = function(e) {
      grDevices::postscript(
        paste0(pathroot, ".eps"),
        width = width,
        height = height,
        horizontal = FALSE,
        onefile = FALSE,
        paper = "special",
        family = "Times",
        bg = "white"
      )
      draw_plot(plot)
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

display_color_group <- function(df) {
  ifelse(
    grepl("^Bayes", df$method_family),
    prior_label(df$prior),
    method_family_label(df$method_family)
  )
}

display_shape_group <- function(df) {
  ifelse(
    grepl("^Bayes", df$method_family),
    constraint_label(df$constraint),
    method_family_label(df$method_family)
  )
}

legend_label <- function(x) {
  gsub(" / ", "\n", as.character(x), fixed = TRUE)
}

baseline_axis_label <- function(x) {
  label <- sub("^.*\\(([^()]*)\\)$", "\\1", as.character(x))
  ifelse(label == as.character(x), as.character(x), label)
}

coef_sign_label <- function(x) {
  dplyr::case_when(
    is.na(x) ~ "",
    x > 0 ~ "+",
    x < 0 ~ "-",
    TRUE ~ "0"
  )
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

load_t1d_metadata <- function(path) {
  if (!file.exists(path)) {
    warning("T1D metadata not found: ", path, call. = FALSE)
    return(data.frame(variable = character(), variable_label = character()))
  }
  meta <- readr::read_csv(path, show_col_types = FALSE)
  required <- c("feature", "genus")
  missing_cols <- setdiff(required, names(meta))
  if (length(missing_cols) > 0) {
    warning(
      "T1D metadata is missing required column(s): ",
      paste(missing_cols, collapse = ", "),
      call. = FALSE
    )
    return(data.frame(variable = character(), variable_label = character()))
  }
  meta %>%
    transmute(
      variable = as.character(feature),
      variable_label = as.character(genus)
    ) %>%
    filter(!is.na(variable), nzchar(variable), !is.na(variable_label), nzchar(variable_label)) %>%
    distinct(variable, .keep_all = TRUE)
}

apply_t1d_variable_labels <- function(df, label_map) {
  if (nrow(df) == 0 || !("variable" %in% names(df))) {
    df$variable_label <- character(nrow(df))
    return(df)
  }
  df %>%
    mutate(variable = as.character(variable)) %>%
    left_join(label_map, by = "variable") %>%
    mutate(
      variable_label = ifelse(
        as.character(dataset_id) == "T1D" & !is.na(variable_label) & nzchar(variable_label),
        variable_label,
        variable
      )
    )
}

variable_level_table <- function(df) {
  df %>%
    distinct(variable_index, variable, variable_label) %>%
    arrange(variable_index)
}

variable_label_map <- function(var_table) {
  stats::setNames(as.character(var_table$variable_label), as.character(var_table$variable))
}

t1d_label_map <- load_t1d_metadata(t1d_metadata_path)
coefs <- apply_t1d_variable_labels(coefs, t1d_label_map)
select_df <- apply_t1d_variable_labels(select_df, t1d_label_map)
rhats <- apply_t1d_variable_labels(rhats, t1d_label_map)

coefs <- coefs %>%
  mutate(
    method_label = method_label(.),
    color_group = display_color_group(.),
    shape_group = display_shape_group(.),
    selected = as.logical(selected),
    interval_available = as.logical(interval_available),
    point_estimate = as.numeric(point_estimate),
    variable_index = as.integer(variable_index)
  )

select_df <- select_df %>%
  mutate(
    method_label = method_label(.),
    color_group = display_color_group(.),
    shape_group = display_shape_group(.),
    selected = as.logical(selected),
    variable_index = as.integer(variable_index)
  )

rhats <- rhats %>%
  mutate(
    method_label = method_label(.),
    color_group = display_color_group(.),
    shape_group = display_shape_group(.),
    variable_index = as.integer(variable_index),
    rhat = as.numeric(rhat)
  )

available_priors <- sort(unique(c(
  as.character(coefs$prior[grepl("^Bayes", as.character(coefs$method_family))]),
  as.character(select_df$prior[grepl("^Bayes", as.character(select_df$method_family))]),
  as.character(rhats$prior[grepl("^Bayes", as.character(rhats$method_family))])
)))
available_priors <- available_priors[nzchar(available_priors)]
bad_priors <- setdiff(prior_filter, available_priors)
if (length(bad_priors) > 0) {
  stop(
    "Unknown --priors value(s): ", paste(bad_priors, collapse = ", "),
    ". Available values: ", paste(available_priors, collapse = ", "),
    call. = FALSE
  )
}

available_baseline_families <- sort(unique(c(
  as.character(coefs$method_family[!grepl("^Bayes", as.character(coefs$method_family))]),
  as.character(select_df$method_family[!grepl("^Bayes", as.character(select_df$method_family))])
)))
available_baseline_families <- available_baseline_families[nzchar(available_baseline_families)]
normalize_baseline_family_filter <- function(values, available) {
  if (is_all_filter(values)) {
    return(character())
  }
  requested <- unique(setdiff(trimws(values), c("", "all")))
  lookup <- stats::setNames(available, tolower(available))
  requested_keys <- tolower(requested)
  bad <- requested[!(requested_keys %in% names(lookup))]
  if (length(bad) > 0) {
    stop(
      "Unknown --baseline-families value(s): ", paste(bad, collapse = ", "),
      ". Available values: ", paste(available, collapse = ", "),
      call. = FALSE
    )
  }
  unname(lookup[requested_keys])
}
baseline_family_filter <- normalize_baseline_family_filter(
  baseline_family_filter_arg,
  available_baseline_families
)

apply_method_filters <- function(df) {
  if (nrow(df) == 0) {
    return(df)
  }
  is_bayes <- grepl("^Bayes", as.character(df$method_family))
  keep <- rep(TRUE, nrow(df))
  if (length(prior_filter) > 0) {
    keep[is_bayes] <- as.character(df$prior[is_bayes]) %in% prior_filter
  }
  if (length(baseline_family_filter) > 0) {
    keep[!is_bayes] <- as.character(df$method_family[!is_bayes]) %in% baseline_family_filter
  }
  df[keep, , drop = FALSE]
}

coefs <- apply_method_filters(coefs)
select_df <- apply_method_filters(select_df)
rhats <- apply_method_filters(rhats)

select_coef_signs <- coefs %>%
  distinct(dataset_id, variable_index, method_family, method, prior, constraint, .keep_all = TRUE) %>%
  dplyr::select(
    dataset_id,
    variable_index,
    method_family,
    method,
    prior,
    constraint,
    select_point_estimate = point_estimate
  )

select_df <- select_df %>%
  left_join(
    select_coef_signs,
    by = c("dataset_id", "variable_index", "method_family", "method", "prior", "constraint")
  )

if (nrow(coefs) == 0 && nrow(select_df) == 0 && nrow(rhats) == 0) {
  stop("No plot rows remain after applying method filters.", call. = FALSE)
}
if (length(prior_filter) > 0) {
  message("Plot prior filter: ", paste(prior_filter, collapse = ", "))
}
if (length(baseline_family_filter) > 0) {
  message("Plot baseline family filter: ", paste(baseline_family_filter, collapse = ", "))
}

forest_dir <- ensure_dir(file.path(summary_dir, "figures", "forest"))
rhat_dir <- ensure_dir(file.path(summary_dir, "figures", "rhat"))
select_dir <- ensure_dir(file.path(summary_dir, "figures", "select"))
combined_dir <- if (length(combined_layouts) > 0) {
  ensure_dir(file.path(summary_dir, "figures", "forest_rhat"))
} else {
  NULL
}

ordered_present <- function(values, preferred_order) {
  present <- unique(as.character(values))
  present <- present[!is.na(present) & nzchar(present)]
  ordered <- preferred_order[preferred_order %in% present]
  c(ordered, setdiff(sort(present), ordered))
}

display_color_values <- c(
  "Cox-HS" = "#0072B2",
  "Cox-RHS" = "#D55E00",
  "Prox-LASSO" = "#009E73",
  "Prox-SSLASSO" = "#CC79A7",
  "FLORAL" = "#E69F00",
  "coda4microbiome" = "#000000"
)
display_shape_values <- c(
  "prox" = 15,
  "soft-center" = 3,
  "FLORAL" = 17,
  "coda4microbiome" = 16
)

display_color_breaks <- ordered_present(
  c(coefs$color_group, select_df$color_group, rhats$color_group),
  names(display_color_values)
)
display_shape_breaks <- ordered_present(
  c(coefs$shape_group, select_df$shape_group, rhats$shape_group),
  names(display_shape_values)
)
display_color_labels <- stats::setNames(display_color_breaks, display_color_breaks)
select_method_letters <- stats::setNames(LETTERS[seq_along(display_color_breaks)], display_color_breaks)
select_color_labels <- display_color_labels
select_color_labels[display_color_breaks] <- paste0("(", select_method_letters, ") ", display_color_breaks)

fixed_legend_height <- function(scale, text_scale = 1, row_height = 0.34) {
  color_cols <- legend_ncol(length(display_color_breaks), max_cols = 6)
  shape_cols <- legend_ncol(length(display_shape_breaks), max_cols = 4)
  legend_extra_height(length(display_color_breaks), color_cols, scale * text_scale, row_height = row_height, padding = 0.25) +
    legend_extra_height(length(display_shape_breaks), shape_cols, scale * text_scale, row_height = row_height, padding = 0.15)
}

combined_width_scale_for_layout <- function(layout) {
  if (layout == "stacked") {
    return(max(forest_width_scale, rhat_width_scale))
  }
  mean(c(forest_width_scale, rhat_width_scale))
}

combined_panel_height <- function(n_var, scale, height_scale = forest_height_scale) {
  max(7, 2.5 + 0.17 * n_var * scale / 1.5) * combined_height_scale * height_scale
}

combined_shared_panel_height <- function(n_var, scale) {
  combined_panel_height(n_var, scale, max(forest_height_scale, rhat_height_scale))
}

combined_rhat_panel_height <- function() {
  7.2 * combined_height_scale * rhat_height_scale
}

combined_method_width <- function(n_method, layout) {
  width_scale <- combined_width_scale_for_layout(layout)
  if (layout %in% c("aligned", "summary")) {
    return(max(30, min(36, 12 + 0.75 * n_method)) * width_scale)
  }
  max(22, min(28, 7.5 + 0.22 * n_method)) * width_scale
}

combined_dimensions <- function(combo, layout, scale) {
  n_method <- length(combo$method_levels)
  legend_height <- fixed_legend_height(scale, text_scale = combined_legend_text_scale, row_height = 0.34)
  panel_height <- if (layout == "stacked") {
    combined_panel_height(combo$n_var, scale, forest_height_scale)
  } else {
    combined_shared_panel_height(combo$n_var, scale)
  }
  if (layout == "stacked") {
    return(list(
      width = combined_method_width(n_method, layout),
      height = panel_height + combined_rhat_panel_height() + legend_height
    ))
  }
  list(
    width = combined_method_width(n_method, layout),
    height = panel_height + legend_height
  )
}

combined_color_scale <- function(method_levels = NULL, labels = display_color_labels) {
  scale_color_manual(
    values = display_color_values,
    limits = display_color_breaks,
    breaks = display_color_breaks,
    labels = unname(labels[display_color_breaks]),
    drop = FALSE
  )
}

combined_shape_scale <- function() {
  scale_shape_manual(
    values = display_shape_values,
    limits = display_shape_breaks,
    breaks = display_shape_breaks,
    drop = FALSE
  )
}

combined_guides <- function(n_method = NULL) {
  legend_cols <- legend_ncol(length(display_color_breaks), max_cols = 6)
  guides(
    color = guide_legend(ncol = legend_cols, byrow = TRUE, override.aes = list(shape = 16, size = 2.8, alpha = 1)),
    shape = guide_legend(nrow = 1, byrow = TRUE, override.aes = list(color = "grey25", size = 2.8, alpha = 1))
  )
}

combined_axis_text_size <- function(n_var, scale) {
  font_size(ifelse(n_var > 100, 5.5, ifelse(n_var > 80, 6.5, 8.5)), scale) * combined_axis_text_scale
}

combined_axis_title_size <- function(scale) {
  font_size(14, scale) * combined_axis_title_scale
}

combined_legend_text_size <- function(scale) {
  font_size(9, scale) * combined_legend_text_scale
}

combined_strip_text_size <- function(scale) {
  font_size(11, scale) * combined_strip_text_scale
}

combined_dataset_ids <- function() {
  available <- intersect(sort(unique(coefs$dataset_id)), sort(unique(rhats$dataset_id)))
  requested <- combined_datasets
  if (length(requested) == 0 || any(tolower(requested) == "all")) {
    return(available)
  }
  missing <- setdiff(requested, available)
  if (length(missing) > 0) {
    warning("Skipping unknown combined dataset(s): ", paste(missing, collapse = ", "), call. = FALSE)
  }
  intersect(requested, available)
}

prepare_combined_data <- function(dataset) {
  forest_dat <- coefs %>%
    filter(dataset_id == dataset) %>%
    arrange(variable_index, method_family, method)
  rhat_dat <- rhats %>%
    filter(dataset_id == dataset, !is.na(rhat)) %>%
    arrange(variable_index, constraint, prior)
  if (nrow(forest_dat) == 0 || nrow(rhat_dat) == 0) {
    return(NULL)
  }

  var_table <- variable_level_table(forest_dat)
  var_levels <- var_table$variable
  var_labels <- variable_label_map(var_table)
  method_levels <- unique(c(forest_dat$method_label, rhat_dat$method_label))
  rhat_xmin <- min(0.98, min(rhat_dat$rhat, na.rm = TRUE))
  rhat_xmax <- max(1.12, stats::quantile(rhat_dat$rhat, 0.99, na.rm = TRUE) * 1.03)

  forest_dat <- forest_dat %>%
    mutate(
      variable = factor(variable, levels = rev(var_levels)),
      method_label = factor(method_label, levels = method_levels),
      color_group = factor(color_group, levels = display_color_breaks),
      shape_group = factor(shape_group, levels = display_shape_breaks)
    )
  rhat_dat <- rhat_dat %>%
    filter(variable_index %in% forest_dat$variable_index) %>%
    mutate(
      variable = factor(variable, levels = rev(var_levels)),
      method_label = factor(method_label, levels = method_levels),
      color_group = factor(color_group, levels = display_color_breaks),
      shape_group = factor(shape_group, levels = display_shape_breaks)
    )

  list(
    dataset = dataset,
    forest = forest_dat,
    rhat = rhat_dat,
    var_levels = var_levels,
    var_labels = var_labels,
    method_levels = method_levels,
    n_var = length(var_levels),
    rhat_xmin = rhat_xmin,
    rhat_xmax = rhat_xmax
  )
}

extract_plot_legend <- function(plot) {
  grob <- ggplotGrob(plot)
  guide_idx <- grep("^guide-box", grob$layout$name)
  for (idx in guide_idx) {
    guide <- grob$grobs[[idx]]
    if (!inherits(guide, "zeroGrob")) {
      return(guide)
    }
  }
  NULL
}

build_combined_forest_plot <- function(combo, show_legend = TRUE, scale = combined_font_scale) {
  dat <- combo$forest
  dodge <- position_dodge(width = 0.75)
  p <- ggplot(dat, aes(x = point_estimate, y = variable, color = color_group, shape = shape_group, group = method_label)) +
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
    combined_color_scale(combo$method_levels) +
    combined_shape_scale() +
    scale_y_discrete(limits = rev(combo$var_levels), labels = combo$var_labels, drop = FALSE) +
    labs(x = "Coefficient", y = NULL) +
    combined_guides(length(combo$method_levels)) +
    base_theme(scale) +
    theme(
      axis.text.y = element_text(size = combined_axis_text_size(combo$n_var, scale)),
      axis.text.x = element_text(size = font_size(11, scale) * combined_axis_text_scale),
      axis.title.x = element_text(size = combined_axis_title_size(scale)),
      legend.text = element_text(size = combined_legend_text_size(scale)),
      legend.key.height = unit(scale * 0.39, "cm"),
      legend.key.width = unit(0.42, "cm"),
      legend.spacing.x = unit(0.12, "cm"),
      legend.position = if (show_legend) "bottom" else "none"
    )
  p
}

build_combined_rhat_index_plot <- function(combo, show_legend = FALSE, scale = combined_font_scale) {
  dat <- combo$rhat
  p <- ggplot(dat, aes(x = variable_index, y = rhat, color = color_group, shape = shape_group, group = method_label)) +
    geom_hline(yintercept = 1.01, linetype = "dashed", color = "red", linewidth = 0.45) +
    geom_hline(yintercept = 1.1, linetype = "dashed", color = "red", linewidth = 0.45) +
    annotate("text", x = Inf, y = 1.01, label = "1.01", hjust = 1.05, vjust = -0.35, color = "red", size = font_size(3.4, scale), family = font_family) +
    annotate("text", x = Inf, y = 1.1, label = "1.1", hjust = 1.05, vjust = -0.35, color = "red", size = font_size(3.4, scale), family = font_family) +
    geom_point(size = 1.7, alpha = 0.82) +
    combined_color_scale(combo$method_levels) +
    combined_shape_scale() +
    coord_cartesian(ylim = c(combo$rhat_xmin, combo$rhat_xmax)) +
    labs(x = "Variable index", y = "Rhat") +
    combined_guides(length(combo$method_levels)) +
    base_theme(scale) +
    theme(
      axis.text.x = element_text(size = font_size(11, scale) * combined_axis_text_scale),
      axis.text.y = element_text(size = font_size(11, scale) * combined_axis_text_scale),
      axis.title.x = element_text(size = combined_axis_title_size(scale)),
      axis.title.y = element_text(size = combined_axis_title_size(scale)),
      legend.text = element_text(size = combined_legend_text_size(scale)),
      legend.key.height = unit(scale * 0.39, "cm"),
      legend.key.width = unit(0.42, "cm"),
      legend.spacing.x = unit(0.12, "cm"),
      legend.position = if (show_legend) "bottom" else "none"
    )
  p
}

build_stacked_combined_plot <- function(combo, scale = combined_font_scale) {
  legend <- extract_plot_legend(build_combined_forest_plot(combo, show_legend = TRUE, scale = scale))
  forest_grob <- ggplotGrob(build_combined_forest_plot(combo, show_legend = FALSE, scale = scale))
  rhat_grob <- ggplotGrob(build_combined_rhat_index_plot(combo, show_legend = FALSE, scale = scale))
  grobs <- list(forest_grob, rhat_grob)
  heights <- grid::unit(c(
    combined_panel_height(combo$n_var, scale, forest_height_scale),
    combined_rhat_panel_height()
  ), "null")
  if (!is.null(legend)) {
    grobs <- c(grobs, list(legend))
    heights <- grid::unit.c(heights, sum(legend$heights) + grid::unit(0.45, "cm"))
  }
  gtable::gtable_matrix(
    "forest_rhat_stacked",
    grobs = matrix(grobs, ncol = 1),
    widths = grid::unit(1, "null"),
    heights = heights
  )
}

combined_facet_theme <- function(n_var, scale) {
  theme(
    axis.text.y = element_text(size = combined_axis_text_size(n_var, scale)),
    axis.text.x = element_text(size = font_size(10, scale) * combined_axis_text_scale),
    axis.title.x = element_text(size = combined_axis_title_size(scale)),
    axis.title.y = element_text(size = combined_axis_title_size(scale)),
    legend.text = element_text(size = combined_legend_text_size(scale)),
    legend.key.height = unit(scale * 0.39, "cm"),
    legend.key.width = unit(0.42, "cm"),
    legend.spacing.x = unit(0.12, "cm"),
    panel.spacing.x = unit(0.45, "cm"),
    strip.text = element_text(size = combined_strip_text_size(scale), face = "bold")
  )
}

build_aligned_combined_plot <- function(combo, scale = combined_font_scale) {
  panel_levels <- c("Coefficient", "Rhat")
  forest_dat <- combo$forest %>%
    mutate(panel = factor("Coefficient", levels = panel_levels))
  rhat_dat <- combo$rhat %>%
    mutate(panel = factor("Rhat", levels = panel_levels))
  rhat_limits <- data.frame(
    panel = factor("Rhat", levels = panel_levels),
    x = c(combo$rhat_xmin, combo$rhat_xmax),
    variable = factor(combo$var_levels[[1]], levels = rev(combo$var_levels))
  )
  coef_ref <- data.frame(panel = factor("Coefficient", levels = panel_levels), xintercept = 0)
  rhat_ref <- data.frame(panel = factor("Rhat", levels = panel_levels), xintercept = c(1.0, 1.1))
  dodge <- position_dodge(width = 0.75)

  ggplot() +
    geom_blank(data = rhat_limits, aes(x = x, y = variable), inherit.aes = FALSE) +
    geom_vline(data = coef_ref, aes(xintercept = xintercept), linetype = "dashed", linewidth = 0.35, color = "grey35") +
    geom_vline(data = rhat_ref, aes(xintercept = xintercept), linetype = "dashed", linewidth = 0.45, color = "red") +
    geom_errorbar(
      data = forest_dat %>% filter(interval_available, !is.na(lower), !is.na(upper)),
      aes(x = point_estimate, y = variable, xmin = lower, xmax = upper, color = color_group, group = method_label),
      orientation = "y",
      width = 0,
      linewidth = 0.25,
      alpha = 0.42,
      position = dodge,
      na.rm = TRUE
    ) +
    geom_point(
      data = forest_dat,
      aes(x = point_estimate, y = variable, color = color_group, shape = shape_group, group = method_label),
      size = 1.2,
      alpha = 0.9,
      position = dodge,
      na.rm = TRUE
    ) +
    geom_point(
      data = rhat_dat,
      aes(x = rhat, y = variable, color = color_group, shape = shape_group, group = method_label),
      size = 1.45,
      alpha = 0.82,
      position = dodge,
      na.rm = TRUE
    ) +
    facet_grid(. ~ panel, scales = "free_x") +
    combined_color_scale(combo$method_levels) +
    combined_shape_scale() +
    scale_y_discrete(limits = rev(combo$var_levels), labels = combo$var_labels, drop = FALSE) +
    labs(x = NULL, y = NULL) +
    combined_guides(length(combo$method_levels)) +
    base_theme(scale) +
    combined_facet_theme(combo$n_var, scale)
}

build_summary_combined_plot <- function(combo, scale = combined_font_scale) {
  panel_levels <- c("Coefficient", "max Rhat")
  forest_dat <- combo$forest %>%
    mutate(panel = factor("Coefficient", levels = panel_levels))
  rhat_summary <- combo$rhat %>%
    group_by(variable_index, variable) %>%
    summarise(
      max_rhat = max(rhat, na.rm = TRUE),
      over_threshold = any(rhat > 1.1, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      variable = factor(as.character(variable), levels = rev(combo$var_levels)),
      panel = factor("max Rhat", levels = panel_levels)
    )
  rhat_limits <- data.frame(
    panel = factor("max Rhat", levels = panel_levels),
    x = c(combo$rhat_xmin, combo$rhat_xmax),
    variable = factor(combo$var_levels[[1]], levels = rev(combo$var_levels))
  )
  coef_ref <- data.frame(panel = factor("Coefficient", levels = panel_levels), xintercept = 0)
  rhat_ref <- data.frame(panel = factor("max Rhat", levels = panel_levels), xintercept = c(1.0, 1.1))
  dodge <- position_dodge(width = 0.75)

  ggplot() +
    geom_blank(data = rhat_limits, aes(x = x, y = variable), inherit.aes = FALSE) +
    geom_vline(data = coef_ref, aes(xintercept = xintercept), linetype = "dashed", linewidth = 0.35, color = "grey35") +
    geom_vline(data = rhat_ref, aes(xintercept = xintercept), linetype = "dashed", linewidth = 0.45, color = "red") +
    geom_errorbar(
      data = forest_dat %>% filter(interval_available, !is.na(lower), !is.na(upper)),
      aes(x = point_estimate, y = variable, xmin = lower, xmax = upper, color = color_group, group = method_label),
      orientation = "y",
      width = 0,
      linewidth = 0.25,
      alpha = 0.42,
      position = dodge,
      na.rm = TRUE
    ) +
    geom_point(
      data = forest_dat,
      aes(x = point_estimate, y = variable, color = color_group, shape = shape_group, group = method_label),
      size = 1.2,
      alpha = 0.9,
      position = dodge,
      na.rm = TRUE
    ) +
    geom_segment(
      data = rhat_summary,
      aes(x = 1.0, xend = max_rhat, y = variable, yend = variable),
      color = "grey78",
      linewidth = 0.35,
      inherit.aes = FALSE
    ) +
    geom_point(
      data = rhat_summary,
      aes(x = max_rhat, y = variable),
      color = "grey25",
      size = 1.7,
      inherit.aes = FALSE
    ) +
    geom_point(
      data = rhat_summary %>% filter(over_threshold),
      aes(x = max_rhat, y = variable),
      color = "red",
      size = 1.9,
      inherit.aes = FALSE
    ) +
    facet_grid(. ~ panel, scales = "free_x") +
    combined_color_scale(combo$method_levels) +
    combined_shape_scale() +
    scale_y_discrete(limits = rev(combo$var_levels), labels = combo$var_labels, drop = FALSE) +
    labs(x = NULL, y = NULL) +
    combined_guides(length(combo$method_levels)) +
    base_theme(scale) +
    combined_facet_theme(combo$n_var, scale)
}

if (!combined_only) {
for (dataset in sort(unique(coefs$dataset_id))) {
  dat <- coefs %>%
    filter(dataset_id == dataset) %>%
    arrange(variable_index, method_family, method)
  if (nrow(dat) == 0) next

  var_table <- variable_level_table(dat)
  var_levels <- var_table$variable
  var_labels <- variable_label_map(var_table)
  method_levels <- unique(dat$method_label)

  dat <- dat %>%
    mutate(
      variable = factor(variable, levels = rev(var_levels)),
      method_label = factor(method_label, levels = method_levels),
      color_group = factor(color_group, levels = display_color_breaks),
      shape_group = factor(shape_group, levels = display_shape_breaks)
    )

  n_var <- length(var_levels)
  n_method <- length(method_levels)
  height <- max(9, 2.5 + 0.25 * n_var * forest_font_scale / 1.5)
  width <- max(20, min(28, 7.5 + 0.2 * n_method)) * forest_width_scale
  forest_legend_cols <- legend_ncol(length(display_color_breaks), max_cols = 6)
  forest_height <- (
    height + fixed_legend_height(forest_font_scale, row_height = 0.34)
  ) * forest_height_scale
  dodge <- position_dodge(width = 0.75)

  p <- ggplot(dat, aes(x = point_estimate, y = variable, color = color_group, shape = shape_group, group = method_label)) +
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
    combined_color_scale(method_levels) +
    combined_shape_scale() +
    scale_y_discrete(limits = rev(var_levels), labels = var_labels, drop = FALSE) +
    labs(x = "Coefficient", y = NULL) +
    guides(
      color = guide_legend(ncol = forest_legend_cols, byrow = TRUE, override.aes = list(shape = 16, size = 2.4, alpha = 1)),
      shape = guide_legend(nrow = 1, byrow = TRUE, override.aes = list(color = "grey25", size = 2.4, alpha = 1))
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
    mutate(
      method_label = factor(method_label, levels = unique(method_label)),
      color_group = factor(color_group, levels = display_color_breaks),
      shape_group = factor(shape_group, levels = display_shape_breaks)
    )

  y_max <- max(1.12, stats::quantile(dat$rhat, 0.99, na.rm = TRUE) * 1.03)
  rhat_n_method <- dplyr::n_distinct(dat$method_label)
  rhat_width <- max(22, min(28, 7.5 + 0.2 * rhat_n_method)) * rhat_width_scale
  rhat_legend_cols <- legend_ncol(length(display_color_breaks), max_cols = 6)
  rhat_height <- (
    7.2 + fixed_legend_height(rhat_font_scale, row_height = 0.34)
  ) * rhat_height_scale
  p <- ggplot(dat, aes(x = variable_index, y = rhat, color = color_group, shape = shape_group, group = method_label)) +
    geom_hline(yintercept = 1.01, linetype = "dashed", color = "red", linewidth = 0.45) +
    geom_hline(yintercept = 1.1, linetype = "dashed", color = "red", linewidth = 0.45) +
    annotate("text", x = Inf, y = 1.01, label = "1.01", hjust = 1.05, vjust = -0.35, color = "red", size = font_size(3.4, rhat_font_scale), family = font_family) +
    annotate("text", x = Inf, y = 1.1, label = "1.1", hjust = 1.05, vjust = -0.35, color = "red", size = font_size(3.4, rhat_font_scale), family = font_family) +
    geom_point(size = 1.7, alpha = 0.82) +
    combined_color_scale(unique(dat$color_group)) +
    combined_shape_scale() +
    coord_cartesian(ylim = c(min(0.98, min(dat$rhat, na.rm = TRUE)), y_max)) +
    labs(x = "Variable index", y = "Rhat") +
    guides(
      color = guide_legend(ncol = rhat_legend_cols, byrow = TRUE, override.aes = list(shape = 16, size = 2.5, alpha = 1)),
      shape = guide_legend(nrow = 1, byrow = TRUE, override.aes = list(color = "grey25", size = 2.5, alpha = 1))
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
    mutate(sign_label = coef_sign_label(select_point_estimate))
  if (nrow(dat) == 0) next

  var_table <- variable_level_table(dat)
  var_levels <- var_table$variable
  var_labels <- as.character(var_table$variable_label)
  method_map <- dat %>%
    distinct(method_label, method_family, method, color_group, shape_group) %>%
    mutate(
      is_bayes = grepl("^Bayes", as.character(method_family)),
      color_order = match(as.character(color_group), display_color_breaks),
      shape_order = match(as.character(shape_group), display_shape_breaks),
      baseline_order = dplyr::case_when(
        grepl("\\(min\\)$", as.character(method_label)) ~ 1L,
        grepl("\\(1se\\)$", as.character(method_label)) ~ 2L,
        TRUE ~ 99L
      ),
      within_group_order = ifelse(is_bayes, shape_order, baseline_order)
    ) %>%
    arrange(ifelse(is_bayes, 0L, 1L), color_order, within_group_order, method_label) %>%
    mutate(method_index = dplyr::row_number()) %>%
    dplyr::select(method_label, color_group, shape_group, method_index, is_bayes)

  method_levels <- method_map$method_label
  axis_map <- method_map %>%
    group_by(color_group) %>%
    summarise(method_axis_x = mean(method_index), .groups = "drop") %>%
    mutate(method_axis_label = unname(select_method_letters[as.character(color_group)])) %>%
    filter(!is.na(method_axis_label)) %>%
    arrange(method_axis_x)

  panel_levels <- c("Selection", "Coefficient sign")

  dat <- dat %>%
    left_join(method_map, by = c("method_label", "color_group", "shape_group")) %>%
    mutate(
      method_index = as.integer(method_index),
      color_group = factor(color_group, levels = display_color_breaks),
      shape_group = factor(shape_group, levels = display_shape_breaks),
      panel = factor("Selection", levels = panel_levels)
    )

  n_var <- length(var_levels)
  n_method <- length(method_levels)
  y_label_width <- max(2.8, min(8.0, 1.0 + 0.055 * select_layout_scale * max(nchar(var_labels))))
  select_legend_cols <- legend_ncol(length(display_color_breaks), max_cols = 6)
  panel_width <- max(9.5, min(18.0, 5.5 + 0.48 * n_method))
  plot_width <- min(42.0, y_label_width + 2 * panel_width + 1.2)
  select_height <- max(8.5, min(34.0, 4.8 + 0.25 * n_var)) +
    fixed_legend_height(select_font_scale, row_height = 0.28) + 0.8
  sign_text_size <- font_size(ifelse(n_var > 100, 1.8, ifelse(n_var > 60, 2.1, 2.5)), select_font_scale)
  legend_seed <- method_map %>%
    mutate(
      color_group = factor(color_group, levels = display_color_breaks),
      shape_group = factor(shape_group, levels = display_shape_breaks),
      variable_index = 1,
      panel = factor("Selection", levels = panel_levels)
    )
  sign_dat <- dat %>%
    mutate(panel = factor("Coefficient sign", levels = panel_levels))
  cell_grid <- expand.grid(
    panel = factor(panel_levels, levels = panel_levels),
    xintercept = seq(0.5, n_method + 0.5, by = 1),
    KEEP.OUT.ATTRS = FALSE
  )
  row_grid <- expand.grid(
    panel = factor(panel_levels, levels = panel_levels),
    yintercept = seq(0.5, n_var + 0.5, by = 1),
    KEEP.OUT.ATTRS = FALSE
  )

  p <- ggplot() +
    geom_vline(
      data = cell_grid,
      aes(xintercept = xintercept),
      color = "grey92",
      linewidth = 0.25,
      inherit.aes = FALSE
    ) +
    geom_hline(
      data = row_grid,
      aes(yintercept = yintercept),
      color = "grey92",
      linewidth = 0.25,
      inherit.aes = FALSE
    ) +
    geom_point(
      data = legend_seed,
      aes(x = method_index, y = variable_index, color = color_group, shape = shape_group),
      inherit.aes = FALSE,
      size = select_point_size,
      alpha = 0,
      show.legend = TRUE
    ) +
    geom_point(
      data = dat %>% filter(selected),
      aes(x = method_index, y = variable_index, color = color_group, shape = shape_group),
      inherit.aes = FALSE,
      size = select_point_size,
      alpha = 0.9
    ) +
    geom_text(
      data = sign_dat,
      aes(x = method_index, y = variable_index, label = sign_label),
      color = "grey20",
      family = font_family,
      size = sign_text_size,
      inherit.aes = FALSE,
      na.rm = TRUE
    ) +
    scale_x_continuous(
      breaks = axis_map$method_axis_x,
      labels = axis_map$method_axis_label,
      limits = c(0.5, n_method + 0.5),
      expand = c(0, 0)
    ) +
    scale_y_reverse(
      breaks = seq_along(var_levels),
      labels = var_labels,
      limits = c(n_var + 0.5, 0.5),
      expand = c(0, 0)
    ) +
    combined_color_scale(method_levels, labels = select_color_labels) +
    combined_shape_scale() +
    facet_grid(. ~ panel) +
    coord_cartesian(clip = "off") +
    labs(x = NULL, y = NULL) +
    guides(
      color = guide_legend(ncol = select_legend_cols, byrow = TRUE, override.aes = list(shape = 16, size = select_legend_point_size, alpha = 1)),
      shape = guide_legend(nrow = 1, byrow = TRUE, override.aes = list(color = "grey25", size = select_legend_point_size, alpha = 1))
    ) +
    base_theme(select_font_scale) +
    theme(
      panel.grid.major.x = element_blank(),
      panel.grid.major.y = element_blank(),
      axis.text.x = element_text(
        angle = 0,
        hjust = 0.5,
        vjust = 0.5,
        size = font_size(ifelse(n_method > 24, 7.0, 8.0), select_font_scale)
      ),
      axis.text.y = element_text(size = font_size(ifelse(n_var > 100, 6.0, ifelse(n_var > 60, 7.0, 8.0)), select_font_scale)),
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.box = "vertical",
      legend.justification = "center",
      legend.text = element_text(size = font_size(8.5, select_font_scale)),
      legend.title = element_blank(),
      legend.key.height = unit(select_font_scale * 0.28, "cm"),
      legend.key.width = unit(select_font_scale * 0.42, "cm"),
      panel.spacing.x = unit(0.55, "cm"),
      strip.text = element_text(size = font_size(8.8, select_font_scale), face = "bold")
    )

  save_plot_bundle(p, file.path(select_dir, paste0(safe_name(dataset), "_select")), plot_width, select_height)
}

}

if (length(combined_layouts) > 0) {
  for (dataset in combined_dataset_ids()) {
    combo <- prepare_combined_data(dataset)
    if (is.null(combo)) {
      next
    }
    for (layout in combined_layouts) {
      plot <- switch(
        layout,
        stacked = build_stacked_combined_plot(combo),
        aligned = build_aligned_combined_plot(combo),
        summary = build_summary_combined_plot(combo)
      )
      dims <- combined_dimensions(combo, layout, combined_font_scale)
      save_plot_bundle(
        plot,
        file.path(combined_dir, paste0(safe_name(dataset), "_forest_rhat_", layout)),
        dims$width,
        dims$height
      )
    }
  }
}

while (grDevices::dev.cur() > 1) {
  grDevices::dev.off()
}
if (!initial_rplots_exists && file.exists("Rplots.pdf")) {
  unlink("Rplots.pdf")
}
