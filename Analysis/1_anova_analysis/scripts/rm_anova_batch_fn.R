library(afex)
library(emmeans)
library(dplyr)
library(ggplot2)

afex_options(type = 3)


# Shared base for the contrast-over-time plots (individual + combined facet).
.contrast_base_plot <- function(data, line_width, point_size) {
  ggplot(
    data,
    aes(x = date, y = estimate,
        group = Nitrogen_Level, linetype = Nitrogen_Level)
  ) +
    geom_line(linewidth = line_width) +
    geom_point(aes(shape = sig), size = point_size) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    scale_shape_manual(values = c(`FALSE` = 1, `TRUE` = 16),
                       labels = c("ns", "sig")) +
    theme_bw(base_size = 12)
}


run_rm_anova_contrast_plots <- function(df_wide,
                                        traits,
                                        out_dir = "../results/plot",
                                        alpha = 0.05,
                                        contrast_pattern = "FG\\s*-\\s*LE",
                                        save_individual = FALSE,
                                        save_combined_png = TRUE,
                                        combined_png_name = "contrast_FG-LE_combined.png",
                                        indiv_width = 10, indiv_height = 5, dpi = 300,
                                        combined_width = 14, combined_height = 8) {
  
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
  
  run_one_trait <- function(target_trait) {
    
    if (!target_trait %in% names(df_wide)) {
      stop("Trait column not found in df_wide: ", target_trait)
    }
    
    df_use <- df_wide %>%
      group_by(date) %>%
      filter(sum(!is.na(.data[[target_trait]])) > 0) %>%
      ungroup()
    
    rm_fit <- aov_ez(
      id      = "plate",
      dv      = target_trait,
      data    = df_use,
      within  = "date",
      between = c("Fungal_Strain", "Nitrogen_Level"),
      type    = 3
    )
    
    emm <- emmeans(rm_fit, ~ Fungal_Strain | Nitrogen_Level * date)
    pairs_df <- as.data.frame(pairs(emm, adjust = "tukey"))
    
    # Keep only FG - LE
    if ("contrast" %in% names(pairs_df)) {
      pairs_df <- pairs_df %>% filter(grepl(contrast_pattern, contrast))
    }
    
    plot_df <- pairs_df %>%
      mutate(
        Feature = target_trait,
        sig = p.value < alpha,
        date = factor(as.character(date),
                      levels = sort(unique(as.character(date))))
      )

    # Individual plot (optional)
    p_indiv <- .contrast_base_plot(plot_df, line_width = 1, point_size = 2.8) +
      labs(
        title = paste0("FG-LE over time (Tukey) | trait: ", target_trait),
        x = "Date", y = "Contrast estimate",
        linetype = "Nitrogen", shape = "Significance"
      ) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1),
            legend.position = "bottom")
    
    out_file <- NA_character_
    if (isTRUE(save_individual)) {
      safe_name <- gsub("[^A-Za-z0-9_\\-]+", "_", target_trait)
      out_file <- file.path(out_dir, paste0("contrast_FG-LE_", safe_name, ".png"))
      ggsave(out_file, p_indiv, width = indiv_width, height = indiv_height, dpi = dpi)
    }
    
    list(
      trait = target_trait,
      rm_fit = rm_fit,
      emm = emm,
      pairs_df = pairs_df,
      plot_df = plot_df,     # <-- key: used for combined plot
      plot = p_indiv,
      out_file = out_file
    )
  }
  
  results <- setNames(vector("list", length(traits)), traits)
  errors  <- list()
  
  for (tr in traits) {
    message("Running: ", tr)
    results[[tr]] <- tryCatch(
      run_one_trait(tr),
      error = function(e) {
        errors[[tr]] <<- e$message
        NULL
      }
    )
  }
  
  # ---- Combined plot (faceted) ----
  combined_file <- NA_character_
  all_pairs_df <- bind_rows(lapply(results, function(x) if (is.null(x)) NULL else x$plot_df))
  
  p_combined <- NULL
  if (isTRUE(save_combined_png) && nrow(all_pairs_df) > 0) {
    
    # Keep trait order as input list
    all_pairs_df$Feature <- factor(all_pairs_df$Feature, levels = traits)
    
    # With <= 8 traits, 4 columns is usually readable
    ncol_facets <- min(5, length(unique(all_pairs_df$Feature)))
    
    p_combined <- .contrast_base_plot(all_pairs_df, line_width = 0.9, point_size = 2.2) +
      facet_wrap(~ Feature, ncol = ncol_facets, scales = "free_y") +
      labs(
        title = paste0("Strain contrast FG-LE over time (Tukey), alpha=", alpha),
        x = "Date",
        y = "Contrast estimate",
        linetype = "Nitrogen",
        shape = "Significance"
      ) +
      theme_bw(base_size = 12) +
      theme(
        axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
        strip.text = element_text(size = 10, face = "bold"),
        legend.position = "bottom"
      )
    
    combined_file <- file.path(out_dir, combined_png_name)
    ggsave(combined_file, p_combined, width = combined_width, height = combined_height, dpi = dpi)
  }
  
  list(
    results = results,
    errors = errors,
    traits_requested = traits,
    out_dir = out_dir,
    alpha = alpha,
    combined_plot = p_combined,
    combined_file = combined_file,
    all_pairs_df = all_pairs_df
  )
}

