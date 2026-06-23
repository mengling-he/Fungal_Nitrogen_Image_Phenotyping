library(lme4)
library(dplyr)
library(tidyr)
library(ggplot2)
library(rstudioapi)
setwd(dirname(getActiveDocumentContext()$path))
source("variance_partition_lmm_fn.R")



# ---- read data ----
df_0 <- read.csv("../data/metrics_masks5_plus_full.csv", stringsAsFactors = FALSE)
df_0 <- df_0[, !(names(df_0) %in% c(
  "image_path", "mask_path", "scale_max_used"
))]
df_0$log_area <- log1p(df_0$area_pixels)
df_0 <- df_0 %>%
  mutate(
    Fungal_Strain  = sub("^[^_]+_([^_]+)_.*$", "\\1", plate),
    Nitrogen_Level = sub("^.*_([^_]+)_\\d+_.*$", "\\1", plate)
  )
colnames(df_0)


# date handling (keep as factor for splitting; OK)
df_0$date <- as.Date(df_0$date)
df_0$date <- as.factor(df_0$date)



meta_cols <- c(
  "plate", "date", "Fungal_Strain", "Nitrogen_Level",
  "view", "mask_type"
)

metric_cols <- setdiff(names(df_0), meta_cols)
df_long <- df_0 %>%
  pivot_longer(
    cols = any_of(metric_cols),
    names_to = "trait",
    values_to = "value"
  )



# CI for area-------------
df_area <- df_long %>%
  filter(
    mask_type != "plug",
    view != "overhead",
    trait %in% c("area_pixels")
  )
unique(df_area$trait)
unique(df_area$mask_type)
df_area <- df_area %>%
  mutate(
    Fungal_Strain  = sub("^[^_]+_([^_]+)_.*$", "\\1", plate),
    Nitrogen_Level = sub("^.*_([^_]+)_\\d+_.*$", "\\1", plate)
  )
sub <- df_area %>%
  filter(
    trait %in% unique(df_area$trait)[1:2],
    is.finite(value)
  ) %>%
  mutate(
    date = as.Date(date),
    mask_type = factor(
      mask_type,
      levels = c("all","all_plug","old_growth","new_growth")
    ),
    trait = factor(trait, levels = unique(df_area$trait)[1:2])
  )

sum_df <- sub %>%
  group_by(trait, mask_type, date, Fungal_Strain, Nitrogen_Level) %>%
  summarise(
    n = sum(!is.na(value)),
    mean = mean(value, na.rm = TRUE),
    ci = ci95(value),
    .groups = "drop"
  ) %>%
  mutate(
    lower = mean - ci,
    upper = mean + ci
  )

p <- ggplot(
  sum_df,
  aes(
    x = date, y = mean,
    color = Fungal_Strain, fill = Fungal_Strain,
    linetype = Nitrogen_Level, shape = Nitrogen_Level,
    group = interaction(Fungal_Strain, Nitrogen_Level)
  )
) +
  geom_ribbon(aes(ymin = lower, ymax = upper),
              alpha = 0.18, color = NA) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 1.8) +
  facet_grid(trait ~ mask_type, scales = "free_y") +
  labs(
    title = "Mean ± 95% CI over time",
    x = "Date", y = "Mean value",
    color = "Fungal strain", fill = "Fungal strain",
    linetype = "Nitrogen", shape = "Nitrogen"
  ) +
  theme_bw() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 7),
    strip.text = element_text(size = 9, face = "bold"),
    panel.grid.major.x = element_blank(),
    legend.position = "bottom"
  )

#ggsave("../results/plot/area_time_plot.png", p, width = 10, height = 6)
ggsave(
  "../results/plot/area_time_plot.pdf",
  p,
  width = 10,
  height = 6,
  dpi = 300
)
