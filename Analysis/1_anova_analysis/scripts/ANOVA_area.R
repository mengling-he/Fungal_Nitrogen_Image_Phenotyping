library(ggplot2)
library(tidyr)
library(dplyr)
library(ez)
library(rstatix)
library(ggplot2)
library(tidyr)


library(rstudioapi)
setwd(dirname(getActiveDocumentContext()$path))
source("rm_anova_batch_fn.R")

# ----read data----
df_wide <- read.csv("../data/metrics_wide.csv",stringsAsFactors = FALSE)
df_wide$Replication <- factor(df_wide$Replication)
colnames(df_wide)
str(df_wide$date)

##### generate a Day column  #####
df_wide$date <- as.Date(df_wide$date)
df_wide$Day <- as.numeric(df_wide$date - min(df_wide$date))+1
df_wide$Day_c <- df_wide$Day - mean(df_wide$Day)# center day, to see the effect in the day

df_wide$date <- as.factor(df_wide$date)

mask_order <- c("all", "all_plug", "plug", "new_growth", "old_growth")
view_order <- c("raw", "overhead")
make_trait_vector <- function(trait) {
  c(
    paste0(trait, "_all_raw"),
    paste0(trait, "_all_plug_raw"),
    paste0(trait, "_plug_raw"),
    paste0(trait, "_new_growth_raw"),
    paste0(trait, "_old_growth_raw"),
    
    paste0(trait, "_all_overhead"),
    paste0(trait, "_all_plug_overhead"),
    paste0(trait, "_plug_overhead"),
    paste0(trait, "_new_growth_overhead"),
    paste0(trait, "_old_growth_overhead")
  )
}

#----- log area  -----
df_wide <- df_wide %>%
  mutate(
    across(
      starts_with("area_pixels"),
      log1p,
      .names = "log_{.col}"
    )
  )
log_area_cols <- names(df_wide)[startsWith(names(df_wide), "log_area_pixels")]





#----- test for one trait-----
# anova _area_pixels_all_raw
outdir <- "../results/plot/"
y_name <- "area_pixels_all_raw"


target_trait <- y_name
df_use <- df_wide %>%
  group_by(date) %>%
  filter(sum(!is.na(.data[[target_trait]])) > 0) %>%
  ungroup()
df_use$Nitrogen_Level <- factor(df_use$Nitrogen_Level)
df_use$Nitrogen_Level <- factor(df_use$Nitrogen_Level,
                                levels = c("N-1", "N-10", "N-100"))





####  CHECK ASSUMPTIONS #######

# --- 2a. Normality (Shapiro-Wilk per group) ---
df_use %>%
  group_by(date) %>%
  shapiro_test(target_trait)
## A tibble: 12 × 4
# date       variable                statistic          p
# <fct>      <chr>                       <dbl>      <dbl>
#   1 2024-12-20 log_area_pixels_all_raw     0.743 0.00000692
# 2 2024-12-21 log_area_pixels_all_raw     0.903 0.0102    
# 3 2024-12-22 log_area_pixels_all_raw     0.919 0.0252    
# 4 2024-12-23 log_area_pixels_all_raw     0.936 0.0696    
# 5 2024-12-24 log_area_pixels_all_raw     0.907 0.0125    
# 6 2024-12-25 log_area_pixels_all_raw     0.854 0.000741  
# 7 2024-12-26 log_area_pixels_all_raw     0.872 0.00186   
# 8 2024-12-27 log_area_pixels_all_raw     0.893 0.00554   
# 9 2024-12-28 log_area_pixels_all_raw     0.803 0.0000753 
# 10 2024-12-29 log_area_pixels_all_raw     0.805 0.0000814 
# 11 2024-12-30 log_area_pixels_all_raw     0.809 0.0000959 
# 12 2024-12-31 log_area_pixels_all_raw     0.857 0.000860 




####  ezANOVA #######
#Two-Way Mixed ANOVA (also known as a Split-Plot ANOVA
model_2way <- ezANOVA(
  data    = df_use,
  dv      = log_area_pixels_all_raw,
  wid     = plate,
  within  = date,
  between = Fungal_Strain,       # between-subjects factor
  detailed = TRUE,
  type    = 3
)

print(model_2way)


model_2way_2 <- ezANOVA(
  data    = df_use,
  dv      = log_area_pixels_all_raw,
  wid     = plate,
  within  = date,
  between = Nitrogen_Level,       # between-subjects factor
  detailed = TRUE,
  type    = 3
)

print(model_2way_2)



####  aov_ez: Three-Way Mixed ANOVA. #######
rm_fit <- aov_ez(
  id      = "plate",
  dv      = target_trait,
  data    = df_use,
  within  = "date",
  between = c("Fungal_Strain", "Nitrogen_Level"),
  type    = 3
)

emm_strain <- emmeans(rm_fit, ~ Fungal_Strain | Nitrogen_Level * date)
pairs_df <- as.data.frame(pairs(emm_strain, adjust = "tukey"))
pairs_df$significant <- pairs_df$p.value < 0.05
plot_df <- pairs_df %>%
  mutate(
    Feature = target_trait,
    sig = p.value < 0.05,
    date_chr = as.character(date)
  ) %>%
  mutate(date = factor(date_chr, levels = sort(unique(date_chr))))

# Individual plot (optional)
p = ggplot(
  plot_df,
  aes(x = date, y = estimate,
      group = Nitrogen_Level, linetype = Nitrogen_Level)
) +
  geom_line(linewidth = 1) +
  geom_point(aes(shape = sig), size = 2.8) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  scale_shape_manual(values = c(`FALSE` = 1, `TRUE` = 16),
                     labels = c("ns", "sig")) +
  labs(
    title = paste0("FG-LE over time (Tukey) | trait: ", target_trait),
    x = "Date", y = "Contrast estimate",
    linetype = "Nitrogen", shape = "Significance"
  ) +
  theme_bw(base_size = 12) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "bottom")

ggsave(paste0(outdir, y_name, "_contrast_FG_LE.pdf"),plot =p,
       width = 10,
       height = 6,
       dpi = 300
)



emm_nitro <- emmeans(rm_fit, ~ Nitrogen_Level | Fungal_Strain * date)
pairs_df <- as.data.frame(pairs(emm_nitro, adjust = "tukey"))
pairs_df$significant <- pairs_df$p.value < 0.05
pairs_df2 <-as.data.frame(contrast(emm_nitro,
         method = list(
           "N-10 - N-1"   = c(-1, 1, 0),
           "N-100 - N-1"  = c(-1, 0, 1),
           "N-100 - N-10" = c(0, -1, 1)
         ),
         adjust = "tukey"))
pairs_df2$significant <- pairs_df2$p.value < 0.05
p = ggplot(pairs_df2,
       aes(x = date,
           y = estimate,
           color = Fungal_Strain,
           group = Fungal_Strain,
           shape = significant)) +   # <-- map shape

geom_line(size = 1) +

geom_point(size = 3) +

geom_hline(yintercept = 0, linetype = "dashed") +

facet_wrap(~ contrast, ncol = 1) +

scale_shape_manual(
  values = c(`FALSE` = 1, `TRUE` = 16),
  labels = c("ns", "sig")
) +

labs(
  title = "Area Differences between Nitrogen Level",
  x = "Date",
  y = "Estimated Nitrogen Level Difference",
  shape = "Significance",
  color = "Fungal Strain"
) +

theme_bw()+
  theme(
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),  # center title
    axis.text.x = element_text(angle = 45, hjust = 1),                  # rotate x labels
    strip.text = element_text(size = 12, face = "bold")
  )

ggsave(paste0(outdir, y_name, "_diff_Nitro.pdf"),plot =p,
       width = 10,
       height = 6,
       dpi = 300
)



















# code contrast strains
res <- run_rm_anova_contrast_plots(df_wide, log_area_cols)

res1 <- run_rm_anova_contrast_plots(df_wide, c("area_pixels_all_raw","area_pixels_new_growth_raw"))



res2 <- run_rm_anova_contrast_plots(df_wide, c("log_area_pixels_all_raw","log_area_pixels_new_growth_raw"))























