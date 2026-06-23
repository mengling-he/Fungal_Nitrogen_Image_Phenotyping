library(fda)
library(magic)
library(dplyr)
library(tidyr)
library(ggplot2)
library(rstudioapi)
library(lmerTest)
library(emmeans)
library(purrr)
if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}
source("../FPCA_code/fpca_analysis_single.R")

#outdir <- "../results/test"

df_wide <- read.csv("../data/metrics_wide.csv", stringsAsFactors = FALSE)
# make date numerical
df_clean <- df_wide
names(df_clean)





  
  
####### Select a list of traits#######
traits <- grep("(all_raw|new_growth_raw)$", colnames(df_wide), value = TRUE)

results <- purrr::map(traits, function(tr) {
  tryCatch(
    run_fpca_trait(df = df_wide, trait = tr, K_est = 2),
    error = function(e) {
      message("Skipping trait: ", tr, " — ", conditionMessage(e))
      NULL
    }
  )
}) %>% 
  setNames(traits)

# Remove failed traits (NULL entries)
results_ok <- purrr::compact(results)  # drops all NULLs

# See which traits failed
failed_traits <- names(results)[sapply(results, is.null)]
message(length(failed_traits), " traits failed: ", paste(failed_traits, collapse = ", "))

# See which succeeded
succeeded_traits <- names(results_ok)
message(length(succeeded_traits), " traits succeeded")
# All traits' score data frames combined
all_scores <- purrr::map_dfr(results_ok, function(res) res$scores$PC_df, .id = "trait")
fpca_scores_df <- all_scores

fpca_scores_df <- fpca_scores_df %>%
  separate(
    plate,
    into = c("col1", "Fungal_Strain", "Nitrogen_Level", "Replication", "col5"),
    sep = "_",
    remove = FALSE
  ) %>%
  select(-col1, -col5)
fpca_scores_df
outfile <- paste0("../results/traits_all_new_growth_raw_pc_scores.csv")
write.table(
  fpca_scores_df,file = outfile,row.names = TRUE,sep       = ","
)


# kw test
library(rstatix)
# kw_results <- fpca_scores_df %>%
#   group_by(trait) %>%
#   kruskal_test(fpca1 ~ Nitrogen_Level)
# 
results_ok[[1]]$scores$PC_df %>% 
  select(plate, Nitrogen_Level) %>% 
  head()


# Keep only traits with 2+ Nitrogen_Level groups
valid_traits <- fpca_scores_df %>%
  group_by(trait) %>%
  summarise(n_groups = n_distinct(Nitrogen_Level)) %>%
  filter(n_groups >= 3) %>%
  pull(trait)

failed_traits <- setdiff(unique(fpca_scores_df$trait), valid_traits)
message(length(failed_traits), " traits skipped (single group): ", 
        paste(failed_traits, collapse = ", "))

kruskal_results <- fpca_scores_df %>%
  filter(trait %in% valid_traits) %>%
  group_by(trait) %>%
  rstatix::kruskal_test(fpca1 ~ Nitrogen_Level) %>%
  ungroup()




sig_traits <- kw_results %>%
  filter(p < 0.05) %>%
  pull(trait)

pairwise_results <- fpca_scores_df %>%
  filter(trait %in% sig_traits) %>%
  group_by(trait) %>%
  dunn_test(fpca1 ~ Nitrogen_Level, p.adjust.method = "BH")
traits_all_sig <- pairwise_results %>%
  group_by(trait) %>%
  summarise(
    all_significant = all(p.adj < 0.05),
    .groups = "drop"
  ) %>%
  filter(all_significant) %>%
  pull(trait)

traits_all_sig
#[1] "LabB_mean_all_raw"          "LabB_mean_new_growth_raw"   "LabB_median_all_raw"       
#[4] "LabB_median_new_growth_raw"


kw_results2 <- fpca_scores_df %>%
  group_by(trait) %>%
  kruskal_test(fpca2 ~ Nitrogen_Level)

sig_traits2 <- kw_results2 %>%
  filter(p < 0.05) %>%
  pull(trait)

pairwise_results2 <- fpca_scores_df %>%
  filter(trait %in% sig_traits2) %>%
  group_by(trait) %>%
  dunn_test(fpca2 ~ Nitrogen_Level, p.adjust.method = "BH")
traits_all_sig2 <- pairwise_results2 %>%
  group_by(trait) %>%
  summarise(
    all_significant = all(p.adj < 0.05),
    .groups = "drop"
  ) %>%
  filter(all_significant) %>%
  pull(trait)

traits_all_sig2



traits_N10_N100_sig2 <- pairwise_results2 %>%
  filter(group1 == "N-10", p.adj < 0.05) %>%
  pull(trait) %>%
  unique()

traits_N10_N100_sig2


for (variable in traits_N10_N100_sig) {
  res <- run_fpca_trait(
    df = df_wide,
    trait = variable,
    K_est = 2
  )
  PC_df_1 <- res$scores$PC_df
  res_1$plots$fpca1_by_Nitrogen_Level_color_Fungal_Strain
}










trait = "LabB_median_all_raw"
res_1 <- run_fpca_trait(
  df = df_wide,
  trait = trait,
  K_est = 2
)
PC_df_1 <- res_1$scores$PC_df
res_1$plots$phi
res_1$plots$fpca1_by_Nitrogen_Level_color_Fungal_Strain
PC_long_1 <- PC_df_1 %>%
  pivot_longer(
    cols = c(fpca1, fpca2),
    names_to = "FPC",
    values_to = "score"
  ) %>%
  mutate(FPC = recode(FPC, fpca1 = "FPC1", fpca2 = "FPC2"))

ggplot(PC_long_1,
       aes(x = Nitrogen_Level, y = score)) +
  geom_jitter(width = 0.1, alpha = 0.7, size = 2) +
  stat_summary(fun = mean, geom = "point", size = 4) +
  stat_summary(fun.data = mean_cl_normal, geom = "errorbar", width = 0.2) +
  facet_wrap(~FPC, nrow = 1, scales = "free_y") +
  theme_bw() +
  labs(
    title = paste0("FPC1 and FPC2 across Nitrogen Levels of ",y_name),
    x = "Nitrogen Level",
    y = "FPC score"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    strip.text = element_text(face = "bold")
  )















######## calculate silhouse score ########

grp_all <- interaction(
  PC_df_area_labB$Nitrogen_Level,
  PC_df_area_labB$Fungal_Strain,
  drop = TRUE,
  sep = ":"
)

calc_sil <- function(df, cols, grp) {
  S <- df[, cols, drop = FALSE]
  
  keep <- complete.cases(S) & !is.na(grp)
  S2 <- scale(S[keep, , drop = FALSE])
  g2 <- droplevels(grp[keep])
  
  tab <- table(g2)
  ok_groups <- names(tab)[tab >= 2]
  keep_ok <- g2 %in% ok_groups
  
  S3 <- S2[keep_ok, , drop = FALSE]
  g3 <- as.integer(droplevels(g2[keep_ok]))
  
  sil <- silhouette(g3, dist(S3))
  list(
    mean_sil = mean(sil[, "sil_width"]),
    sil = sil,
    n_used = nrow(S3),
    n_groups = length(unique(g3))
  )
}

res1 <- calc_sil(PC_df_area_labB,
                 c("Area_pixels_fpca1","Area_pixels_fpca2"),
                 grp_all)

res2 <- calc_sil(PC_df_area_labB,
                 c("LabB_mean_fpca1","LabB_mean_fpca2"),
                 grp_all)

res3 <- calc_sil(PC_df_area_labB,
                 c("LabB_mean_fpca1","Area_pixels_fpca1"),
                 grp_all)

summary_df <- data.frame(
  S_definition = c("Area: (fpca1, fpca2)",
                   "LabB_mean: (fpca1, fpca2)",
                   "Mixed: (LabB_mean_fpca1, Area_pixels_fpca1)"),
  mean_silhouette = c(res1$mean_sil, res2$mean_sil, res3$mean_sil),
  n_used = c(res1$n_used, res2$n_used, res3$n_used),
  n_groups = c(res1$n_groups, res2$n_groups, res3$n_groups)
)

print(summary_df)





