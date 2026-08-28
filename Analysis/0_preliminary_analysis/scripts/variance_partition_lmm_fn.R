library(lme4)
library(dplyr)
library(tidyr)


calc_variance_lmm_long <- function(data_long,
                                   min_n = 3,
                                   min_levels = 2) {
  
  # data_long must already be filtered to ONE date/trait/mask/direction
  # and contain columns: value, Fungal_Strain, Nitrogen_Level
  
  sub <- data_long %>% filter(!is.na(value))
  
  if (nrow(sub) < min_n) {
    return(tibble(
      n_used = nrow(sub),
      ok = FALSE,
      reason = "Too few non-NA rows",
      Fungal_Strain = NA_real_,
      Nitrogen_Level = NA_real_,
      `Fungal_Strain:Nitrogen_Level` = NA_real_,
      Residual = NA_real_
    ))
  }
  
  if (length(unique(sub$value)) < min_levels) {
    return(tibble(
      n_used = nrow(sub),
      ok = FALSE,
      reason = "Too few unique values",
      Fungal_Strain = NA_real_,
      Nitrogen_Level = NA_real_,
      `Fungal_Strain:Nitrogen_Level` = NA_real_,
      Residual = NA_real_
    ))
  }
  
  fit <- tryCatch(
    lmer(
      value ~ (1|Fungal_Strain) + (1|Nitrogen_Level) + (1|Fungal_Strain:Nitrogen_Level),
      data = sub,
      REML = TRUE
    ),
    error = function(e) e
  )
  
  if (inherits(fit, "error")) {
    return(tibble(
      n_used = nrow(sub),
      ok = FALSE,
      reason = paste0("lmer error: ", fit$message),
      Fungal_Strain = NA_real_,
      Nitrogen_Level = NA_real_,
      `Fungal_Strain:Nitrogen_Level` = NA_real_,
      Residual = NA_real_
    ))
  }
  
  var_comp <- as.data.frame(VarCorr(fit)) %>%
    transmute(grp, vcov) %>%
    pivot_wider(names_from = grp, values_from = vcov)
  
  needed <- c("Fungal_Strain", "Nitrogen_Level", "Fungal_Strain:Nitrogen_Level", "Residual")
  for (nm in needed) if (!nm %in% names(var_comp)) var_comp[[nm]] <- NA_real_
  
  var_comp %>%
    mutate(n_used = nrow(sub), ok = TRUE, reason = NA_character_) %>%
    select(n_used, ok, reason, all_of(needed))
}