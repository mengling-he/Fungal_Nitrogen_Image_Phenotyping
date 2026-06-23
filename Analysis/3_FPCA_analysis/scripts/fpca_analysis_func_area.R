library(fda)
library(magic)
library(dplyr)
library(tidyr)
library(rstudioapi)
setwd(dirname(getActiveDocumentContext()$path))

# load R functions
source("../FPCA_code/fpca_analysis_single.R")


outdir <- "../results/"

#### read data ####

# read the data preprocessed from .py (combined data, without imputaion)
df_wide <- read.csv("../data/metrics_wide.csv",stringsAsFactors = FALSE)

# select meta and area
df_clean <- df_wide %>%
  dplyr::select(
    plate,
    date,
    Fungal_Strain,
    Nitrogen_Level,
    Replication,
    starts_with("area_pixels"),
  )

result <- run_fpca_trait(df = df_clean, trait = 'area_pixels_all_raw', K_est = 2)

