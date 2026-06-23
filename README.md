# Image-based microbial phenotyping and CO2 flux offer new insights into fungal nitrogen use efficiency

Code repository for:

> Song Y, He M, Orebaugh J, Seethepalli A, Rush TA, Emrich S, and York LM. *Image-based microbial phenotyping and CO2 flux offer new insights into fungal nitrogen use efficiency.*

## Overview

This repository contains the analysis code for quantifying dynamic fungal responses to nitrate availability using image-based phenotyping and respiration measurements. Two fungal isolates (*Fusarium graminearum* species complex strain PMI3048 and *Linnemannia elongata* strain AG77) were grown on glucose minimal medium under three nitrate levels and imaged daily for 12 days.

**Note:** The code for image alignment and U-Net segmentation training is available separately.

## Repository Structure

```
.
├── Data/                              # Primary datasets
│
├── Code/                              # Shared Python modules
│   ├── Run_ml_model.py               # ML model training framework
│   ├── helper_plot.py                # Plotting utilities
│   └── Feature_importance.py         # Feature importance extraction
│
├── Analysis/
│   ├── 0_preliminary_analysis/        # Data preprocessing & exploratory analysis
│   ├── 1_anova_analysis/              # Repeated-measures ANOVA
│   │
│   ├── 3_FPCA_analysis/               # Functional PCA
│   │
│   ├── 4_regression_analysis/         # ML classification & regression
│  
│   └── 5_feature_importance/          # Feature importance

└── requirements.txt                   # Python dependencies
```

## Analysis Pipeline

The analyses are organized to follow the paper's workflow:

| Step | Folder | Description | 
|------|--------|-------------|
| 1 | `0_preliminary_analysis` | preliminary analysis of the data; PCA of image traits | 
| 2 | `1_anova_analysis` | Repeated-measures ANOVA on colony area over time; two-way ANOVA on respiration | 
| 3 | `3_FPCA_analysis` | Functional PCA of area and Lab B growth trajectories |
| 4 | `4_regression_analysis` | ML classification (strain & nitrogen) using nested CV |
| 5 | `5_feature_importance` | Feature importance from Elastic Net models |

## Software Requirements

- **Python** 3.10.9 (PCA, ML classification/regression)
- **R** 4.3.1 (ANOVA, FPCA)

### Python packages
Install with: `pip install -r requirements.txt`

Key packages: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `openpyxl`

### R packages
- `afex` - repeated-measures ANOVA
- `emmeans` - post-hoc pairwise comparisons
- `fda` - functional data analysis (FPCA)

### Computational notes
- HPC scripts (files starting with `1_`) were run on high-performance computing clusters
- Notebooks (files starting with `2_`) process the HPC results and generate figures
- Some scripts use `joblib` for parallel execution

## Data Description

| File | Description |
|------|-------------|
| `metrics_masks5_plus_full.csv` | Image-derived traits extracted from colony masks: area, width, circularity, and pixel statistics (mean, median, min, max, variance) in RGB, HSV, and LAB color spaces for full and new-growth masks under transmission and overhead illumination |
| `MV_final_cleaned_combined.xlsx` | Cleaned combined dataset with sample metadata and respiration measurements |
| `barcodes--*.csv` | Barcode-to-sample mapping for the imaging experiment |
| `description--*.csv` | Experimental design description |
