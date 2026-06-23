# Image-based microbial phenotyping and CO2 flux offer new insights into fungal nitrogen use efficiency

Code repository for:

> Song Y, He M, Orebaugh J, Seethepalli A, Rush TA, Emrich S, and York LM. *Image-based microbial phenotyping and CO2 flux offer new insights into fungal nitrogen use efficiency.*

## Overview

This repository contains the analysis code for quantifying dynamic fungal responses to nitrate availability using image-based phenotyping and respiration measurements. Two fungal isolates (*Fusarium graminearum* species complex strain PMI3048 and *Linnemannia elongata* strain AG77) were grown on glucose minimal medium under three nitrate levels (0.01X, 0.1X, 1X) and imaged daily for 12 days.

**Note:** The code for image alignment and U-Net segmentation training is available separately.

## Repository Structure

```
.
├── Data/                              # Primary datasets
│   ├── metrics_masks5_plus_full.csv   # Image-derived traits (area, color, shape)
│   ├── MV_final_cleaned_combined.xlsx # Cleaned combined dataset
│   ├── barcodes--*.csv                # Sample barcode mapping
│   └── description--*.csv            # Experiment description
│
├── Code/                              # Shared Python modules
│   ├── Run_ml_model.py               # ML model training framework
│   ├── helper_plot.py                # Plotting utilities
│   └── Feature_importance.py         # Feature importance extraction
│
├── Analysis/
│   ├── 0_preliminary_analysis/        # Data preprocessing & exploratory analysis
│   │   └── scripts/
│   │       ├── preliminary_analysis.ipynb  # Trait extraction from masks (OpenCV)
│   │       ├── PCA_strain.ipynb            # PCA: strain separation (Fig. 4, S3, S5)
│   │       ├── PCA_nitro.ipynb             # PCA: nitrogen separation (Fig. S5)
│   │       ├── respiration_analysis.ipynb  # Respiration visualization (Fig. 8)
│   │       └── plot_timeseries_area.R      # Colony area time-series plots (Fig. 2)
│   │
│   ├── 1_anova_analysis/              # Repeated-measures ANOVA
│   │   └── scripts/
│   │       ├── ANOVA_area.R           # RM-ANOVA on colony area (Fig. 2, Table S1)
│   │       ├── ANOVA_respiration.R    # Two-way ANOVA on respiration (Fig. 8)
│   │       └── rm_anova_batch_fn.R    # Helper: batch RM-ANOVA function
│   │
│   ├── 3_FPCA_analysis/               # Functional PCA
│   │   ├── FPCA_code/                 # Core FPCA functions
│   │   │   ├── fpca_analysis_single.R # Single-trait FPCA pipeline
│   │   │   ├── pca_fun.R             # PCA helper functions
│   │   │   ├── pca_score.R           # PCA score computation
│   │   │   └── tuning_nointer.R      # FPCA tuning (no interaction)
│   │   └── scripts/
│   │       ├── 1_FPCA_fungal.R        # FPCA on area: strain effects (Fig. 3, Table S2)
│   │       ├── 1_FPCA_nitrogen.R      # FPCA on area: nitrogen effects (Fig. S2)
│   │       ├── fpca_analysis_area.R   # FPCA area analysis pipeline
│   │       ├── fpca_analysis_area_result.R  # FPCA area results processing
│   │       ├── fpca_analysis_LabBmean.R     # FPCA on Lab B trajectories (Fig. S7)
│   │       └── run_fpca_function.R    # FPCA runner
│   │
│   ├── 4_regression_analysis/         # ML classification & regression
│   │   └── scripts/
│   │       ├── 1_fungus_results_nestedCV_classification.py       # Strain classification (Fig. 5)
│   │       ├── 1_fungus_results_nestedCV_classification_color.py # Strain: color-only (Fig. S4)
│   │       ├── 1_nitrogen_classification.py       # Nitrogen classification (Fig. 6)
│   │       ├── 1_nitrogen_classification_color.py # Nitrogen: color-only
│   │       ├── 1_respiration_regression.py        # Respiration regression (Fig. S8, S9)
│   │       ├── 2_classification_analysis_fungus.ipynb    # Strain results analysis
│   │       ├── 2_classification_analysis_nitrogen.ipynb  # Nitrogen results analysis
│   │       ├── 2_FPCA_area_LabBclassification.ipynb     # FPCA scores as features
│   │       ├── 2_respiration_PLS.ipynb            # PLS respiration model (Fig. S8)
│   │       ├── 2_respiration_results_analysis.ipynb      # Respiration results
│   │       └── respiration_analysis_fn.py         # Respiration helper functions
│   │
│   └── 5_feature_importance/          # Feature importance
│       └── scripts/
│           ├── 1_nitrogen_feature_importance.py           # Elastic Net coefficients (Fig. 7)
│           ├── 2_nitrogen_feature_importance_analysis.ipynb # Feature importance plots (Fig. S6)
│           └── 3_univariate_FPCA_analysis.ipynb           # FPCA + univariate analysis (Fig. S7)
│
└── requirements.txt                   # Python dependencies
```

## Analysis Pipeline

The analyses are organized to follow the paper's workflow:

| Step | Folder | Description | Paper Reference |
|------|--------|-------------|-----------------|
| 1 | `0_preliminary_analysis` | Trait extraction from segmented colony masks using OpenCV; PCA of image traits | Methods; Fig. 4, S3, S5 |
| 2 | `1_anova_analysis` | Repeated-measures ANOVA on colony area over time; two-way ANOVA on respiration | Fig. 2, 8; Table S1 |
| 3 | `3_FPCA_analysis` | Functional PCA of area and Lab B growth trajectories | Fig. 3, S2, S7; Table S2 |
| 4 | `4_regression_analysis` | ML classification (strain & nitrogen) and respiration regression using nested CV | Fig. 5, 6, S4, S8, S9 |
| 5 | `5_feature_importance` | Feature importance from Elastic Net models | Fig. 7, S6 |

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
