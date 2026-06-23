from pathlib import Path
import sys

# directory of THIS script file
HERE = Path(__file__).resolve().parent

# add Code/ to python path (absolute)
CODE_DIR = (HERE / "../../../Code").resolve()
sys.path.insert(0, str(CODE_DIR))
from Run_ml_model import run_models

from Run_ml_model import make_one_per_group_folds, run_models

import numpy as np
import pandas as pd
import pickle
from joblib import Parallel, delayed
from tqdm import tqdm # for progress bars
import warnings
warnings.filterwarnings("default")


# build data path relative to this script
DATA_PATH = (HERE / "../data/metrics_masks5_plus_full.csv").resolve()

df0 = pd.read_csv(DATA_PATH)
print("Loaded:", DATA_PATH)


df0 = df0.drop(columns=["image_path", "mask_path","scale_max_used"])
feature_cols = df0.columns[4:]


df0["Fungal_Strain"] = df0["plate"].str.split("_").str[1]
df0["Nitrogen_Level"] = df0["plate"].str.split("_").str[2]

df0 = df0[df0["Nitrogen_Level"] != "N-1"]


views = df0['view'].unique()
mask_types = df0['mask_type'].unique()
dates = df0['date'].unique()[1:] # exclude the first date since some is missing





plates = df0["plate"].unique()
df_plate = (
    df0[["plate", "Fungal_Strain", "Nitrogen_Level"]]
    .drop_duplicates()
    .reset_index(drop=True)
)
outplate_class = make_one_per_group_folds(df_plate, group_cols=list(("Fungal_Strain", "Nitrogen_Level")),n_splits=5, random_state=42)
test_plates_folds = []
for _, test_plate_idx in outplate_class:
    test_plates  = df_plate.iloc[test_plate_idx]["plate"].values
    test_plates_folds.append(test_plates)
print("Created test_plates_folds for CV based on plates.")
print(test_plates_folds)










#################################################################
###################### Modeling #################################
#################################################################
def run_single_combination(view, mask_type, date, df_combined, feature_cols, test_plates_folds):
    subset = df_combined[(df_combined['view'] == view) & 
                         (df_combined['mask_type'] == mask_type) & 
                         (df_combined['date'] == date)]
    if subset.empty:
        return None

    print(f"=========Starting: {view}, {mask_type}, {date}======")
    results = run_models(
        subset, feature_cols,
        target_col="Nitrogen_Level",
        group_cols=("Fungal_Strain", "Nitrogen_Level"),
        outer_splits=5, 
        outer_plate_folds=test_plates_folds,
        inner_splits=4,
        random_state=42,n_jobs=1)  # Note: Set to 1 here as parallelization is handled at the outer loop level.
    return (view, mask_type, date), results








# Create a list of all task combinations to iterate through
df_sub = df0.copy()
views = df0['view'].unique()
mask_types = df0['mask_type'].unique()
dates = df0['date'].unique()[1:] # exclude the first date since some is missing

tasks = []
for view in views:
    for mask_type in mask_types:
        for date in dates:
            tasks.append((view, mask_type, date))

# Execute the outer loop in parallel using 48 CPU cores
# n_jobs=48 specifies the number of concurrent worker processes
results_list = Parallel(n_jobs=48)(
    delayed(run_single_combination)(v, m, d, df_combined=df_sub, feature_cols=feature_cols, test_plates_folds=test_plates_folds) 
    for v, m, d in tqdm(tasks, desc="Parallel Training")
    #for v, m, d in  tasks
)

# Organize and aggregate the results into a dictionary
# Filter out None values in case some combinations were skipped
combined_results = {res[0]: res[1] for res in results_list if res is not None}

# Save combined_results to file
RESULTS_DIR = (HERE / "../results/nitrogen_class").resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTFILE = RESULTS_DIR / "results_nitrogen_binary_update.pkl"

with open(OUTFILE, "wb") as f:
    pickle.dump(combined_results, f)

print(f"\nSaved combined_results to {OUTFILE}")
