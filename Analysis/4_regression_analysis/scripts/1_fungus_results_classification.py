from pathlib import Path
import re
import sys

# directory of THIS script file
HERE = Path(__file__).resolve().parent

# add Code/ to python path (absolute)
CODE_DIR = (HERE / "../../../Code").resolve()
sys.path.insert(0, str(CODE_DIR))

from Run_ml_model import make_one_per_group_folds,cv_classify


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




# #################################################################
# ###################### Modeling #################################
# #################################################################
# def run_single_combination(view, mask_type, date, subset, feature_cols, test_plates_folds):
#     if subset.empty:
#         return None

#     print(f"=========Starting: {view}, {mask_type}, {date}======")

#     all_summaries = {}

#     for model_type in ["logistic", "lda", "lasso"]:
#         _, summary = cv_classify(
#             df=subset,
#             target_col = "Fungal_Strain",
#             feature_cols=feature_cols,
#             model_type=model_type,
#             outer_plate_folds =test_plates_folds
#         )
#         all_summaries[model_type] = summary

#     # summary table across models
#     summary_df = pd.DataFrame(all_summaries).T.drop(columns="classes")
    
#     return (view, mask_type, date), summary_df



# df_sub = df0.copy()
# grouped = {
#     key: g.copy()
#     for key, g in df_sub.groupby(["view", "mask_type", "date"])
# }

# tasks = []
# for (view, mask_type, date), subset in grouped.items():
#     tasks.append((view, mask_type, date, subset))

# # extract all unique dates from the dataframe, excluding the first date since some is missing
# all_dates = sorted(df_sub["date"].dropna().unique())
# dates_to_use = set(all_dates[1:])
# tasks = [t for t in tasks if t[2] in dates_to_use]



# # Execute the outer loop in parallel using 48 CPU cores
# # n_jobs=48 specifies the number of concurrent worker processes
# results_list = Parallel(n_jobs=48)(
#     delayed(run_single_combination)(v, m, d, subset, feature_cols=feature_cols, test_plates_folds=test_plates_folds) 
#    for v, m, d, subset in tqdm(tasks, desc="Parallel Training")
#     #for v, m, d in  tasks
# )

# # Organize and aggregate the results into a dictionary
# # Filter out None values in case some combinations were skipped
# combined_results = {res[0]: res[1] for res in results_list if res is not None}

# # Save combined_results to file
# RESULTS_DIR = (HERE / "../results/fungus_class").resolve()
# RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# OUTFILE = RESULTS_DIR / "results_fungal_simplemodel.pkl"

# with open(OUTFILE, "wb") as f:
#     pickle.dump(combined_results, f)

# print(f"\nSaved combined_results to {OUTFILE}")








#################################################################
###################### Modeling : color#################################
#################################################################
def run_single_combination_color(view, mask_type, date, subset, trait_set, test_plates_folds):
    if subset.empty:
        return None

    print(f"=========Starting: {view}, {mask_type}, {date}======")
    rows = []

    for trait_name, trait_cols in trait_set.items():
        for model_type in ["logistic", "lda", "lasso"]:
            _, summary = cv_classify(
                df=subset,
                feature_cols=trait_cols,
                model_type=model_type,
                outer_plate_folds=test_plates_folds
            )

            row = {"trait_set": trait_name, "model": model_type}
            row.update(summary)
            rows.append(row)

    summary_df = pd.DataFrame(rows)

    if "classes" in summary_df.columns:
        summary_df = summary_df.drop(columns="classes")

    return (view, mask_type, date), summary_df

# Create a list of all task combinations to iterate through
df_sub = df0.copy()
grouped = {
    key: g.copy()
    for key, g in df_sub.groupby(["view", "mask_type", "date"])
}

tasks = []
for (view, mask_type, date), subset in grouped.items():
    tasks.append((view, mask_type, date, subset))

# extract all unique dates from the dataframe, excluding the first date since some is missing
all_dates = sorted(df_sub["date"].dropna().unique())
dates_to_use = set(all_dates[1:])
tasks = [t for t in tasks if t[2] in dates_to_use]
cols = df0.columns
RGB_traits = [c for c in cols if re.match(r"^[RGB]_8bit_", c)]
HSV_traits = [c for c in cols if re.match(r"^[HSV]_", c)]
LAB_traits = [c for c in cols if re.match(r"^(L_|A_|LabB_)", c)]
Grey_traits = [c for c in cols if c.startswith("Gray_8bit_")]
Ncolor_traits = ['area_pixels', 'circularity',
       'contour_count', 'hull_area', 'max_width', 'min_width', 'perimeter',
       'rect_angle_deg', 'solidity']
trait_sets = {
    "RGB": RGB_traits,
    "HSV": HSV_traits,
    "LAB": LAB_traits,
    "Grey": Grey_traits,
    "Ncolor": Ncolor_traits
}

# Execute the outer loop in parallel using 48 CPU cores
# n_jobs=48 specifies the number of concurrent worker processes
results_list = Parallel(n_jobs=48)(
    delayed(run_single_combination_color)(v, m, d, subset, trait_set=trait_sets, test_plates_folds=test_plates_folds) 
    for v, m, d, subset in tqdm(tasks, desc="Parallel Training")
    #for v, m, d in  tasks
)

all_dfs = []

for res in results_list:
    if res is None:
        continue

    (view, mask_type, date), df_res = res
    df_tmp = df_res.copy()
    df_tmp["view"] = view
    df_tmp["mask_type"] = mask_type
    df_tmp["date"] = date
    all_dfs.append(df_tmp)

df_all = pd.concat(all_dfs, ignore_index=True)
RESULTS_DIR = (HERE / "../results/fungus_class").resolve()
df_all.to_csv(RESULTS_DIR /"results_fungal_color_simplemodel.csv", index=False)