from pathlib import Path
import sys
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse

# directory of THIS script file
HERE = Path(__file__).resolve().parent
# add Code/ to python path (absolute)
CODE_DIR = (HERE / "../../../Code").resolve()
sys.path.insert(0, str(CODE_DIR))
from Run_ml_model import make_one_per_group_folds
from Feature_importance import feature_cluster_map,nested_cv_with_cluster_pfi_classification




# ── CLI argument ──────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    choices=["ElasticNet_MultinomialLR", "Lasso_MultinomialLR", "ShrinkageLDA", "RandomForest"],
    default="ElasticNet_MultinomialLR",
    help="Which model to run"
)
args = parser.parse_args()
MODEL_NAME = args.model




#################################################################
###################### function #################################
#################################################################
def run_single_combination(view, mask_type, date, df, feature_cols, test_plates_folds,
                           pretrained_params_dict,model_name="ElasticNet_MultinomialLR", random_state=42):
    # 1. Filter the data
    subset = df[
        (df['view'] == view) & (df['mask_type'] == mask_type) & (df['date'] == date)
    ].drop(columns=['view', 'mask_type','date']).reset_index(drop=True)

    if subset.empty:
        return None
    
    pretrained_params_dict_filtered = pretrained_params_dict.get((view, mask_type, date), None)
    
    estimator_map = {
    "ElasticNet_MultinomialLR": LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        multi_class="multinomial",
        max_iter=20000,
        random_state=random_state
    ),
    "Lasso_MultinomialLR": LogisticRegression(
        penalty="l1",
        solver="saga",
        multi_class="multinomial",
        max_iter=20000,
        random_state=random_state
    ),
    "ShrinkageLDA": LinearDiscriminantAnalysis(),
    
    "RandomForest": RandomForestClassifier(
        random_state=random_state,
        class_weight="balanced"
    )
}
    print(f"========= Starting Classification: {view}, {mask_type}, {date} ======")
    
    # 2. Cluster Mapping
    X = subset[feature_cols]
    cluster_map_df = feature_cluster_map(X, threshold=0.85)

    # 3. Containers for results
    all_results = {
        # "importances": {},   # Store the PFI DataFrames
        # "fold_metrics": {},  # Store the train/test score DataFrames
        # "summaries": {}      # Store the mean/std summary dicts
    }
    
    # # 4. Loop through models (ElasticNet, RF, etc.)
    # for model_name, model_hypter_df in pretrained_params_dict_filtered.items():
    print(f"  -> Training {model_name}...")
    estimator_name = estimator_map[model_name]
    # ⭐ Note: Using accuracy_score for classification
    # Adjust n_jobs=1 if the outer loop is already parallelized
    fold_df, imp_df, summary_dict = nested_cv_with_cluster_pfi_classification(
        df=subset, 
        feature_cols=feature_cols, 
        target_col="Nitrogen_Level", 
        outer_plate_folds=test_plates_folds,  
        estimator=estimator_name,
        pretrained_params_df=pretrained_params_dict_filtered,
        feature_cluster=cluster_map_df,
        metric_fn=accuracy_score,    
        permutation_repeats=50,  
        random_state=random_state,
        n_jobs=1      
    )
        
    all_results["importances"] = imp_df
    all_results["fold_metrics"] = fold_df
    all_results["summaries"] = summary_dict

    # 5. Return everything as a structured dictionary
    return (view, mask_type, date), all_results





#################################################################
###################### Analysis #################################
#################################################################
#################################################################

###################### Data #################################
# build data path relative to this script
DATA_PATH = (HERE / "../data/metrics_masks5_plus_full.csv").resolve()
df0 = pd.read_csv(DATA_PATH)
df0 = df0.drop(columns=["image_path", "mask_path","scale_max_used"])

df0["Fungal_Strain"] = df0["plate"].str.split("_").str[1]
df0["Nitrogen_Level"] = df0["plate"].str.split("_").str[2]

df_binary = df0[df0["Nitrogen_Level"] != "N-1"]


# create CV fold for samples, to make them the sample across different conditions,
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


# get the hyperparameters for each model, for each fold, for each dataset combination
DATA_PATH_hypter = (HERE / "../data/results_nitrogen_update.pkl").resolve()
with open(DATA_PATH_hypter, "rb") as f:
    results_nitrogen_class = pickle.load(f)

hypter_dict = {k: v[0] for k, v in results_nitrogen_class.items()}



# get the data for analysis
df_sub = df0.copy()
trait_cols = [
    c for c in df_sub.columns
    if c not in ['plate', 'date',  'view','mask_type', 'scale_max_used','Fungal_Strain',
       'Nitrogen_Level', 'respiration']
]
# Create a list of all task combinations to iterate through
tasks = []
#mask_type_study = ["all","new_growth"]
for view in df_sub['view'].unique():
    for mask_type in df_sub['mask_type'].unique():
        for date in df_sub['date'].unique()[1:]:
            tasks.append((view, mask_type, date))

hypter_dict_model = {k: v.get(MODEL_NAME, None) for k, v in hypter_dict.items()}





results_list = Parallel(n_jobs=48)(
    delayed(run_single_combination)(v, m, d, df=df_sub,
                                    feature_cols=trait_cols, test_plates_folds=test_plates_folds,
                                    pretrained_params_dict= hypter_dict_model,model_name=MODEL_NAME, random_state=42) 
    for v, m, d in tqdm(tasks, desc="Parallel Training")
    #for v, m, d in  tasks
)

# Organize and aggregate the results into a dictionary
# Filter out None values in case some combinations were skipped
combined_results = {res[0]: res[1] for res in results_list if res is not None}

# ========= SAVE RESULTS =========
RESULTS_DIR = (HERE / f"../results/nitrogen_class/{MODEL_NAME}").resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

imp_frames, fold_frames, summary_rows = [], [], []

for (view, mask_type, date), res in combined_results.items():
    id_cols = {"view": view, "mask_type": mask_type, "date": date}

    imp = res["importances"].copy()
    imp[list(id_cols)] = list(id_cols.values())
    imp_frames.append(imp)

    fold = res["fold_metrics"].copy()
    fold[list(id_cols)] = list(id_cols.values())
    fold_frames.append(fold)

    summary_rows.append({**id_cols, **res["summaries"]})

#parquet is smaller and faster to read/write than csv, and it can store data types better, so we will save the importances and fold metrics as parquet, and the summaries as csv since it's just a small dataframe
pd.concat(imp_frames,  ignore_index=True).to_parquet(RESULTS_DIR / f"nitrogen_importance_{MODEL_NAME}_importances.parquet",  index=False)
pd.concat(fold_frames, ignore_index=True).to_parquet(RESULTS_DIR / f"nitrogen_importance_{MODEL_NAME}_fold_metrics.parquet", index=False)
pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / f"nitrogen_importance_{MODEL_NAME}_summaries.csv", index=False)

print(f"\nSaved results to {RESULTS_DIR}")
print(f"  - nitrogen_importance_{MODEL_NAME}_importances.parquet  ({len(imp_frames)} combinations)")
print(f"  - nitrogen_importance_{MODEL_NAME}_fold_metrics.parquet")
print(f"  - nitrogen_importance_{MODEL_NAME}_summaries.csv")

