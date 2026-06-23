from pathlib import Path
import sys
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.cluster import hierarchy

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.inspection import permutation_importance

# directory of THIS script file
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from Run_ml_model import make_sorted_target_folds,make_one_per_group_folds


def feature_cluster_map(X, threshold=0.85):

    # 1️⃣ Identify constant features
    constant_mask = X.nunique() <= 1
    constant_features = X.columns[constant_mask]
    variable_features = X.columns[~constant_mask]

    # Initialize cluster map
    cluster_map = pd.Series(index=X.columns, dtype=int)

    # Assign constant features to cluster 0
    cluster_map.loc[constant_features] = 0

    if len(variable_features) == 0:
        print("⚠️ All features are constant.")
        return cluster_map

    # 2️⃣ Correlation on variable features only
    X_var = X[variable_features]
    corr = X_var.corr(method='spearman').to_numpy()
    corr = np.nan_to_num(corr, nan=0)  # safety
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)

    # Distance metric
    dist_matrix = 1 - np.abs(corr)

    # Ward requires condensed distance matrix
    dist_condensed = hierarchy.distance.squareform(dist_matrix)
    dist_linkage = hierarchy.ward(dist_condensed)

    # 3️⃣ Clustering
    dist_threshold = 1 - threshold
    cluster_ids = hierarchy.fcluster(
        dist_linkage,
        t=dist_threshold,
        criterion='distance'
    )

    # Assign cluster IDs (start from 1)
    cluster_map.loc[variable_features] = cluster_ids

    n_clusters = len(np.unique(cluster_ids))
    print(f"✅ Clustering Complete:")
    print(f"   Variable clusters: {n_clusters}")
    print(f"   Constant features: {len(constant_features)} (Cluster 0)")
    print(f"   Threshold: |r| >= {threshold}")

    return cluster_map



def nested_cv_with_cluster_pfi_classification(
    df,
    feature_cols,
    target_col,
    outer_plate_folds,
    estimator,
    pretrained_params_df,
    feature_cluster,
    metric_fn=accuracy_score,  # ⭐ Default changed to accuracy
    permutation_repeats=50,    # Reduced slightly for speed, adjust as needed
    random_state=42,
    n_jobs=1,
):
    # use the pretrained_params_df and same outer_plate_folds to get the best hyperparameters to get feature importance without doing GridSearchCV again, this is to save time and also to make sure the same hyperparameters are used for feature importance as in the main model evaluation
    
    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()
    
    # ========= OUTER FOLDS =========
    outer_folds = []
    for test_plates in outer_plate_folds:
        test_mask = df["plate"].isin(test_plates)
        train_mask = ~test_mask
        outer_folds.append((np.where(train_mask)[0], np.where(test_mask)[0]))

    # ========= PRETRAINED PARAMS VALIDATION =========
    assert pretrained_params_df["fold"].nunique() == len(outer_folds), (
        f"pretrained_params_df has {pretrained_params_df['fold'].nunique()} unique folds "
        f"but outer_folds has {len(outer_folds)}. "
        "Ensure the same outer_plate_folds and df are used in both calls."
    )

    def _cast(v):
        """Cast whole-number floats to int (e.g. n_estimators stored as 300.0 -> 300)."""
        if isinstance(v, float) and v.is_integer():
            return int(v)
        return v

    fold_rows = []
    all_fold_importances_train = []
    all_fold_importances_test = []

    for fold_i, (train_idx, test_idx) in enumerate(outer_folds, start=1):
        X_train_full, X_test = X[train_idx], X[test_idx]
        y_train_full, y_test = y[train_idx], y[test_idx]
    
        scaler = StandardScaler()
        X_train_full = scaler.fit_transform(X_train_full)
        X_test = scaler.transform(X_test)

        # ========= FIT MODEL (pretrained params) =========
        # Look up the pre-computed best params for this fold
        fold_row = pretrained_params_df[pretrained_params_df["fold"] == fold_i].iloc[0] #selects the row in pretrained_params_df corresponding to the current fold (fold_i) and returns it as a Series.
        #collect all columns that store best hyperparameters, remove the "best_" prefix, convert their values to the correct type, and store them in a dictionary to initialize the model
        best_params = {
            k[len("best_"):].replace("clf__", ""): _cast(fold_row[k])
            for k in fold_row.index
            if k.startswith("best_")
            }
        model_fold = clone(estimator)
        model_fold.set_params(**best_params)
        model_fold.fit(X_train_full, y_train_full)
        
        
        # ⭐ REPORTING TRAINING AND TESTING METRICS
        y_train_pred = model_fold.predict(X_train_full)
        y_test_pred = model_fold.predict(X_test)

        # Calculate Accuracy
        train_acc = accuracy_score(y_train_full, y_train_pred)
        test_acc  = accuracy_score(y_test, y_test_pred)

        # Calculate F1 (Macro average is good for balanced view of 3 classes)
        train_f1 = f1_score(y_train_full, y_train_pred, average='macro')
        test_f1  = f1_score(y_test, y_test_pred, average='macro')

        baseline_score_train = metric_fn(y_train_full, y_train_pred)
        baseline_score_test = metric_fn(y_test, y_test_pred)

        fold_rows.append({
            "fold": fold_i,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "train_f1_macro": train_f1,
            "test_f1_macro": test_f1,
            "baseline_metric_train": baseline_score_train, # Typically matches train_acc
            "baseline_metric_test": baseline_score_test, # Typically matches test_acc
            })

        # ========= CLUSTER PERMUTATION IMPORTANCE (CVPFI-C) =========
        current_fold_imp_train = {}
        current_fold_imp_test = {}
        unique_clusters = np.sort(np.unique(feature_cluster.values))
        
        for c_id in unique_clusters:
            repeat_scores_train = []
            repeat_scores_test = []
            feats_in_cluster_mask = (feature_cluster == c_id).values
            feat_indices = np.where(feats_in_cluster_mask)[0]
            
            for r in range(permutation_repeats):
                X_train_permuted = X_train_full.copy()
                X_test_permuted = X_test.copy()
                for x in [X_train_permuted, X_test_permuted]:
                    # Permute rows of the cluster's features together
                    X_permuted = x.copy()
                    rng = np.random.default_rng(random_state + fold_i + int(c_id) + r)
                    perm_idx = rng.permutation(x.shape[0])
                    # for each feature in the cluster, permute based on perm_idx the same way to keep them together
                    X_permuted[:, feat_indices] = X_permuted[perm_idx][:, feat_indices]

                    if x is X_train_permuted:
                        X_train_permuted = X_permuted
                    else:
                        X_test_permuted = X_permuted

                # Importance is the DROP in accuracy/F1
                perm_score_train = metric_fn(y_train_full, model_fold.predict(X_train_permuted))
                perm_score_test = metric_fn(y_test, model_fold.predict(X_test_permuted))
                repeat_scores_train.append(baseline_score_train - perm_score_train)
                repeat_scores_test.append(baseline_score_test - perm_score_test)

            current_fold_imp_train[c_id] = np.mean(repeat_scores_train)
            current_fold_imp_test[c_id] = np.mean(repeat_scores_test)
            
        all_fold_importances_train.append(current_fold_imp_train)
        all_fold_importances_test.append(current_fold_imp_test)

    # ========= AGGREGATE RESULTS =========
    imp_per_fold_df_train = pd.DataFrame(all_fold_importances_train)  # Dictionary keys is the column name (cluster id), rows are folds
    imp_per_fold_df_test = pd.DataFrame(all_fold_importances_test)
    cluster_stats = pd.DataFrame({
        "cluster": imp_per_fold_df_train.columns,
        "mean_importance_train": imp_per_fold_df_train.mean(),
        "std_importance_train": imp_per_fold_df_train.std(),
        "mean_importance_test": imp_per_fold_df_test.mean(),
        "std_importance_test": imp_per_fold_df_test.std()
    }).reset_index(drop=True)

    importance_df = (# series to dataframe with feature names and cluster ids, then merge with cluster_stats to get mean/std importance for each feature based on its cluster
        feature_cluster
        .to_frame(name='cluster')
        .reset_index()
        .rename(columns={'index': 'feature_name'})
        )
    importance_df = importance_df.merge(cluster_stats, on='cluster', how='left')
    importance_df = importance_df.sort_values("mean_importance_test", ascending=False)

    results_df = pd.DataFrame(fold_rows)
    summary = {
        "mean_train_acc":    results_df["train_acc"].mean(),
        "std_train_acc":     results_df["train_acc"].std(),
        "mean_test_acc":     results_df["test_acc"].mean(),
        "std_test_acc":      results_df["test_acc"].std(),
        "mean_train_f1":     results_df["train_f1_macro"].mean(),
        "mean_test_f1":      results_df["test_f1_macro"].mean(),
        "std_test_f1":       results_df["test_f1_macro"].std(),
    }

    return results_df, importance_df, summary







def nested_cv_coef_importance_classification(
    df,
    feature_cols,
    target_col,
    outer_plate_folds,
    estimator,
    pretrained_params_df,
):
    """
    Fit a coefficient-based classification model (e.g. lasso / elasticnet multinomial logistic regression)
    on predefined outer CV folds using pretrained hyperparameters, and summarize coefficient importance.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe. Must contain feature columns, target column, and 'plate'.
    feature_cols : list of str
        Feature column names.
    target_col : str
        Target column name.
    outer_plate_folds : list
        Each element is a collection of test plates for one outer fold.
    estimator : sklearn estimator
        Usually a LogisticRegression estimator or compatible classifier with coef_.
    pretrained_params_df : pd.DataFrame
        DataFrame containing one row per fold and columns such as:
        fold, best_C, best_l1_ratio, ...
        or pipeline-style names such as best_clf__C.

    Returns
    -------
    class_coef_df : pd.DataFrame
        Fold-level per-class coefficients.
    importance_df : pd.DataFrame
        Aggregated mean/std absolute coefficient importance across folds.
    summary : dict
        Summary of train/test metrics.
    """
    required_cols = set(feature_cols + [target_col, "plate"])
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"df is missing required columns: {sorted(missing_cols)}")

    if "fold" not in pretrained_params_df.columns:
        raise ValueError("pretrained_params_df must contain a 'fold' column.")

    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()

    # ========= OUTER FOLDS =========
    outer_folds = []
    for test_plates in outer_plate_folds:
        test_mask = df["plate"].isin(test_plates)
        train_mask = ~test_mask
        outer_folds.append((np.where(train_mask)[0], np.where(test_mask)[0]))

    # ========= PRETRAINED PARAMS VALIDATION =========
    n_unique_folds = pretrained_params_df["fold"].nunique()
    if n_unique_folds != len(outer_folds):
        raise ValueError(
            f"pretrained_params_df has {n_unique_folds} unique folds but "
            f"outer_folds has {len(outer_folds)}. Ensure the same outer folds "
            f"and dataframe were used."
        )

    def _cast(v):
        """Robust casting for hyperparameters loaded from DataFrame."""
        if pd.isna(v):
            return None
        if isinstance(v, (np.integer, int)):
            return int(v)
        if isinstance(v, (np.floating, float)):
            return int(v) if float(v).is_integer() else float(v)
        return v

    def _normalize_param_name(param_name, valid_params):
        """
        Keep param if already valid; otherwise try removing 'clf__' prefix.
        """
        if param_name in valid_params:
            return param_name
        if param_name.startswith("clf__"):
            stripped = param_name.replace("clf__", "", 1)
            if stripped in valid_params:
                return stripped
        return None

    fold_rows = []
    all_fold_importances = []
    class_coef_rows = []

    for fold_i, (train_idx, test_idx) in enumerate(outer_folds, start=1):
        X_train_full, X_test = X[train_idx], X[test_idx]
        y_train_full, y_test = y[train_idx], y[test_idx]

        # Standardize within training fold only
        scaler = StandardScaler()
        X_train_full = scaler.fit_transform(X_train_full)
        X_test = scaler.transform(X_test)

        # ========= LOAD PRETRAINED PARAMS =========
        fold_match = pretrained_params_df[pretrained_params_df["fold"] == fold_i]
        if len(fold_match) != 1:
            raise ValueError(
                f"Expected exactly 1 row in pretrained_params_df for fold={fold_i}, "
                f"found {len(fold_match)}."
            )
        fold_row = fold_match.iloc[0]

        model_fold = clone(estimator)
        valid_params = set(model_fold.get_params().keys())

        best_params = {}
        for k in fold_row.index:
            if not k.startswith("best_"):
                continue
            raw_name = k[len("best_"):]
            norm_name = _normalize_param_name(raw_name, valid_params)
            if norm_name is not None:
                best_params[norm_name] = _cast(fold_row[k])

        model_fold.set_params(**best_params)
        model_fold.fit(X_train_full, y_train_full)

        if not hasattr(model_fold, "coef_"):
            raise ValueError(
                f"Estimator of type {type(model_fold).__name__} does not have coef_. "
                "This function is intended for linear classifiers such as logistic regression."
            )

        # ========= REPORT TRAIN/TEST METRICS =========
        y_train_pred = model_fold.predict(X_train_full)
        y_test_pred = model_fold.predict(X_test)

        train_acc = accuracy_score(y_train_full, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        train_f1 = f1_score(y_train_full, y_train_pred, average="macro")
        test_f1 = f1_score(y_test, y_test_pred, average="macro")

        fold_rows.append({
            "fold": fold_i,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "train_f1_macro": train_f1,
            "test_f1_macro": test_f1,
        })

        # ========= COEFFICIENT-BASED IMPORTANCE =========
        coef_mat = np.atleast_2d(model_fold.coef_)  # shape: (n_classes_or_1, n_features)
        coef_importance = np.mean(np.abs(coef_mat), axis=0)

        fold_imp_row = {"fold": fold_i}
        for feat, imp in zip(feature_cols, coef_importance):
            fold_imp_row[feat] = imp
        all_fold_importances.append(fold_imp_row)

        classes = getattr(model_fold, "classes_", np.arange(coef_mat.shape[0]))
        for class_i, class_label in enumerate(classes[:coef_mat.shape[0]]):
            row = {"fold": fold_i, "class_label": class_label}
            for feat_i, feat in enumerate(feature_cols):
                row[feat] = coef_mat[class_i, feat_i]
            class_coef_rows.append(row)

    # ========= AGGREGATE RESULTS =========
    imp_per_fold_df = pd.DataFrame(all_fold_importances)

    importance_df = pd.DataFrame({
        "feature_name": feature_cols,
        "mean_importance": [imp_per_fold_df[f].mean() for f in feature_cols],
        "std_importance": [imp_per_fold_df[f].std() for f in feature_cols],
    }).sort_values("mean_importance", ascending=False).reset_index(drop=True)

    class_coef_df = pd.DataFrame(class_coef_rows)
    results_df = pd.DataFrame(fold_rows)

    summary = {
        "mean_train_acc": results_df["train_acc"].mean(),
        "std_train_acc": results_df["train_acc"].std(),
        "mean_test_acc": results_df["test_acc"].mean(),
        "std_test_acc": results_df["test_acc"].std(),
        "mean_train_f1": results_df["train_f1_macro"].mean(),
        "std_train_f1": results_df["train_f1_macro"].std(),
        "mean_test_f1": results_df["test_f1_macro"].mean(),
        "std_test_f1": results_df["test_f1_macro"].std(),
    }

    return class_coef_df, importance_df, summary




def nested_cv_with_cluster_pfi(
    df,
    feature_cols,
    target_col,
    estimator,
    param_grid,
    feature_cluster,
    metric_fn=r2_score,      # ⭐ Added: Pass any scoring function (r2_score, accuracy_score, etc.)
    outer_plate_folds=None,   
    inner_splits=4,
    permutation_repeats=50,
    random_state=42,
    scoring="r2",             # This is for GridSearchCV internal tuning
    n_jobs=1 
):
    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy().astype(float)
    
    # ========= OUTER FOLDS =========
    if outer_plate_folds is None:
        # Assuming make_sorted_target_folds is pre-defined
        outer_folds = make_sorted_target_folds(
            df=df,
            target_col=target_col,
            n_splits=inner_splits + 1,
            random_state=random_state
        )
    else:
        outer_folds = []
        for test_plates in outer_plate_folds:
            test_mask = df["plate"].isin(test_plates)
            train_mask = ~test_mask
            outer_folds.append((np.where(train_mask)[0], np.where(test_mask)[0]))

    fold_rows = []
    all_fold_importances = [] 

    for fold_i, (train_idx, test_idx) in enumerate(outer_folds, start=1):
        X_train_full, X_test = X[train_idx], X[test_idx]
        y_train_full, y_test = y[train_idx], y[test_idx]
    
        # Scale data
        scaler = StandardScaler()
        X_train_full = scaler.fit_transform(X_train_full)
        X_test = scaler.transform(X_test)

        # Inner folds for hyperparameter tuning
        df_train = df.iloc[train_idx].reset_index(drop=True)
        inner_folds = make_sorted_target_folds(
            df=df_train,
            target_col=target_col,
            n_splits=inner_splits,
            random_state=random_state + fold_i
        )

        grid = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
            cv=inner_folds,
            refit=True,
            n_jobs=n_jobs
        )
        grid.fit(X_train_full, y_train_full)

        model_fold = grid.best_estimator_
        
        # Baseline performance on test set using metric_fn
        y_test_pred = model_fold.predict(X_test)
        baseline_score = metric_fn(y_test, y_test_pred)

        # General metrics for regression (optional logging)
        rmse_fold = np.sqrt(mean_squared_error(y_test, y_test_pred))
        r2_fold = r2_score(y_test, y_test_pred)

        fold_rows.append({
            "fold": fold_i,
            "rmse": rmse_fold,
            "r2": r2_fold,
            "baseline_metric": baseline_score,
            "best_params": grid.best_params_
        })

        # ========= CLUSTER PERMUTATION IMPORTANCE (CVPFI-C) =========
        current_fold_imp = {}
        unique_clusters = np.sort(np.unique(feature_cluster.values))
        
        for c_id in unique_clusters:
            repeat_scores = [] # ⭐ Reset for each cluster
            
            # Identify indices of features in the current cluster
            feats_in_cluster_mask = (feature_cluster == c_id).values
            feat_indices = np.where(feats_in_cluster_mask)[0]
            
            for r in range(permutation_repeats):
                X_test_permuted = X_test.copy()
                
                # Permute rows of the cluster's features
                rng = np.random.default_rng(random_state + fold_i + int(c_id) + r)
                perm_idx = rng.permutation(X_test_permuted.shape[0])
                X_test_permuted[:, feat_indices] = X_test_permuted[perm_idx][:, feat_indices]
                
                # Score after permutation
                perm_score = metric_fn(y_test, model_fold.predict(X_test_permuted))
                
                # Importance is the drop in performance
                repeat_scores.append(baseline_score - perm_score)

            # Average importance for this cluster in the current fold
            current_fold_imp[c_id] = np.mean(repeat_scores)
            
        all_fold_importances.append(current_fold_imp)

    # ========= AGGREGATE RESULTS =========
    results_df = pd.DataFrame(fold_rows)
    
    # Calculate Mean/Std per Cluster across folds
    imp_per_fold_df = pd.DataFrame(all_fold_importances) 
    cluster_stats = pd.DataFrame({
        "cluster": imp_per_fold_df.columns,
        "mean_importance": imp_per_fold_df.mean(),
        "std_importance": imp_per_fold_df.std()
    }).reset_index(drop=True)

    # Map back to Feature Names for detailed report
    importance_df = feature_cluster.to_frame(name='cluster').reset_index()
    importance_df.columns = ['feature_name', 'cluster']
    importance_df = importance_df.merge(cluster_stats, on='cluster', how='left')
    importance_df = importance_df.sort_values("mean_importance", ascending=False)

    summary = {
        "mean_rmse": results_df["rmse"].mean(),
        "mean_r2": results_df["r2"].mean(),
        "mean_custom_metric": results_df["baseline_metric"].mean(),
        "std_custom_metric": results_df["baseline_metric"].std()
    }

    return results_df, importance_df, summary

