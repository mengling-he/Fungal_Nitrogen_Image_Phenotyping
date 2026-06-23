import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("default")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score,mean_squared_error, mean_absolute_error, r2_score, roc_auc_score


# -----------------------------
# 1) CV splitter that matches the data structure
#  1.1) Method 1: create folds so that each fold's test set has exactly 1 sample from each group
#    - Outer: 5 folds, each fold holds out 1 sample per (fungus × nitrogen)
#    - Inner: 4 folds on the training set, holds out 1 sample per (fungus × nitrogen)
# -----------------------------
def make_one_per_group_folds(df, group_cols, n_splits, random_state=42):
    """
    Create folds where each fold's test set contains exactly 1 sample from each group
    defined by group_cols.

    Assumption:
      - Each group has exactly n_splits samples (outer: 5, inner: 4 after outer split).
    Returns:
      folds: list of (train_idx_array, test_idx_array) using integer positions (iloc indices)
    """
    rng = np.random.default_rng(random_state)

    # Store fold assignment per row (by integer position)
    fold_id = np.full(len(df), -1, dtype=int)

    # Assign fold ids within each group after shuffling
    for _, g in df.groupby(group_cols, sort=False):
        idx = g.index.to_numpy()  # NOTE: these are df.index values, not positions
        idx_shuffled = idx.copy()
        rng.shuffle(idx_shuffled)

        if len(idx_shuffled) != n_splits:
            raise ValueError(
                f"Group {group_cols} has {len(idx_shuffled)} samples, expected {n_splits}. "
                f"Group key example: {tuple(g.iloc[0][group_cols].values)}"
            )

        # fold 0..n_splits-1 gets exactly one sample from this group
        for f, row_index in enumerate(idx_shuffled):
            fold_id[df.index.get_loc(row_index)] = f  # convert index -> positional integer

    if np.any(fold_id < 0):
        raise RuntimeError("Some rows did not get assigned to a fold. Check grouping columns.")

    folds = []
    for f in range(n_splits):
        test_pos = np.where(fold_id == f)[0]
        train_pos = np.where(fold_id != f)[0]
        folds.append((train_pos, test_pos))

    return folds



#  1.2) Method 2: target-stratified CV
def make_sorted_target_folds(
    df,
    target_col,
    n_splits,
    group_cols=None,      # optional: keep balancing within each group
    random_state=42,
):
    """
    Build CV folds by sorting target and assigning fold IDs round-robin.

    If group_cols is None:
      - global sorted target -> round-robin fold assignment.

    If group_cols is provided:
      - do the same within each group, which keeps groups balanced AND target-spread per fold.
    """
    rng = np.random.default_rng(random_state)
    fold_id = np.full(len(df), -1, dtype=int)

    if group_cols is None:
        y = df[target_col].to_numpy()
        order = np.argsort(y)

        # Optional: jitter equal values so ties don't always land in same fold pattern
        # (stable and reproducible)
        # We shuffle within blocks of equal y.
        y_sorted = y[order]
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and y_sorted[end] == y_sorted[start]:
                end += 1
            if end - start > 1:
                block = order[start:end].copy()
                rng.shuffle(block)
                order[start:end] = block
            start = end

        for i, pos in enumerate(order):
            fold_id[pos] = i % n_splits

    else:
        group_cols = list(group_cols)
        for _, g in df.groupby(group_cols, sort=False):
            # positions (iloc indices) for rows in this group
            group_pos = np.array([df.index.get_loc(ix) for ix in g.index.to_numpy()])

            y_g = g[target_col].to_numpy()
            order_local = np.argsort(y_g)
            group_pos_sorted = group_pos[order_local]

            # tie-handling within group
            y_sorted = y_g[order_local]
            start = 0
            while start < len(group_pos_sorted):
                end = start + 1
                while end < len(group_pos_sorted) and y_sorted[end] == y_sorted[start]:
                    end += 1
                if end - start > 1:
                    block = group_pos_sorted[start:end].copy()
                    rng.shuffle(block)
                    group_pos_sorted[start:end] = block
                start = end

            for i, pos in enumerate(group_pos_sorted):
                fold_id[pos] = i % n_splits

    if np.any(fold_id < 0):
        raise RuntimeError("Some rows did not get assigned to a fold.")

    folds = []
    for f in range(n_splits):
        test_pos = np.where(fold_id == f)[0]
        train_pos = np.where(fold_id != f)[0]
        folds.append((train_pos, test_pos))
    return folds




# -----------------------------
# 2)  CV without parameter tuning for simple task multinomial logistic regression (for small traits)
# -----------------------------



def cv_classify(
    df,
    feature_cols,
    model_type="logistic",          # "logistic", "lda", "lasso"
    target_col="Fungal_Strain",
    group_cols=("Fungal_Strain", "Nitrogen_Level"),
    outer_splits=5,
    outer_plate_folds=None,
    random_state=42,
    # logistic / lasso specific
    max_iter=5000,
    C=1.0,                          # inverse regularization strength for lasso
):
    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()
    classes = np.unique(y)

    # ===== BUILD MODEL =====
    if model_type == "logistic":
        clf = LogisticRegression(
            penalty="none",
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=max_iter
        )
    elif model_type == "lasso":
        clf = LogisticRegression(
            penalty="l1",
            C=C,
            multi_class="multinomial",
            solver="saga",          # saga supports l1 + multinomial
            max_iter=max_iter
        )
    elif model_type == "lda":
        clf = LinearDiscriminantAnalysis()
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from: 'logistic', 'lda', 'lasso'")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf)
    ])

    # ===== OUTER FOLDS =====
    if outer_plate_folds is None:
        outer_folds = make_sorted_target_folds(
            df=df,
            target_col=group_cols,
            n_splits=outer_splits,
            random_state=random_state
        )
    else:
        outer_folds = []
        for test_plates in outer_plate_folds:
            test_mask = df["plate"].isin(test_plates)
            train_mask = ~test_mask
            train_pos = np.where(train_mask)[0]
            test_pos  = np.where(test_mask)[0]
            outer_folds.append((train_pos, test_pos))

    # ===== EVALUATION =====
    fold_rows = []
    for fold_i, (train_pos, test_pos) in enumerate(outer_folds, start=1):
        X_train, y_train = X[train_pos], y[train_pos]
        X_test,  y_test  = X[test_pos],  y[test_pos]

        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        acc  = accuracy_score(y_test, y_pred)
        bacc = balanced_accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average="macro")

        try:
            if len(classes) == 2:
                auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                auc = roc_auc_score(
                    y_test, y_proba,
                    multi_class="ovr",
                    average="macro",
                    labels=classes
                )
        except ValueError:
            auc = np.nan

        fold_rows.append({
            "fold": fold_i,
            "model": model_type,
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "f1_macro": f1,
            "auc_macro_ovr": auc
        })

    results_df = pd.DataFrame(fold_rows)
    summary = {
        "model": model_type,
        "mean_accuracy":           float(results_df["accuracy"].mean()),
        "std_accuracy":            float(results_df["accuracy"].std()),
        "mean_balanced_accuracy":  float(results_df["balanced_accuracy"].mean()),
        "std_balanced_accuracy":   float(results_df["balanced_accuracy"].std()),
        "mean_f1_macro":           float(results_df["f1_macro"].mean()),
        "std_f1_macro":            float(results_df["f1_macro"].std()),
        "mean_auc_macro_ovr":      float(results_df["auc_macro_ovr"].mean(skipna=True)),
        "std_auc_macro_ovr":       float(results_df["auc_macro_ovr"].std(skipna=True)),
        "classes": classes
    }
    return results_df, summary




# -----------------------------
# 3) nested CV modeling function for classification
#    Your strategy:
#      - Outer 5-fold: test = 6 samples (1 per strain×nitrogen), train = 24
#      - Inner 4-fold: within train, tune hyperparams using same structure
# -----------------------------

# -----------------------------
# 3.1) Generic nested CV engine
# -----------------------------
def nested_cv_classification(
    df,
    feature_cols,
    target_col,
    group_cols,                 # e.g. ("fungus", "nitrogen")
    estimator,                  # Pipeline or model
    param_grid,
    outer_splits=5,
    outer_plate_folds=None,   # ⭐ plate-level folds, a list of fold of test plates
    inner_splits=4,
    random_state=42,
    scoring="balanced_accuracy",
    n_jobs=1 # when you call it on HPC, pass n_jobs=48.
):
    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()
    classes = np.unique(y)

    # ========= OUTER FOLDS =========
    if outer_plate_folds is None:
        outer_folds = make_sorted_target_folds(
            df=df,
            target_col=target_col,
            n_splits=outer_splits,
            random_state=random_state
        )
    else:
        outer_folds = []
        for test_plates in outer_plate_folds:
            test_mask = df["plate"].isin(test_plates)
            train_mask = ~test_mask
            train_pos = np.where(train_mask)[0]
            test_pos  = np.where(test_mask)[0]
            outer_folds.append((train_pos, test_pos))
    #print("Number of outer folds:", len(outer_folds))

    fold_rows = []
    pooled_true, pooled_pred = [], []

    for fold_i, (train_pos, test_pos) in enumerate(outer_folds, start=1):
        X_train, y_train = X[train_pos], y[train_pos]
        X_test,  y_test  = X[test_pos],  y[test_pos]

        # Inner folds on the 24-sample training set
        df_train = df.iloc[train_pos].reset_index(drop=True)
        inner_folds = make_one_per_group_folds(
            df=df_train,
            group_cols=list(group_cols),
            n_splits=inner_splits,
            random_state=random_state + fold_i
        )

        grid = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
            cv=inner_folds,
            refit=True,
            verbose=0,
            n_jobs=n_jobs
        )
        grid.fit(X_train, y_train)

        y_pred = grid.best_estimator_.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        bacc = balanced_accuracy_score(y_test, y_pred)

        fold_rows.append({
            "fold": fold_i,
            "accuracy": acc,
            "balanced_accuracy": bacc,
            **{f"best_{k}": v for k, v in grid.best_params_.items()}
        })

        pooled_true.append(y_test)
        pooled_pred.append(y_pred)

    results_df = pd.DataFrame(fold_rows)
    y_true_all = np.concatenate(pooled_true)
    y_pred_all = np.concatenate(pooled_pred)

    summary = {
        "mean_accuracy": float(results_df["accuracy"].mean()),
        "mean_balanced_accuracy": float(results_df["balanced_accuracy"].mean()),
        "classes": classes,
        "pooled_confusion_matrix": confusion_matrix(y_true_all, y_pred_all, labels=classes),
    }
    return results_df, summary


# -----------------------------
# 3.2) PLS-DA: implement as PLS -> use X-scores as features -> multinomial LR
# -----------------------------
class PLSScores(BaseEstimator, TransformerMixin):
    """
    Transformer that fits PLSRegression (using one-vs-rest targets) and returns X scores.
    Works for multiclass by one-hot encoding y during fit.
    """
    def __init__(self, n_components=2, scale=False):
        self.n_components = n_components
        self.scale = scale
        self._pls = None
        self._classes = None

    def fit(self, X, y):
        self._classes = np.unique(y)
        Y = pd.get_dummies(pd.Series(y, dtype="category"), drop_first=False).to_numpy()
        self._pls = PLSRegression(n_components=self.n_components, scale=self.scale)
        self._pls.fit(X, Y)
        return self

    def transform(self, X):
        if self._pls is None:
            raise RuntimeError("Transformer not fitted yet.")
        return self._pls.transform(X)


# -----------------------------
# 3.3) Model specs (your list)
# -----------------------------
def get_model_specs(random_state=42, use_xgboost_if_available=True):
    specs = {}

    # 1) Elastic Net multinomial logistic regression
    specs["ElasticNet_MultinomialLR"] = {
        "estimator": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                multi_class="multinomial",
                max_iter=20000,
                n_jobs=1,
                random_state=random_state
            ))
        ]),
        "param_grid": {
            "clf__C": np.logspace(-2, 2, 9),
            "clf__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        }
    }

    specs["Lasso_MultinomialLR"] = {
    "estimator": Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l1",
            solver="saga",
            multi_class="multinomial",
            max_iter=20000,
            n_jobs=1,
            random_state=random_state
        ))
    ]),
    "param_grid": {
        "clf__C": np.logspace(-3, 1, 9),  # stronger regularization than ElasticNet
    }
}

    # 2) Shrinkage LDA (interpretability baseline, use shrinkage for stability with small n and correlated features)
    specs["ShrinkageLDA"] = {
        "estimator": Pipeline([
            ("scale", StandardScaler()),
            ("lda", LinearDiscriminantAnalysis())
        ]),
        "param_grid": {
            "lda__solver": ["lsqr", "eigen"],
            "lda__shrinkage": ["auto"],  # stable default for small n / correlated features
        }
    }



    # 3) PLS-DA (PLS scores + multinomial logistic regression)Partial Least Squares Discriminant Analysis
    # Keep components small (2–5) due to n=24 in training folds.
    specs["PLSDA_PLSScores_plus_LR"] = {
        "estimator": Pipeline([
            ("scale", StandardScaler()),
            ("pls", PLSScores()),
            ("clf", LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                multi_class="multinomial",
                max_iter=20000,
                n_jobs=1,
                random_state=random_state
            ))
        ]),
        "param_grid": {
            "pls__n_components": [2, 3, 4, 5],
            "clf__C": np.logspace(-2, 2, 9),
        }
    }

    # 4) Random Forest (secondary, regularized)
    specs["RandomForest"] = {
        "estimator": RandomForestClassifier(
            random_state=random_state,
            class_weight="balanced",
            n_jobs=1
        ),
        "param_grid": {
            "n_estimators": [300, 600],
            "max_depth": [2, 3, 4],          # keep shallow for n=24 training
            "min_samples_leaf": [1, 2, 3],
            "max_features": ["sqrt", 0.3, 0.5],
        }
    }

    # Optional: XGBoost (if installed)
    if use_xgboost_if_available:
        try:
            from xgboost import XGBClassifier
            specs["XGBoost_Regularized"] = {
                "estimator": XGBClassifier(
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    tree_method="hist",
                    random_state=random_state,
                    n_jobs=1
                ),
                "param_grid": {
                    "n_estimators": [200, 500],
                    "max_depth": [2, 3],
                    "learning_rate": [0.05, 0.1],
                    "subsample": [0.6, 0.8],
                    "colsample_bytree": [0.6, 0.8],
                    "min_child_weight": [3, 5],
                    "reg_lambda": [1, 5, 10],
                }
            }
        except Exception:
            pass

    return specs


# -----------------------------
# 3.4) Run all models fairly (same nested CV settings)
# -----------------------------
def run_models(df, feature_cols,
               target_col="nitrogen",
               group_cols=("fungus", "nitrogen"),
               outer_splits=5, 
               outer_plate_folds=None, 
               inner_splits=4,
               random_state=42,n_jobs=1): # when you call it on HPC, pass n_jobs=48.):

    specs = get_model_specs(random_state=random_state, use_xgboost_if_available=True)

    all_results = {}
    all_summaries = {}

    for name, spec in specs.items():
        res_df, summ = nested_cv_classification(
            df=df,
            feature_cols=feature_cols,
            target_col=target_col,
            group_cols=group_cols,
            estimator=spec["estimator"],
            param_grid=spec["param_grid"],
            outer_splits=outer_splits,
            outer_plate_folds=outer_plate_folds, 
            inner_splits=inner_splits,
            random_state=random_state,
            scoring="balanced_accuracy",
            n_jobs=n_jobs
        )
        all_results[name] = res_df
        all_summaries[name] = summ

        print(f"\n=== {name} ===")
        print("Pooled confusion matrix (rows=true, cols=pred):")
        print(pd.DataFrame(summ["pooled_confusion_matrix"],
                           index=summ["classes"], columns=summ["classes"]))

    compare = pd.DataFrame({
        name: {
            "mean_accuracy": summ["mean_accuracy"],
            "mean_balanced_accuracy": summ["mean_balanced_accuracy"],
        }
        for name, summ in all_summaries.items()
    }).T.sort_values("mean_balanced_accuracy", ascending=False)

    print("\n=== Comparison ===")
    print(compare)

    return all_results, all_summaries, compare







####  4) nested CV modeling function for regression #### 
#    Your strategy:
# -----------------------------

# -----------------------------
# 4.1) Generic nested CV engine
# -----------------------------

def nested_cv_group_regression(
    df,
    feature_cols,
    target_col,
    group_cols,                 # e.g. ("fungus", "nitrogen") or any columns you want to balance on
    estimator,                  # Pipeline or model
    param_grid,
    outer_splits=5,
    inner_splits=4,
    random_state=42,
    scoring="neg_root_mean_squared_error",  # good default
    n_jobs=1 # when you call it on HPC, pass n_jobs=48.
):
    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy().astype(float)


    outer_folds = make_one_per_group_folds(
    df=df,
    group_cols=list(group_cols),
    n_splits=outer_splits,
    random_state=random_state
    )

    fold_rows = []
    pooled_true, pooled_pred,fold_id_list = [], [],[]

    for fold_i, (train_pos, test_pos) in enumerate(outer_folds, start=1):
        X_train, y_train = X[train_pos], y[train_pos]
        X_test,  y_test  = X[test_pos],  y[test_pos]

        # Inner folds on training set
        df_train = df.iloc[train_pos].reset_index(drop=True)
        inner_folds = make_one_per_group_folds(
            df=df_train,
            group_cols=list(group_cols),
            n_splits=inner_splits,
            random_state=random_state + fold_i
        )

        grid = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
            cv=inner_folds,
            refit=True,
            verbose=0,
            n_jobs=n_jobs
        )
        grid.fit(X_train, y_train)

        y_pred = grid.best_estimator_.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae  = float(mean_absolute_error(y_test, y_pred))
        r2   = float(r2_score(y_test, y_pred))

        fold_rows.append({
            "fold": fold_i,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            **{f"best_{k}": v for k, v in grid.best_params_.items()}
        })

        pooled_true.append(y_test)
        pooled_pred.append(y_pred)
        fold_id_list.append(np.repeat(fold_i, repeats=len(y_pred)))

    results_df = pd.DataFrame(fold_rows)

    y_true_all = np.concatenate(pooled_true)
    y_pred_all = np.concatenate(pooled_pred)
    fold_id_all =  np.concatenate(fold_id_list)

    summary = {
        "mean_rmse": float(results_df["rmse"].mean()),
        "mean_mae": float(results_df["mae"].mean()),
        "mean_r2": float(results_df["r2"].mean()),
        "pooled_rmse": float(np.sqrt(mean_squared_error(y_true_all, y_pred_all))),
        "pooled_mae": float(mean_absolute_error(y_true_all, y_pred_all)),
        "pooled_r2": float(r2_score(y_true_all, y_pred_all)),
        "y_true_all": y_true_all,
        "y_pred_all": y_pred_all,
        "fold":fold_id_all
    }
    return results_df, summary



def nested_cv_sorted_target_regression(
    df,
    feature_cols,
    target_col,
    estimator,
    param_grid,
    outer_splits=5,
    outer_plate_folds=None,   # ⭐ plate-level folds, a list of fold of test plates
    inner_splits=4,
    random_state=42,
    scoring="r2",# neg_root_mean_squared_error is also good, but r2 is more interpretable for regression
    n_jobs=1 
):
    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy().astype(float)
    # ========= OUTER FOLDS =========
    if outer_plate_folds is None:
        outer_folds = make_sorted_target_folds(
            df=df,
            target_col=target_col,
            n_splits=outer_splits,
            random_state=random_state
        )
    else:
        outer_folds = []
        for test_plates in outer_plate_folds:
            test_mask = df["plate"].isin(test_plates)
            train_mask = ~test_mask
            train_pos = np.where(train_mask)[0]
            test_pos  = np.where(test_mask)[0]
            outer_folds.append((train_pos, test_pos))
    #print("Number of outer folds:", len(outer_folds))
    fold_rows = []
    pooled_true, pooled_pred, fold_id_list = [], [], []

    for fold_i, (train_pos, test_pos) in enumerate(outer_folds, start=1):
        X_train, y_train = X[train_pos], y[train_pos]
        X_test,  y_test  = X[test_pos],  y[test_pos]

        # inner folds on training set
        df_train = df.iloc[train_pos].reset_index(drop=True)
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
            verbose=0,
            n_jobs=n_jobs
        )
        grid.fit(X_train, y_train)

        # ⭐ Predictions for both Training and Testing
        y_train_pred = grid.best_estimator_.predict(X_train)
        y_test_pred  = grid.best_estimator_.predict(X_test)

        # ⭐ Calculate Training Metrics
        train_rmse = float(np.sqrt(mean_squared_error(y_train, y_train_pred)))
        train_r2   = float(r2_score(y_train, y_train_pred))

        # Calculate Testing Metrics
        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
        test_mae  = float(mean_absolute_error(y_test, y_test_pred))
        test_r2   = float(r2_score(y_test, y_test_pred))

        fold_rows.append({
            "fold": fold_i,
            "train_rmse": train_rmse,  # Added
            "train_r2": train_r2,      # Added
            "rmse": test_rmse,
            "mae": test_mae,
            "r2": test_r2,
            **{f"best_{k}": v for k, v in grid.best_params_.items()}
        })

        pooled_true.append(y_test)
        pooled_pred.append(y_test_pred)
        fold_id_list.append(np.repeat(fold_i, repeats=len(y_test_pred)))
    results_df = pd.DataFrame(fold_rows)

    y_true_all = np.concatenate(pooled_true)
    y_pred_all = np.concatenate(pooled_pred)
    fold_id_all =  np.concatenate(fold_id_list)

    summary = {
        "mean_rmse": float(results_df["rmse"].mean()),
        "mean_mae": float(results_df["mae"].mean()),
        "mean_r2": float(results_df["r2"].mean()),
        "median_r2": float(results_df["r2"].median()),
        "pooled_rmse": float(np.sqrt(mean_squared_error(y_true_all, y_pred_all))),
        "pooled_mae": float(mean_absolute_error(y_true_all, y_pred_all)),
        "pooled_r2": float(r2_score(y_true_all, y_pred_all)),
        "y_true_all": y_true_all,
        "y_pred_all": y_pred_all,
        "fold":fold_id_all
    }
    return results_df, summary
#results_df is the result of each fold including hyperparameter
# summary is a summary across fold by mean and pooled metrics



def get_model_specs_regression(random_state=42, use_xgboost_if_available=True):
    specs = {}

    # 1) Elastic Net regression
    specs["ElasticNet"] = {
        "estimator": Pipeline([
            ("scale", StandardScaler()),
            ("reg", ElasticNet(max_iter=20000, random_state=random_state))
        ]),
        "param_grid": {
            "reg__alpha": np.logspace(-3, 2, 12),
            "reg__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        }
    }

    # 2) Ridge (very stable baseline)
    specs["Ridge"] = {
        "estimator": Pipeline([
            ("scale", StandardScaler()),
            ("reg", Ridge(random_state=random_state))
        ]),
        "param_grid": {
            "reg__alpha": np.logspace(-3, 3, 13),
        }
    }

    # 3) PLS regression
    specs["PLSRegression"] = {
        "estimator": Pipeline([
            ("scale", StandardScaler()),
            ("pls", PLSRegression())
        ]),
        "param_grid": {
            "pls__n_components": [1, 2, 3, 4, 5],
        }
    }

    # 4) Random Forest regressor (keep shallow)
    specs["RandomForestRegressor"] = {
        "estimator": RandomForestRegressor(
            random_state=random_state,
            n_jobs=1
        ),
        "param_grid": {
            "n_estimators": [300, 600],
            "max_depth": [2, 3, 4],
            "min_samples_leaf": [1, 2, 3],
            "max_features": ["sqrt", 0.3, 0.5],
        }
    }

    # Optional: XGBoost regressor not used now, if need to use
    if use_xgboost_if_available:
        try:
            from xgboost import XGBRegressor
            specs["XGBoost_Regressor"] = {
                "estimator": XGBRegressor(
                    objective="reg:squarederror",
                    tree_method="hist",
                    random_state=random_state,
                    max_depth=3,           # Limit tree depth to prevent overfitting on small samples
                    subsample=0.8,          # Use 80% of data per tree to increase robustness
                    n_jobs=1
                ),
                "param_grid": {
                "reg__n_estimators": [20, 50, 100],  # Reduce number of boosting rounds
                "reg__learning_rate": [0.01, 0.05], # Lower learning rate for smoother convergence
                "reg__max_depth": [2, 3],           # Shallow trees are sufficient for small n
                "reg__reg_alpha": [0.1, 1.0],       # L1 regularization (Lasso term)
                "reg__reg_lambda": [0.1, 1.0]       # L2 regularization (Ridge term)
                }
            }
        except Exception:
            pass

    return specs


def run_models_regression(df, feature_cols,target_col,
                          group_cols=None,outer_folds=None,outer_splits=5, inner_splits=4,
                          random_state=42,scoring="neg_root_mean_squared_error",
                          cv_method="sorted_target",  # or "one_per_group"
                          n_jobs=48):
    specs = get_model_specs_regression(random_state=random_state, use_xgboost_if_available=True)

    all_results = {}
    all_summaries = {}

    for name, spec in specs.items():
        print(f"--- Training {name} ---")
        if cv_method == "one_per_group":
            if group_cols is None:
                raise ValueError("group_cols must be provided when cv_method='one_per_group'")

            res_df, summ = nested_cv_group_regression(
                df=df,
                feature_cols=feature_cols,
                target_col=target_col,
                group_cols=group_cols,
                estimator=spec["estimator"],
                param_grid=spec["param_grid"],
                outer_splits=outer_splits,
                inner_splits=inner_splits,
                random_state=random_state,
                scoring=scoring,
            )
        elif cv_method == "sorted_target":
            # group_cols not needed/ignored for sorted_target CV
            res_df, summ = nested_cv_sorted_target_regression(
                df=df,
                feature_cols=feature_cols,
                target_col=target_col,
                estimator=spec["estimator"],
                param_grid=spec["param_grid"],
                outer_plate_folds=outer_folds,
                outer_splits=outer_splits,
                inner_splits=inner_splits,
                random_state=random_state,
                scoring=scoring,
                n_jobs=n_jobs
            )
        else:
            raise ValueError(f"Unknown cv_method: {cv_method}")
        
        all_results[name] = res_df
        all_summaries[name] = summ

        print(f"\n=== {name} ===")
        print(res_df)

    compare = pd.DataFrame({
        name: {
            "mean_rmse": summ["mean_rmse"],
            "mean_mae": summ["mean_mae"],
            "mean_r2": summ["mean_r2"],
            "pooled_rmse": summ["pooled_rmse"],
            "pooled_r2": summ["pooled_r2"],
        } for name, summ in all_summaries.items()
    }).T.sort_values("mean_rmse", ascending=True)

    print("\n=== Comparison (sorted by mean_rmse) ===")
    print(compare)

    return all_results, all_summaries, compare
# of eac dataset (view* mask *date)
# results_df is the result of each fold including hyperparameter for one model
# summary is a summary across fold by mean and pooled metrics for one model (including y_true and y_predict)
# compare is a summary of each model