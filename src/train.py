
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import joblib

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "cancer_patient_data_sets.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
TARGET_COL = "Level"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_JOBS = -1  

os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_fig(fig, filename, dpi=150):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    print(f"Saved: {path}")

def plot_distribution(df, numeric_cols):
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        save_fig(fig, f"dist_{col}.png")
        plt.close(fig)

def plot_boxplots(df, numeric_cols):
    if not numeric_cols:
        return
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5*len(numeric_cols))))
    sns.boxplot(data=df[numeric_cols], orient="h", ax=ax)
    ax.set_title("Boxplots of numeric features")
    save_fig(fig, "boxplots_numeric.png")
    plt.close(fig)

def plot_correlation(df, numeric_cols):
    if len(numeric_cols) < 2:
        return
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation matrix")
    save_fig(fig, "correlation_matrix.png")
    plt.close(fig)

def plot_model_comparison(results):
    fig, ax = plt.subplots(figsize=(7, 5))
    names = list(results.keys())
    vals = [results[n] for n in names]
    sns.barplot(x=vals, y=names, orient="h", ax=ax)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Accuracy")
    ax.set_title("Model accuracy comparison (test set)")
    save_fig(fig, "model_accuracy_comparison.png")
    plt.close(fig)

def plot_cv_bar(cv_summary, filename="cv_comparison.png"):
    names = list(cv_summary.keys())
    means = [cv_summary[n][0] for n in names]
    stds = [cv_summary[n][1] for n in names]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.5*len(names))))
    ax.barh(names, means, xerr=stds, align='center')
    ax.set_xlim(0, 1)
    ax.set_xlabel("CV Accuracy (mean ± std)")
    ax.set_title("Cross-validation comparison")
    save_fig(fig, filename)
    plt.close(fig)

def plot_confusion(cm, labels, title, filename):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    save_fig(fig, filename)
    plt.close(fig)

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    print(f"Loaded dataset with shape: {df.shape}")
    return df


def preprocess(df, target_col):
    df = df.drop(columns=["Patient Id"], errors="ignore")
    
    before = df.shape[0]
    df = df.drop_duplicates().reset_index(drop=True)
    after = df.shape[0]
    print(f"Dropped {before - after} duplicate rows")

    summary = df.describe(include='all')
    summary.to_csv(os.path.join(OUTPUT_DIR, "data_summary.csv"))
    print(f"Saved data summary to {os.path.join(OUTPUT_DIR, 'data_summary.csv')}")

    X = df.drop(columns=[target_col])
    y = df[target_col].copy()

    if y.dtype == object:
        try:
            y_numeric = y.astype(str).str.extract(r'(\d+)').astype(float).iloc[:, 0]
            if not y_numeric.isna().all():
                mask = ~y_numeric.isna()
                if mask.all():
                    y = y_numeric.astype(int)
                    print("Converted target strings to numeric levels.")
                else:
                    pass
        except Exception:
            pass

    label_encoder = None
    if y.dtype == object or y.nunique() <= 20 and not np.issubdtype(y.dtype, np.integer):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        mapping = dict(enumerate(label_encoder.classes_))
        pd.Series(mapping).to_csv(os.path.join(OUTPUT_DIR, "target_label_mapping.csv"))
        print(f"Encoded target and saved mapping to {os.path.join(OUTPUT_DIR, 'target_label_mapping.csv')}")

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    print(f"Numeric cols: {numeric_cols}")
    print(f"Categorical cols: {categorical_cols}")

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ], remainder="drop")

    return X, y, preprocessor, numeric_cols, categorical_cols, label_encoder

def perform_cv_and_tuning(X_train, y_train, preprocessor):

    log_clf = Pipeline(steps=[("preprocessor", preprocessor),
                              ("model", LogisticRegression(max_iter=500, random_state=RANDOM_STATE))])

    dt_base = Pipeline(steps=[("preprocessor", preprocessor),
                              ("model", DecisionTreeClassifier(random_state=RANDOM_STATE))])

    rf_base = Pipeline(steps=[("preprocessor", preprocessor),
                              ("model", RandomForestClassifier(random_state=RANDOM_STATE))])

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    baseline_scores = {}
    print("\nPerforming baseline cross-validation (accuracy) for models...")

    for name, pipe in [("LogisticRegression", log_clf), ("DecisionTree", dt_base), ("RandomForest", rf_base)]:
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=N_JOBS)
        baseline_scores[name] = (scores.mean(), scores.std(), scores)
        print(f"{name} CV accuracy: mean={scores.mean():.4f}, std={scores.std():.4f}")

    dt_param_grid = {
        "model__criterion": ["gini", "entropy"],
        "model__max_depth": [None, 5, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }
    print("\nStarting GridSearchCV for DecisionTree...")
    dt_grid = GridSearchCV(
        dt_base,
        param_grid=dt_param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=N_JOBS,
        verbose=1,
        refit=True,
    )
    dt_grid.fit(X_train, y_train)
    print(f"DecisionTree best params: {dt_grid.best_params_}")
    print(f"DecisionTree best CV score: {dt_grid.best_score_:.4f}")

    rf_param_grid = {
        "model__n_estimators": [100, 200, 400],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": ["sqrt", "log2", None],
    }
    print("\nStarting GridSearchCV for RandomForest...")
    rf_grid = GridSearchCV(
        rf_base,
        param_grid=rf_param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=N_JOBS,
        verbose=1,
        refit=True,
    )
    rf_grid.fit(X_train, y_train)
    print(f"RandomForest best params: {rf_grid.best_params_}")
    print(f"RandomForest best CV score: {rf_grid.best_score_:.4f}")

    tuned_scores = {}
    for name, gs in [("DecisionTree_tuned", dt_grid), ("RandomForest_tuned", rf_grid)]:
        best_pipe = gs.best_estimator_
        scores = cross_val_score(best_pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=N_JOBS)
        tuned_scores[name] = (scores.mean(), scores.std(), scores)
        print(f"{name} CV accuracy: mean={scores.mean():.4f}, std={scores.std():.4f}")

    pd.DataFrame(dt_grid.cv_results_).to_csv(os.path.join(OUTPUT_DIR, "dt_grid_cv_results.csv"), index=False)
    pd.DataFrame(rf_grid.cv_results_).to_csv(os.path.join(OUTPUT_DIR, "rf_grid_cv_results.csv"), index=False)
    print(f"Saved GridSearchCV results to {OUTPUT_DIR}")

    with open(os.path.join(OUTPUT_DIR, "best_params.txt"), "w") as f:
        f.write("DecisionTree best params:\n")
        f.write(str(dt_grid.best_params_) + "\n")
        f.write(f"DecisionTree best CV score: {dt_grid.best_score_:.6f}\n\n")
        f.write("RandomForest best params:\n")
        f.write(str(rf_grid.best_params_) + "\n")
        f.write(f"RandomForest best CV score: {rf_grid.best_score_:.6f}\n")
    print("Saved best parameters to best_params.txt")

    joblib.dump(dt_grid.best_estimator_, os.path.join(OUTPUT_DIR, "DecisionTree_best_pipeline.joblib"))
    joblib.dump(rf_grid.best_estimator_, os.path.join(OUTPUT_DIR, "RandomForest_best_pipeline.joblib"))
    print("Saved best pipelines for DecisionTree and RandomForest.")

    return baseline_scores, dt_grid, rf_grid, tuned_scores

def train_evaluate_final(X_train, X_test, y_train, y_test, preprocessor, dt_grid, rf_grid):
    final_models = {}

    log_pipe = Pipeline(steps=[("preprocessor", preprocessor),
                               ("model", LogisticRegression(max_iter=500, random_state=RANDOM_STATE))])
    log_pipe.fit(X_train, y_train)
    final_models["LogisticRegression"] = log_pipe
    joblib.dump(log_pipe, os.path.join(OUTPUT_DIR, "LogisticRegression_pipeline.joblib"))

    dt_pipe = dt_grid.best_estimator_
    dt_pipe.fit(X_train, y_train)
    final_models["DecisionTree"] = dt_pipe
    joblib.dump(dt_pipe, os.path.join(OUTPUT_DIR, "DecisionTree_tuned_pipeline.joblib"))

    rf_pipe = rf_grid.best_estimator_
    rf_pipe.fit(X_train, y_train)
    final_models["RandomForest"] = rf_pipe
    joblib.dump(rf_pipe, os.path.join(OUTPUT_DIR, "RandomForest_tuned_pipeline.joblib"))

    results = {}
    for name, pipe in final_models.items():
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        cr = classification_report(y_test, y_pred, zero_division=0)
        print(f"\n{name} -> Accuracy: {acc:.4f}, Recall (macro): {rec:.4f}, F1 (macro): {f1:.4f}")
        print("Classification Report:")
        print(cr)

        cm = confusion_matrix(y_test, y_pred)
        labels = np.unique(np.concatenate((y_test, y_pred)))
        plot_confusion(cm, labels, f"{name} Confusion Matrix", f"confusion_{name}.png")

        cr_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        cr_df = pd.DataFrame(cr_dict).transpose()
        cr_df.to_csv(os.path.join(OUTPUT_DIR, f"classification_report_{name}.csv"))

        results[name] = {"accuracy": acc, "recall_macro": rec, "f1_macro": f1, "classification_report": cr_dict}

        if name in ("DecisionTree", "RandomForest"):
            try:
                pre = pipe.named_steps["preprocessor"]

                num_cols = pre.transformers_[0][2] if pre.transformers_[0][2] is not None else []
                cat_transformer = pre.transformers_[1][1].named_steps["onehot"]
                cat_cols = pre.transformers_[1][2] if pre.transformers_[1][2] is not None else []
                cat_names = []
                if len(cat_cols) > 0:
                    cat_names = list(cat_transformer.get_feature_names_out(cat_cols))
                feature_names = list(num_cols) + cat_names

                importance = pipe.named_steps["model"].feature_importances_
                fi = pd.Series(importance, index=feature_names).sort_values(ascending=False).head(30)
                fig, ax = plt.subplots(figsize=(8, min(6, 0.4*len(fi))))
                sns.barplot(x=fi.values, y=fi.index, ax=ax)
                ax.set_title(f"Top feature importances ({name})")
                save_fig(fig, f"feature_importance_{name}.png")
                plt.close(fig)
            except Exception as e:
                print(f"Could not compute feature importances for {name}: {e}")

    return results, final_models

def main():
    df = load_data(DATA_PATH)

    target_col = TARGET_COL

    print("\nInitial dataset info:")
    print(df.info())
    null_counts = df.isnull().sum().sort_values(ascending=False)
    null_counts.to_csv(os.path.join(OUTPUT_DIR, "null_counts.csv"))
    print(f"Saved null counts to {os.path.join(OUTPUT_DIR, 'null_counts.csv')}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        plot_distribution(df, numeric_cols)
        plot_boxplots(df, numeric_cols)
        plot_correlation(df, numeric_cols)

    X, y, preprocessor, num_cols, cat_cols, label_encoder = preprocess(df, target_col)

    class_counts = pd.Series(y).value_counts().sort_index()
    class_counts.to_csv(os.path.join(OUTPUT_DIR, "target_distribution.csv"))
    print(f"Target distribution saved to {os.path.join(OUTPUT_DIR, 'target_distribution.csv')}")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax)
    ax.set_xlabel("Target class")
    ax.set_ylabel("Count")
    ax.set_title("Target class distribution")
    save_fig(fig, "target_distribution.png")
    plt.close(fig)

    stratify_arg = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify_arg
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    baseline_scores, dt_grid, rf_grid, tuned_scores = perform_cv_and_tuning(X_train, y_train, preprocessor)

    cv_summary = {}
    for k, v in baseline_scores.items():
        cv_summary[k] = (v[0], v[1])  
    for k, v in tuned_scores.items():
        cv_summary[k] = (v[0], v[1])

    cv_df = pd.DataFrame({
        "model": list(cv_summary.keys()),
        "cv_mean_accuracy": [cv_summary[m][0] for m in cv_summary],
        "cv_std_accuracy": [cv_summary[m][1] for m in cv_summary],
    })
    cv_df.to_csv(os.path.join(OUTPUT_DIR, "cv_summary.csv"), index=False)
    print(f"Saved cross-validation summary to {os.path.join(OUTPUT_DIR, 'cv_summary.csv')}")

    plot_cv_bar(cv_summary)

    final_results, final_models = train_evaluate_final(X_train, X_test, y_train, y_test, preprocessor, dt_grid, rf_grid)

    results_df = pd.DataFrame({
        model: {
            "accuracy": res["accuracy"],
            "recall_macro": res["recall_macro"],
            "f1_macro": res["f1_macro"]
        } for model, res in final_results.items()
    }).T
    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_test_results_summary.csv"))
    print(f"Saved model test results to {os.path.join(OUTPUT_DIR, 'model_test_results_summary.csv')}")

    accs = {m: res["accuracy"] for m, res in final_results.items()}
    plot_model_comparison(accs)

    print("\nAll outputs (plots, models, CV & grid results) saved in:", OUTPUT_DIR)
    print("Script finished successfully.")

if __name__ == "__main__":
    main()
