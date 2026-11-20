from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import os
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# ==========================================
# 1. LOAD THE PROCESSED DATASET
# ==========================================

df = pd.read_csv("dataset/processed_credit_data.csv")

X = df.drop(columns=["credit_risk"])
y = df["credit_risk"]

print("Dataset loaded successfully")
print("Dataset shape:", df.shape)

# ==========================================
# 2. SPLIT THE DATASET (80:20)
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# ==========================================
# 3. DEFINE THE MODELS
# ==========================================

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "SVM": SVC(
        probability=True,
        class_weight="balanced"
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        scale_pos_weight=neg / pos,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1
    )
}

# ==========================================
# 4. TRAIN AND EVALUATE THE MODELS
# ==========================================

results = {}
fitted_models = {}
test_predictions = {}
test_probabilities = {}

for name, model in models.items():

    print("\nTraining:", name)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", model)
    ])

    # ==========================================
    # 5-FOLD CROSS-VALIDATION
    # ==========================================

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    print(
        "5-Fold CV ROC-AUC:",
        round(cv_mean, 4),
        "+/-",
        round(cv_std, 4)
    )

    # ==========================================
    # MODEL TRAINING AND TESTING
    # ==========================================

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    # Store fitted models and predictions for visualisation
    fitted_models[name] = pipeline
    test_predictions[name] = predictions
    test_probabilities[name] = probabilities
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions
    ).ravel()

    results[name] = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC AUC": roc_auc_score(y_test, probabilities),
        "CV ROC-AUC Mean": cv_mean,
        "CV ROC-AUC Std": cv_std,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }

    print("Accuracy:", round(results[name]["Accuracy"], 4))
    print("Precision:", round(results[name]["Precision"], 4))
    print("Recall:", round(results[name]["Recall"], 4))
    print("F1 Score:", round(results[name]["F1 Score"], 4))
    print("ROC AUC:", round(results[name]["ROC AUC"], 4))

# ==========================================
# 7. GENERATE CONFUSION MATRICES
# ==========================================

from sklearn.metrics import ConfusionMatrixDisplay

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, (name, predictions) in zip(
    axes.ravel(), test_predictions.items()
):
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        ax=ax,
        cmap="Blues",
        colorbar=False
    )

    ax.set_title(name)

plt.tight_layout()
plt.savefig(
    "results/confusion_matrices.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Confusion matrices saved successfully.")


# ==========================================
# 8. GENERATE ROC CURVES
# ==========================================

from sklearn.metrics import RocCurveDisplay

fig, ax = plt.subplots(figsize=(9, 7))

for name, probabilities in test_probabilities.items():

    RocCurveDisplay.from_predictions(
        y_test,
        probabilities,
        name=name,
        ax=ax
    )

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    label="Random Classifier"
)

ax.set_title("ROC Curve Comparison of Machine Learning Models")
ax.legend(loc="lower right")

plt.tight_layout()
plt.savefig(
    "results/roc_curves.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("ROC curves saved successfully.")

print("\nAll visualisations generated successfully!")

# ==========================================
# 9. FEATURE IMPORTANCE ANALYSIS
# ==========================================

os.makedirs("results", exist_ok=True)

for model_name in ["Random Forest", "XGBoost"]:

    print("\nFeature Importance Analysis:", model_name)

    # Get the trained classifier
    classifier = fitted_models[model_name].named_steps["classifier"]

    # --------------------------------------
    # A. BUILT-IN FEATURE IMPORTANCE
    # --------------------------------------

    importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": classifier.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    importance_df.to_csv(
        f"results/{model_name.lower().replace(' ', '_')}_feature_importance.csv",
        index=False
    )

    # Plot the top 15 features
    top_features = importance_df.head(15).sort_values(
        by="Importance",
        ascending=True
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    plt.title(f"Top 15 Feature Importance - {model_name}")
    plt.xlabel("Importance Score")
    plt.ylabel("Customer Features")
    plt.tight_layout()

    plt.savefig(
        f"results/{model_name.lower().replace(' ', '_')}_feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------
    # B. PERMUTATION IMPORTANCE
    # --------------------------------------

    print("Calculating permutation importance...")

    permutation_result = permutation_importance(
        fitted_models[model_name],
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=5,
        random_state=42,
        n_jobs=-1
    )

    permutation_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": permutation_result.importances_mean,
        "Std": permutation_result.importances_std
    })

    permutation_df = permutation_df.sort_values(
        by="Importance",
        ascending=False
    )

    permutation_df.to_csv(
        f"results/{model_name.lower().replace(' ', '_')}_permutation_importance.csv",
        index=False
    )

    top_permutation = permutation_df.head(15).sort_values(
        by="Importance",
        ascending=True
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        top_permutation["Feature"],
        top_permutation["Importance"],
        xerr=top_permutation["Std"]
    )

    plt.title(f"Top 15 Permutation Importance - {model_name}")
    plt.xlabel("Mean Decrease in ROC-AUC")
    plt.ylabel("Customer Features")
    plt.tight_layout()

    plt.savefig(
        f"results/{model_name.lower().replace(' ', '_')}_permutation_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

print("\nFeature importance analysis completed successfully!")

