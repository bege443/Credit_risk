
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from xgboost import XGBClassifier


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("dataset/processed_credit_data.csv")

X = df.drop(columns=["credit_risk"])
y = df["credit_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Dataset loaded successfully.")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# 2. DEFINE MODELS
# ==========================================

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        scale_pos_weight=negative / positive,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1
    )
}


# ==========================================
# 3. DEFINE THRESHOLDS
# ==========================================

thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

all_results = []


# ==========================================
# 4. TRAIN AND ANALYSE MODELS
# ==========================================

for name, model in models.items():

    print("\nTraining temporary model:", name)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        result = {
            "Model": name,
            "Threshold": threshold,
            "Accuracy": accuracy_score(
                y_test, predictions
            ),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, predictions, zero_division=0
            )
        }

        all_results.append(result)

        print(
            f"Threshold: {threshold:.2f} | "
            f"Accuracy: {result['Accuracy']:.4f} | "
            f"Precision: {result['Precision']:.4f} | "
            f"Recall: {result['Recall']:.4f} | "
            f"F1: {result['F1 Score']:.4f}"
        )


# ==========================================
# 5. SAVE RESULTS SEPARATELY
# ==========================================

output_folder = "threshold_results"

os.makedirs(output_folder, exist_ok=True)

results_df = pd.DataFrame(all_results)

results_df.to_csv(
    os.path.join(output_folder, "threshold_comparison.csv"),
    index=False
)

print("\n========== THRESHOLD COMPARISON ==========")
print(results_df.round(4).to_string(index=False))


# ==========================================
# 6. GENERATE F1 SCORE COMPARISON CHART
# ==========================================

plt.figure(figsize=(10, 6))

for name in models.keys():

    model_data = results_df[
        results_df["Model"] == name
    ]

    plt.plot(
        model_data["Threshold"],
        model_data["F1 Score"],
        marker="o",
        label=name
    )

plt.title("F1 Score Comparison Across Prediction Thresholds")
plt.xlabel("Prediction Threshold")
plt.ylabel("F1 Score")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "threshold_f1_comparison.png"),
    dpi=300
)

plt.close()

print("\nThreshold analysis completed successfully!")
print("Results saved in:", output_folder)
