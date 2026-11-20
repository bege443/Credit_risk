import json, os, warnings
import numpy as np, pandas as pd, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, f1_score,
                             accuracy_score, roc_curve, confusion_matrix)
warnings.filterwarnings("ignore")
try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False
    print("xgboost not installed: XGBoost will be skipped")

DEPLOYED = "Random Forest"          # the model used by app.py
OUT = "results"; os.makedirs(OUT, exist_ok=True)
NAVY, TEAL, RED, AMB = "#12304f", "#0b7a6b", "#b3263e", "#b7791f"

df = pd.read_csv("credit.csv")
X = df.drop(columns=["default"]); y = df["default"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42)
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
    "SVM": SVC(probability=True, class_weight="balanced"),
}
if HAVE_XGB:
    models["XGBoost"] = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                                      scale_pos_weight=neg / pos, eval_metric="auc")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
R = {"n_total": int(len(df)), "n_default": int(y.sum()), "n_train": int(len(X_train)), "n_test": int(len(X_test)),
     "test_default": int(y_test.sum()), "test_nondefault": int((y_test == 0).sum()),
     "train_default": int(pos), "train_nondefault": int(neg), "models": {}, "have_xgb": HAVE_XGB}
R["edu_undocumented"] = {str(k): int(v) for k, v in X["EDUCATION"].value_counts().items() if k not in (1, 2, 3, 4)}
R["mar_undocumented"] = {str(k): int(v) for k, v in X["MARRIAGE"].value_counts().items() if k not in (1, 2, 3)}
R["missing_values"] = int(df.isna().sum().sum())
corr = df.corr()["default"].drop("default").sort_values(key=abs, ascending=False)
R["corr_with_default"] = {k: round(float(v), 3) for k, v in corr.items()}
bills = df[[f"BILL_AMT{i}" for i in range(1, 7)]].corr().values
R["bill_corr_min"] = round(float(bills[np.triu_indices(6, 1)].min()), 3)
R["bill_corr_max"] = round(float(bills[np.triu_indices(6, 1)].max()), 3)

fitted, probs = {}, {}
for name, model in models.items():
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", model)])
    cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
    pipe.fit(X_train, y_train)
    prob = pipe.predict_proba(X_test)[:, 1]; pred = (prob >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_test, prob)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    R["models"][name] = {"cv_auc": round(cv_auc, 3), "test_auc": round(roc_auc_score(y_test, prob), 3),
        "ks": round((tpr - fpr).max(), 3), "precision": round(precision_score(y_test, pred), 3),
        "recall": round(recall_score(y_test, pred), 3), "f1": round(f1_score(y_test, pred), 3),
        "accuracy": round(accuracy_score(y_test, pred), 3), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    fitted[name] = pipe; probs[name] = (fpr, tpr, prob)
    print(name, R["models"][name])

table = pd.DataFrame(R["models"]).T.sort_values("test_auc", ascending=False)
table.to_csv(f"{OUT}/model_comparison.csv"); print(table)
R["ranking"] = list(table.index)

# threshold analysis and permutation importance for the deployed model
prob = probs[DEPLOYED][2]; rows = []
for t in (0.2, 0.3, 0.4, 0.5, 0.6):
    p = (prob >= t).astype(int)
    rows.append({"threshold": t, "precision": round(precision_score(y_test, p, zero_division=0), 3),
                 "recall": round(recall_score(y_test, p), 3), "f1": round(f1_score(y_test, p), 3),
                 "accuracy": round(accuracy_score(y_test, p), 3), "flagged": int(p.sum())})
R["thresholds"] = rows; pd.DataFrame(rows).to_csv(f"{OUT}/thresholds.csv", index=False)
pi = permutation_importance(fitted[DEPLOYED], X_test, y_test, scoring="roc_auc", n_repeats=5, random_state=42, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False)
R["importance"] = {k: round(float(v), 4) for k, v in imp.items()}
R["deployed"] = DEPLOYED
joblib.dump(fitted[DEPLOYED], f"{OUT}/credit_model.pkl")

# charts
plt.rcParams.update({"font.size": 10})
fig, ax = plt.subplots(figsize=(5.6, 3.6)); v = [R["n_total"] - R["n_default"], R["n_default"]]
b = ax.bar(["Non-default (0)", "Default (1)"], v, color=[TEAL, RED], width=0.55)
for r_, x_ in zip(b, v): ax.text(r_.get_x() + r_.get_width() / 2, x_ + 500, f"{x_:,}\n({x_ / R['n_total'] * 100:.2f}%)", ha="center")
ax.set_ylim(0, max(v) * 1.2); ax.set_ylabel("Number of customers"); [ax.spines[s].set_visible(False) for s in ("top", "right")]
fig.savefig(f"{OUT}/fig4_5_class_distribution.png", dpi=200, bbox_inches="tight"); plt.close()

c = df.corr(); fig, ax = plt.subplots(figsize=(7.2, 6.2)); im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(c))); ax.set_yticks(range(len(c))); ax.set_xticklabels(c.columns, rotation=90, fontsize=7); ax.set_yticklabels(c.columns, fontsize=7)
fig.colorbar(im, fraction=0.04); fig.savefig(f"{OUT}/fig4_6_correlation.png", dpi=200, bbox_inches="tight"); plt.close()

names = list(R["models"]); met = [("test_auc", "AUC"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score")]
fig, ax = plt.subplots(figsize=(7, 3.8)); w = 0.8 / len(met); cols = [NAVY, AMB, TEAL, RED]
for i, (k, lab) in enumerate(met):
    ax.bar(np.arange(len(names)) + i * w, [R["models"][n][k] for n in names], w, label=lab, color=cols[i])
ax.set_xticks(np.arange(len(names)) + 0.4 - w / 2); ax.set_xticklabels(names); ax.set_ylim(0, 1); ax.legend(ncol=4, fontsize=8, loc="upper center")
[ax.spines[s].set_visible(False) for s in ("top", "right")]; fig.savefig(f"{OUT}/fig4_7_model_comparison.png", dpi=200, bbox_inches="tight"); plt.close()

fig, ax = plt.subplots(figsize=(5.6, 4.4))
for i, n in enumerate(names):
    ax.plot(probs[n][0], probs[n][1], lw=1.8, color=[NAVY, TEAL, AMB, RED][i], label=f"{n} (AUC = {R['models'][n]['test_auc']:.3f})")
ax.plot([0, 1], [0, 1], "--", color="grey", lw=1); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(fontsize=8, loc="lower right")
[ax.spines[s].set_visible(False) for s in ("top", "right")]; fig.savefig(f"{OUT}/fig4_8_roc.png", dpi=200, bbox_inches="tight"); plt.close()

m = R["models"][DEPLOYED]; cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
fig, ax = plt.subplots(figsize=(4.6, 3.8)); ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2): ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=13, color="white" if cm[i, j] > cm.max() / 2 else NAVY)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(["Predicted 0", "Predicted 1"]); ax.set_yticklabels(["Actual 0", "Actual 1"])
fig.savefig(f"{OUT}/fig4_9_confusion_matrix.png", dpi=200, bbox_inches="tight"); plt.close()

top = imp.head(10)[::-1]; fig, ax = plt.subplots(figsize=(5.6, 3.8)); ax.barh(top.index, top.values, color=TEAL)
ax.set_xlabel("Drop in test AUC when the variable is shuffled"); [ax.spines[s].set_visible(False) for s in ("top", "right")]
fig.savefig(f"{OUT}/fig4_10_importance.png", dpi=200, bbox_inches="tight"); plt.close()

json.dump(R, open(f"{OUT}/results.json", "w"), indent=1)
print("Done. Upload the whole 'results' folder (or at least results.json and the PNG files).")