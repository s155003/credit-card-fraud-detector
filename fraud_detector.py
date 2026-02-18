import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                              roc_curve, precision_recall_curve, average_precision_score,
                              f1_score, precision_score, recall_score)
from sklearn.utils import resample
import warnings
warnings.filterwarnings("ignore")


RANDOM_SEED   = 42
TEST_SIZE     = 0.2
FRAUD_RATIO   = 0.02
N_SAMPLES     = 50000
THRESHOLD     = 0.5


def generate_data(n=N_SAMPLES, fraud_ratio=FRAUD_RATIO):
    np.random.seed(RANDOM_SEED)
    n_fraud   = int(n * fraud_ratio)
    n_legit   = n - n_fraud

    def make_transactions(n, is_fraud):
        hour        = np.random.choice(range(24), n, p=([0.02]*6 + [0.04]*4 + [0.06]*8 + [0.04]*6) if not is_fraud
                                       else ([0.07]*6 + [0.03]*4 + [0.03]*8 + [0.07]*6))
        amount      = np.random.exponential(80, n) if not is_fraud else np.random.exponential(300, n)
        amount      = np.clip(amount, 0.5, 25000)
        v_cols      = {}
        for i in range(1, 29):
            mean = 0 if not is_fraud else np.random.uniform(-2, 2)
            std  = 1 if not is_fraud else np.random.uniform(1.5, 3.5)
            v_cols[f"V{i}"] = np.random.normal(mean, std, n)
        df = pd.DataFrame(v_cols)
        df["Amount"]      = amount.round(2)
        df["Hour"]        = hour
        df["Is_Weekend"]  = np.random.choice([0, 1], n, p=[0.71, 0.29])
        df["Class"]       = int(is_fraud)
        return df

    legit = make_transactions(n_legit, False)
    fraud = make_transactions(n_fraud, True)
    df    = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=RANDOM_SEED)
    return df


def add_features(df):
    df = df.copy()
    df["Amount_log"]   = np.log1p(df["Amount"])
    df["High_amount"]  = (df["Amount"] > df["Amount"].quantile(0.95)).astype(int)
    df["Night_tx"]     = df["Hour"].apply(lambda h: 1 if h < 6 or h >= 22 else 0)
    df["V1_V2"]        = df["V1"] * df["V2"]
    df["V3_V4"]        = df["V3"] * df["V4"]
    df["V_risk"]       = df[["V1","V2","V3","V4","V10","V11","V12","V14","V17"]].abs().sum(axis=1)
    return df


def balance_classes(X_train, y_train, method="oversample"):
    X = pd.concat([X_train, y_train], axis=1)
    majority = X[X["Class"] == 0]
    minority = X[X["Class"] == 1]
    if method == "oversample":
        minority_up = resample(minority, replace=True, n_samples=len(majority), random_state=RANDOM_SEED)
        balanced    = pd.concat([majority, minority_up])
    else:
        majority_dn = resample(majority, replace=False, n_samples=len(minority)*10, random_state=RANDOM_SEED)
        balanced    = pd.concat([majority_dn, minority])
    balanced = balanced.sample(frac=1, random_state=RANDOM_SEED)
    return balanced.drop("Class", axis=1), balanced["Class"]


def evaluate(name, y_true, y_pred, y_prob):
    auc = roc_auc_score(y_true, y_prob)
    ap  = average_precision_score(y_true, y_prob)
    f1  = f1_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    print(f"\n  {name}")
    print(f"  {'─'*35}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"  Avg Prec:  {ap:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Precision: {pre:.4f}  (of flagged, how many are real fraud)")
    print(f"  Recall:    {rec:.4f}  (of all fraud, how many did we catch)")
    return {"auc": auc, "ap": ap, "f1": f1, "precision": pre, "recall": rec}


def plot_results(y_test, results, feature_cols, best_name):
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("Credit Card Fraud Detection — ML Analysis", fontsize=16, fontweight="bold")

    ax1 = fig.add_subplot(3, 3, 1)
    for name, data in results.items():
        fpr, tpr, _ = roc_curve(y_test, data["probs"])
        ax1.plot(fpr, tpr, label=f"{name} (AUC={data['scores']['auc']:.3f})", linewidth=2)
    ax1.plot([0,1],[0,1], "k--", linewidth=1)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curves")
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(3, 3, 2)
    for name, data in results.items():
        prec, rec, _ = precision_recall_curve(y_test, data["probs"])
        ax2.plot(rec, prec, label=f"{name} (AP={data['scores']['ap']:.3f})", linewidth=2)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curves")
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(3, 3, 3)
    best_preds = results[best_name]["preds"]
    cm = confusion_matrix(y_test, best_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3,
                xticklabels=["Legit","Fraud"], yticklabels=["Legit","Fraud"])
    ax3.set_title(f"Confusion Matrix ({best_name})")
    ax3.set_ylabel("Actual")
    ax3.set_xlabel("Predicted")

    ax4 = fig.add_subplot(3, 3, 4)
    best_probs = results[best_name]["probs"]
    fraud_probs = best_probs[y_test == 1]
    legit_probs = best_probs[y_test == 0]
    ax4.hist(legit_probs, bins=50, alpha=0.6, color="#1f77b4", label="Legitimate", density=True)
    ax4.hist(fraud_probs, bins=50, alpha=0.6, color="#d62728", label="Fraud",      density=True)
    ax4.axvline(THRESHOLD, color="black", linestyle="--", linewidth=2, label=f"Threshold ({THRESHOLD})")
    ax4.set_xlabel("Fraud Probability")
    ax4.set_ylabel("Density")
    ax4.set_title("Score Distribution")
    ax4.legend()

    ax5 = fig.add_subplot(3, 3, 5)
    model_names = list(results.keys())
    aucs  = [results[n]["scores"]["auc"] for n in model_names]
    colors = ["#2ca02c" if n == best_name else "#1f77b4" for n in model_names]
    ax5.bar(model_names, aucs, color=colors)
    ax5.set_ylim(min(aucs) - 0.05, 1.0)
    ax5.set_ylabel("ROC-AUC")
    ax5.set_title("Model Comparison — ROC-AUC")
    for i, v in enumerate(aucs):
        ax5.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=9)

    ax6 = fig.add_subplot(3, 3, 6)
    best_model = results[best_name]["model"]
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        sorted_idx  = np.argsort(importances)[::-1][:15]
        sorted_names = [feature_cols[i] for i in sorted_idx]
        bar_colors   = ["#d62728" if i == 0 else "#1f77b4" for i in range(len(sorted_idx))]
        ax6.barh(sorted_names[::-1], importances[sorted_idx][::-1], color=bar_colors[::-1])
        ax6.set_xlabel("Importance")
        ax6.set_title(f"Feature Importances ({best_name})")

    ax7 = fig.add_subplot(3, 3, 7)
    thresholds   = np.linspace(0.01, 0.99, 100)
    f1_scores    = [f1_score(y_test, (best_probs >= t).astype(int)) for t in thresholds]
    precisions   = [precision_score(y_test, (best_probs >= t).astype(int), zero_division=0) for t in thresholds]
    recalls      = [recall_score(y_test, (best_probs >= t).astype(int)) for t in thresholds]
    ax7.plot(thresholds, f1_scores,  label="F1",        linewidth=2)
    ax7.plot(thresholds, precisions, label="Precision",  linewidth=2)
    ax7.plot(thresholds, recalls,    label="Recall",     linewidth=2)
    ax7.axvline(THRESHOLD, color="black", linestyle="--", linewidth=1.5, label=f"Threshold={THRESHOLD}")
    ax7.set_xlabel("Decision Threshold")
    ax7.set_ylabel("Score")
    ax7.set_title("Metrics vs Threshold")
    ax7.legend(fontsize=8)

    ax8 = fig.add_subplot(3, 3, 8)
    f1_vals = [results[n]["scores"]["f1"] for n in model_names]
    pre_vals = [results[n]["scores"]["precision"] for n in model_names]
    rec_vals = [results[n]["scores"]["recall"] for n in model_names]
    x = np.arange(len(model_names))
    w = 0.25
    ax8.bar(x - w, f1_vals,  w, label="F1",        color="#1f77b4")
    ax8.bar(x,     pre_vals, w, label="Precision",  color="#ff7f0e")
    ax8.bar(x + w, rec_vals, w, label="Recall",     color="#2ca02c")
    ax8.set_xticks(x)
    ax8.set_xticklabels(model_names, fontsize=8)
    ax8.set_title("F1 / Precision / Recall Comparison")
    ax8.legend(fontsize=8)

    ax9 = fig.add_subplot(3, 3, 9)
    fraud_amounts = pd.Series(results[best_name]["probs"]).values
    actual_fraud  = y_test.values
    legit_amt     = [p for p, a in zip(fraud_amounts, actual_fraud) if a == 0]
    fraud_amt     = [p for p, a in zip(fraud_amounts, actual_fraud) if a == 1]
    ax9.boxplot([legit_amt, fraud_amt], labels=["Legitimate", "Fraud"])
    ax9.set_ylabel("Predicted Fraud Probability")
    ax9.set_title("Score Distribution by Class")

    plt.tight_layout()
    plt.savefig("fraud_detector.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Chart saved as fraud_detector.png")


def predict_transaction(model, scaler, feature_cols, transaction: dict, threshold=THRESHOLD):
    row = pd.DataFrame([transaction])
    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0
    row       = row[feature_cols]
    row_sc    = scaler.transform(row)
    prob      = model.predict_proba(row_sc)[0][1]
    flagged   = prob >= threshold
    print(f"\n{'='*45}")
    print(f"  TRANSACTION ANALYSIS")
    print(f"{'='*45}")
    print(f"  Amount:          ${transaction.get('Amount', 'N/A')}")
    print(f"  Hour:            {transaction.get('Hour', 'N/A')}:00")
    print(f"  Fraud Score:     {prob:.4f} ({prob*100:.1f}%)")
    print(f"  Decision:        {'FRAUD - FLAGGED' if flagged else 'LEGITIMATE'}")
    print(f"  Confidence:      {'High' if abs(prob - 0.5) > 0.3 else 'Medium' if abs(prob - 0.5) > 0.15 else 'Low'}")
    print(f"{'='*45}\n")
    return prob, flagged


print("Generating synthetic transaction dataset...")
df = generate_data()
print(f"Dataset: {len(df):,} transactions | Fraud: {df['Class'].sum():,} ({df['Class'].mean()*100:.1f}%)")

df = add_features(df)

feature_cols = [c for c in df.columns if c != "Class"]
X = df[feature_cols]
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)

print(f"\nTrain fraud rate: {y_train.mean()*100:.2f}%")
print(f"Test fraud rate:  {y_test.mean()*100:.2f}%")

print("Balancing training classes via oversampling...")
X_train_bal, y_train_bal = balance_classes(X_train, y_train, method="oversample")
print(f"Balanced train set: {len(X_train_bal):,} samples | Fraud: {y_train_bal.mean()*100:.1f}%")

scaler       = StandardScaler()
X_train_sc   = scaler.fit_transform(X_train_bal)
X_test_sc    = scaler.transform(X_test)

models = {
    "Logistic Regression":  LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_SEED),
    "Random Forest":        RandomForestClassifier(n_estimators=200, max_depth=10,
                                                    class_weight="balanced", random_state=RANDOM_SEED),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                        max_depth=5, subsample=0.8,
                                                        random_state=RANDOM_SEED),
}

print(f"\n{'='*50}")
print(f"  MODEL EVALUATION")
print(f"{'='*50}")

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_sc, y_train_bal)
    probs  = model.predict_proba(X_test_sc)[:, 1]
    preds  = (probs >= THRESHOLD).astype(int)
    scores = evaluate(name, y_test, preds, probs)
    results[name] = {"model": model, "probs": probs, "preds": preds, "scores": scores}

best_name  = max(results, key=lambda k: results[k]["scores"]["auc"])
best_model = results[best_name]["model"]
print(f"\n  Best model: {best_name} (AUC: {results[best_name]['scores']['auc']:.4f})")

print("\nFull classification report (best model):")
print(classification_report(y_test, results[best_name]["preds"], target_names=["Legitimate", "Fraud"]))

plot_results(y_test, results, feature_cols, best_name)

legit_tx = {
    "Amount": 42.50, "Hour": 14, "Is_Weekend": 0,
    "Amount_log": np.log1p(42.50), "High_amount": 0, "Night_tx": 0,
    **{f"V{i}": np.random.normal(0, 1) for i in range(1, 29)},
    "V1_V2": 0.1, "V3_V4": -0.2, "V_risk": 2.1
}

fraud_tx = {
    "Amount": 2847.99, "Hour": 3, "Is_Weekend": 1,
    "Amount_log": np.log1p(2847.99), "High_amount": 1, "Night_tx": 1,
    **{f"V{i}": np.random.normal(2, 3) for i in range(1, 29)},
    "V1_V2": 4.2, "V3_V4": -3.8, "V_risk": 18.4
}

print("--- Sample Legitimate Transaction ---")
predict_transaction(best_model, scaler, feature_cols, legit_tx)

print("--- Sample Fraudulent Transaction ---")
predict_transaction(best_model, scaler, feature_cols, fraud_tx)

print("Training complete. Output saved to fraud_detector.png")
