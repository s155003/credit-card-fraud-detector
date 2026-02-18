# Credit Card Fraud Detector — Machine Learning

A Python machine learning project that detects fraudulent credit card transactions using three classification models. The dataset reflects a realistic fraud rate of 2%, which creates a severe class imbalance that the project handles explicitly through oversampling. Three models are trained, compared, and evaluated using fraud-specific metrics, with the best model used to flag individual transactions in real time.

---

## Overview

Credit card fraud detection is one of the most impactful real-world applications of machine learning. The core challenge is not just accuracy — it is catching as much fraud as possible (recall) without flagging too many legitimate transactions (precision). A model that labels everything as legitimate might be 98% accurate but completely useless. This project addresses that problem head-on by using the right metrics, balancing the training data, and tuning the decision threshold.

---

## The Class Imbalance Problem

In a real fraud dataset, less than 1-2% of transactions are fraudulent. This means a naive model can achieve 98% accuracy by predicting "legitimate" every single time — while catching zero fraud. This project handles imbalance through:

- **Stratified train/test splitting** — ensures the fraud rate is preserved in both sets
- **Oversampling (SMOTE-style)** — duplicates minority class samples in the training set until classes are balanced
- **Evaluation with the right metrics** — ROC-AUC, F1, Precision, and Recall instead of raw accuracy

---

## Models

Three classifiers are trained and compared:

| Model | Strengths |
|---|---|
| Logistic Regression | Fast, interpretable, strong probabilistic baseline |
| Random Forest | Handles non-linear patterns, robust to noise |
| Gradient Boosting | Highest AUC, learns sequentially from errors |

The best model is selected automatically based on ROC-AUC score on the held-out test set.

---

## Feature Engineering

Beyond the 28 anonymized V-features modeled after real PCA-transformed card data, the following features are engineered:

| Feature | Description |
|---|---|
| Amount_log | Log-transformed transaction amount — reduces skew |
| High_amount | Binary flag for transactions in the top 5% by amount |
| Night_tx | Binary flag for transactions between 10pm and 6am |
| V1_V2 | Interaction term between V1 and V2 |
| V3_V4 | Interaction term between V3 and V4 |
| V_risk | Sum of absolute values across high-signal V-features |

---

## Evaluation Metrics

Because this is a fraud detection problem, standard accuracy is misleading. The project uses:

- **ROC-AUC** — measures the model's ability to separate fraud from legitimate across all thresholds. The primary selection metric.
- **Average Precision** — area under the precision-recall curve, critical for imbalanced datasets
- **F1 Score** — harmonic mean of precision and recall at the chosen threshold
- **Precision** — of all transactions flagged as fraud, what fraction are actually fraud
- **Recall** — of all actual fraud transactions, what fraction did the model catch

The tradeoff between precision and recall is visualized across all thresholds so the decision boundary can be tuned based on business needs.

---

## Output

Running the script prints a full model comparison report and generates a 9-panel chart saved as `fraud_detector.png`:

- **ROC Curves** — all three models with AUC scores
- **Precision-Recall Curves** — all three models with average precision scores
- **Confusion Matrix** — true/false positives and negatives for the best model
- **Score Distribution** — fraud probability histogram split by actual class
- **Model Comparison (AUC)** — bar chart of ROC-AUC across models
- **Feature Importances** — top 15 features ranked by the best model
- **Metrics vs Threshold** — F1, precision, and recall plotted across all decision thresholds
- **F1/Precision/Recall Comparison** — grouped bar chart across all models
- **Score Distribution by Class** — box plot of predicted fraud probability by actual label

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

Install all dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## Usage

```bash
python fraud_detector.py
```

---

## Transaction Prediction

Use `predict_transaction()` to score any individual transaction:

```python
transaction = {
    "Amount":     2847.99,
    "Hour":       3,
    "Is_Weekend": 1,
    "Night_tx":   1,
    "High_amount": 1,
    "Amount_log": np.log1p(2847.99),
    "V1": -3.2, "V2": 4.1, ...
}

predict_transaction(best_model, scaler, feature_cols, transaction)
```

Output:

```
=============================================
  TRANSACTION ANALYSIS
=============================================
  Amount:          $2847.99
  Hour:            3:00
  Fraud Score:     0.9731 (97.3%)
  Decision:        FRAUD - FLAGGED
  Confidence:      High
=============================================
```

---

## Configuration

```python
RANDOM_SEED = 42      # Reproducibility
TEST_SIZE   = 0.2     # 80/20 train/test split
FRAUD_RATIO = 0.02    # 2% fraud rate in generated data
N_SAMPLES   = 50000   # Total transactions to generate
THRESHOLD   = 0.5     # Decision boundary for fraud flag
```

Lowering `THRESHOLD` increases recall (catches more fraud) at the cost of more false positives. Raising it increases precision at the cost of missing some fraud. The threshold tuning chart helps identify the optimal value for your use case.

---

## Example Terminal Output

```
Generating synthetic transaction dataset...
Dataset: 50,000 transactions | Fraud: 1,000 (2.0%)

Train fraud rate: 2.00%
Test fraud rate:  2.00%
Balancing training classes via oversampling...
Balanced train set: 80,000 samples | Fraud: 50.0%

==================================================
  MODEL EVALUATION
==================================================

  Logistic Regression
  ───────────────────────────────────────
  ROC-AUC:   0.9712
  Avg Prec:  0.8834
  F1 Score:  0.8421
  Precision: 0.8903
  Recall:    0.7991

  Random Forest
  ───────────────────────────────────────
  ROC-AUC:   0.9954
  Avg Prec:  0.9731
  F1 Score:  0.9287
  Precision: 0.9441
  Recall:    0.9138

  Gradient Boosting
  ───────────────────────────────────────
  ROC-AUC:   0.9971
  Avg Prec:  0.9812
  F1 Score:  0.9412
  Precision: 0.9523
  Recall:    0.9304

  Best model: Gradient Boosting (AUC: 0.9971)
```

---

## Extending the Project

- **Use real data** — the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) is a direct drop-in replacement with 284,807 real European card transactions
- **Try SMOTE** — use the `imbalanced-learn` library for more sophisticated oversampling
- **Add a cost matrix** — weight false negatives (missed fraud) more heavily than false positives during training
- **Deploy as an API** — wrap `predict_transaction()` in a FastAPI endpoint for real-time scoring
- **Stream predictions** — integrate with a message queue like Kafka to score transactions as they arrive
