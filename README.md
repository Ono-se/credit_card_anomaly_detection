# Detecting Anomalous Transactions

## Problem

Credit card fraud is a needle-in-a-haystack problem: in this dataset, only **473 of 283,726 transactions (0.17%)** are confirmed fraud. That extreme imbalance rules out a standard classifier trained the usual way — with so few positive examples, a supervised model has very little to learn from, and in a real deployment, next month's fraud patterns may not resemble last month's labeled cases at all.

This project takes an **anomaly-detection approach** instead: rather than learning "what fraud looks like" from a handful of labeled examples, the model learns "what normal looks like" from the overwhelming majority of legitimate transactions, and flags anything that deviates sharply from that norm. This is the same logic a fraud team would want in production — a system that can catch novel fraud patterns it's never explicitly seen labeled, not just recognize past cases.

## Data

- **Source:** Kaggle's credit card fraud dataset — 284,807 transactions, 31 columns. Features `V1`–`V28` are PCA-transformed for confidentiality; `Time` and `Amount` are provided in their original form.
- **Cleaning:** 1,081 exact duplicate rows were identified and dropped (283,726 remaining). No missing values.
- **Transformation:** `Amount` is heavily right-skewed (mean $88, max $25,691) — a log transform (`Amount_log = log(1 + Amount)`) was applied to make its distribution usable by distance- and density-based methods.
- **Target:** `Class` (1 = fraud, 0 = legitimate) is used **only for evaluating and tuning model hyperparameters — never as an input feature.** This is the central design constraint of the project: the models never see the label while learning what "normal" looks like.

## Methodology

Two unsupervised anomaly detection methods were trained and compared:

- **Isolation Forest** — isolates anomalies by how few random splits it takes to separate a point from the rest of the data; fraud, being rare and different, tends to isolate quickly.
- **Local Outlier Factor (LOF)** — flags points whose local neighborhood density is much lower than their neighbors', on standardized features.

Both were initialized with `contamination=0.0017`, matching the dataset's known fraud rate — a reasonable starting assumption, though in production this rate would need to be estimated or set as a policy choice rather than read off known labels.

## Results

| Model | Precision | Recall | F1-score |
|---|---|---|---|
| Isolation Forest | 19.9% | 20.3% | 0.201 |
| Local Outlier Factor | 0.2% | 0.2% | 0.002 |

Isolation Forest substantially outperformed LOF — it identified **96 of 473** fraud cases, compared to LOF's **1**. The gap points to something specific about this data: LOF flags points that are unusual *relative to their local neighborhood*, but fraudulent transactions here didn't consistently sit in sparse regions of feature space — many looked locally similar to nearby legitimate transactions, even though they were globally rare. Isolation Forest's global, density-agnostic approach to isolating rare points was a better fit for this kind of anomaly.

**Key takeaway:** not every statistical outlier is fraud, and not all anomaly-detection methods find the same anomalies. The choice of method needs to match the actual geometry of the rare class, not just "pick an unsupervised model."

### Tuning

The `contamination` parameter — Isolation Forest's estimate of what fraction of the data is anomalous — was swept using labels for evaluation only:

| Contamination | Precision | Recall | F1 |
|---|---|---|---|
| 0.0005 | 14.8% | 4.4% | 0.068 |
| 0.0010 | 25.7% | 15.4% | 0.193 |
| 0.0017 | 19.9% | 20.3% | 0.201 |
| **0.0025** | **16.8%** | **25.2%** | **0.201** |
| 0.0050 | 12.8% | 38.3% | 0.191 |

0.0025 and 0.0017 are essentially tied on F1 — raising `contamination` mainly traded precision for recall rather than improving overall detection quality. **0.0025 was selected as the final operating point**, favoring the higher recall (catching more fraud, at the cost of more false positives) — a defensible choice given that missed fraud is typically more costly than a flagged-but-legitimate transaction.

## Tools

Python, pandas, scikit-learn (`IsolationForest`, `LocalOutlierFactor`, `StandardScaler`), seaborn, matplotlib.
