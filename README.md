# Detecting Anomalous Transactions

## Problem

Credit card fraud is highly imbalanced: only **473 of 283,726 transactions (0.17%)** in this dataset are fraudulent.

This project uses **Isolation Forest and Local Outlier Factor (LOF)** to identify unusual transactions and assess their usefulness for fraud detection.

## Data

* **Source:** Kaggle Credit Card Fraud dataset
* **Transactions:** 283,726 after removing 1,081 duplicates
* **Features:** `V1`–`V28`, `Time`, `Amount`
* **Transformation:** Applied `log(1 + Amount)` to the highly skewed `Amount` feature
* `Class` was not used as a model input and was used only for evaluation and contamination analysis.

## Methodology

Two unsupervised anomaly-detection methods were compared:

* **Isolation Forest** — identifies observations that are easier to isolate from the rest of the data.
* **LOF** — identifies observations with unusually low local density.

The initial models used a **0.17% contamination rate**, approximately matching the fraud proportion in the dataset.

## Results

| Model            | Precision | Recall | F1-score |
| ---------------- | --------: | -----: | -------: |
| Isolation Forest |     19.9% |  20.3% |    0.201 |
| LOF              |      0.2% |   0.2% |    0.002 |

Isolation Forest identified **96 of 473** known fraud cases, while LOF identified **1**.

The results show that Isolation Forest was better suited to this dataset than LOF, although many flagged transactions were legitimate.

### Contamination Analysis

| Contamination | Precision |    Recall |        F1 |
| ------------: | --------: | --------: | --------: |
|        0.0005 |     14.8% |      4.4% |     0.068 |
|        0.0010 |     25.7% |     15.4% |     0.193 |
|        0.0017 |     19.9% |     20.3% |     0.201 |
|    **0.0025** | **16.8%** | **25.2%** | **0.201** |
|        0.0050 |     12.8% |     38.3% |     0.191 |

The **0.0025** setting was selected as the final operating point because it increased recall while maintaining the same F1-score as 0.0017. This selection was based on labeled evaluation data, so the final contamination setting is **label-informed**, although the anomaly models themselves were trained without fraud labels.

## Tools

Python, pandas, scikit-learn (IsolationForest, LocalOutlierFactor, StandardScaler), seaborn, matplotlib.
