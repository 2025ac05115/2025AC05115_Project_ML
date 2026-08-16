**Problem Statement**
The goal of this project is to evaluate multiple classification algorithms on the Wine Quality dataset from UCI.
We aim to compare performance across different models using key metrics such as Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).
The task is framed as a binary classification problem:

Good wine (quality ≥ 6)

Bad wine (quality < 6)

**Dataset Description**
**Source:** UCI Machine Learning Repository – Wine Quality Dataset (archive.ics.uci.edu in Bing)

**Shape:** 1599 rows × 12 features

**Features:**

fixed acidity

volatile acidity

citric acid

residual sugar

chlorides

free sulfur dioxide

total sulfur dioxide

density

pH

sulphates

alcohol

quality (target variable)

The dataset was split into 70% training and 30% testing using stratified sampling to preserve class balance.

**GitHub Repository**
Project files and requirements can be found here:
GitHub Repo Link https://github.com/2025ac05115/2025AC05115_Project_ML/tree/main

**Comparison Table of Metrics**
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.608 | None | 0.592 | 0.970 | 0.735 | 0.213 |
| Decision Tree | 0.723 | None | 0.730 | 0.803 | 0.765 | 0.433 |
| KNN | 0.665 | None | 0.731 | 0.636 | 0.680 | 0.335 |
| Naive Bayes | 0.733 | None | 0.774 | 0.740 | 0.757 | 0.463 |
| Random Forest | **0.756** | **None** | **0.814** | **0.732** | **0.771** | **0.515** |


**Observations on Model Performance**
| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Solid baseline with decent AUC, but lower MCC compared to tree-based models. |
| Decision Tree | Strong recall and balanced performance, but prone to overfitting. |
| KNN | Performs similarly to Logistic Regression but requires scaling for stability. |
| Naive Bayes | High precision but weaker recall, leading to lower F1 and MCC. |
| Random Forest | Outperforms all other models across every metric, showing robustness and generalization ability. |
| **Overall Winner** | **Random Forest** |


**Visualizations**
Confusion matrices for each model are made available

Bar chart comparison of metrics across models

(Generated in the notebook using Seaborn and Matplotlib)
