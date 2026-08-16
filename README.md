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
| Logistic Regression | 0.7333 | 0.8243 | 0.7549 | 0.7432 | 0.7490 | 0.4647 |
| Decision Tree | 0.7792 | 0.7772 | 0.7871 | 0.8054 | 0.7962 | 0.5555 |
| KNN | 0.7313 | 0.7972 | 0.7443 | 0.7588 | 0.7514 | 0.4591 |
| Naive Bayes | 0.7229 | 0.7980 | 0.7672 | 0.6926 | 0.7280 | 0.4496 |
| Random Forest | **0.7979** | **0.8777** | **0.8077** | **0.8171** | **0.8124** | **0.5935** |


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
