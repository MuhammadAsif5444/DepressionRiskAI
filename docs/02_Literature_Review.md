# Chapter 2: Literature Review

## 2.1 Introduction

This chapter reviews existing research related to depression screening, the Patient Health Questionnaire-9 (PHQ-9), machine learning, and explainable artificial intelligence.

The purpose of this review is to understand existing approaches and identify the research gap addressed by DepressionRiskAI.

---

## 2.2 Depression Screening

Depression screening is an important part of identifying individuals who may be experiencing depressive symptoms.

Traditional screening approaches commonly use standardized questionnaires. These questionnaires provide a structured method for collecting information about symptoms such as low mood, lack of interest, sleep difficulties, fatigue, appetite changes, concentration difficulties, and thoughts of self-harm.

Automated machine-learning approaches have increasingly been investigated as research tools for analyzing such information.

However, machine-learning screening systems must be carefully evaluated because prediction accuracy does not necessarily mean that a system can provide a clinical diagnosis.

---

## 2.3 Patient Health Questionnaire-9

The Patient Health Questionnaire-9 (PHQ-9) is a widely used questionnaire for assessing depressive symptoms.

It contains nine questions corresponding to the diagnostic criteria for depressive symptoms.

Each question is scored from 0 to 3:

| Response | Score |
|---|---:|
| Not at all | 0 |
| Several days | 1 |
| More than half the days | 2 |
| Nearly every day | 3 |

The total PHQ-9 score ranges from 0 to 27.

The commonly used severity interpretation is:

| Score | Category |
|---:|---|
| 0–4 | Minimal |
| 5–9 | Mild |
| 10–14 | Moderate |
| 15–19 | Moderately severe |
| 20–27 | Severe |

In DepressionRiskAI, these five categories are used as the target classes for machine-learning experiments.

---

## 2.4 Machine Learning for Depression Screening

Machine learning can identify patterns in structured data and use those patterns to classify new observations.

In depression-related research, machine-learning methods have been investigated using different types of information, including questionnaires, demographic information, behavioral information, and other digital signals.

Classification algorithms can be particularly useful when the objective is to assign observations to predefined categories.

In this project, the classification task consists of predicting one of five PHQ-9 severity categories.

---

## 2.5 Logistic Regression

Logistic Regression is a commonly used statistical and machine-learning classification method.

Although the name contains the word "regression", it can be used for classification tasks.

In this project, Logistic Regression was used as a baseline model.

The baseline model achieved an accuracy of 83.21% on the held-out test set.

Its weighted F1-score was 82.88%.

The model provided a useful baseline against which the other machine-learning algorithms could be compared.

---

## 2.6 Random Forest

Random Forest is an ensemble machine-learning algorithm that combines multiple decision trees.

Each tree contributes to the final prediction, and the ensemble can capture nonlinear relationships between input features.

Random Forest was included in this project to compare a tree-based approach with linear classification models.

The Random Forest model achieved approximately 80% test accuracy in the experiments.

Although it performed reasonably well, it did not outperform the SVM model on the selected dataset.

---

## 2.7 Support Vector Machine

Support Vector Machine (SVM) is a supervised machine-learning algorithm commonly used for classification.

A linear SVM attempts to identify decision boundaries that separate different classes while maximizing the margin between classes.

SVM was particularly effective in this project.

The initial SVM achieved:

- Accuracy: 84.67%
- Precision: 86.05%
- Recall: 84.67%
- Weighted F1-score: 84.81%

After hyperparameter tuning, the Linear SVM with C=100 achieved substantially stronger performance on the held-out test set.

---

## 2.8 Hyperparameter Optimization

Machine-learning algorithms contain parameters that influence their behavior.

For the SVM model, different combinations of:

- C
- Kernel
- Gamma

were evaluated using cross-validation.

A total of 16 parameter combinations were evaluated using 5-fold cross-validation, resulting in 80 individual model fits.

The best configuration was:

```text
Kernel = linear
C = 100
Gamma = scale