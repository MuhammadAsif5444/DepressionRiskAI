# Chapter 3: Methodology

## 3.1 Introduction

This chapter presents the methodology used to design, develop, train, evaluate, and deploy DepressionRiskAI.

The proposed system combines the Patient Health Questionnaire-9 (PHQ-9) with machine-learning classification techniques to estimate one of five depression-severity categories.

The overall methodology consists of the following stages:

1. Dataset acquisition
2. Dataset inspection
3. Data preprocessing
4. Feature selection
5. Dataset splitting
6. Baseline model development
7. Machine-learning model comparison
8. Cross-validation
9. Hyperparameter tuning
10. Final model evaluation
11. Explainable AI analysis
12. Web application development

---

# 3.2 Research Methodology

The project follows an experimental machine-learning methodology.

First, the research dataset was inspected to understand its structure, feature types, missing values, and class distribution.

The data was then prepared for machine-learning experiments.

Three classification algorithms were implemented:

- Logistic Regression
- Random Forest
- Support Vector Machine

The models were evaluated using a consistent experimental procedure.

The best-performing model was subsequently optimized through hyperparameter tuning and evaluated on a held-out test set.

Finally, the trained model was integrated into an interactive Streamlit web application.

---

# 3.3 System Workflow

The overall workflow of DepressionRiskAI can be represented as:

```text
Research Dataset
       │
       ▼
Dataset Inspection
       │
       ▼
Data Preprocessing
       │
       ▼
Feature Transformation
       │
       ▼
Train/Test Split
       │
       ├───────────────┐
       ▼               ▼
Logistic Regression  Random Forest
       │
       └───────┐
               ▼
             SVM
               │
               ▼
       Model Comparison
               │
               ▼
       5-Fold Cross-Validation
               │
               ▼
       SVM Hyperparameter Tuning
               │
               ▼
        Final Tuned SVM
               │
       ┌───────┴────────┐
       ▼                ▼
 Model Evaluation   Explainability
       │                │
       └───────┬────────┘
               ▼
       Streamlit Web App