[README.md](https://github.com/user-attachments/files/31699917/README.md)
# DepressionRiskAI

**AI-Based Depression Risk Screening System**

DepressionRiskAI is a machine learning research prototype that combines the PHQ-9 questionnaire with a trained classifier to estimate depression-severity category. It takes the nine standard PHQ-9 responses along with basic contextual information (age, gender, sleep quality, study pressure, financial pressure) and predicts one of five severity classes.

> **Note:** This is an educational and research screening prototype, not a medical diagnostic tool. It is not a substitute for a professional mental health assessment.

---

## Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [PHQ-9 Features](#phq-9-features)
- [Machine Learning Models](#machine-learning-models)
- [Model Comparison](#model-comparison)
- [Cross-Validation](#cross-validation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Final Model](#final-model)
- [Explainability](#explainability)
- [Web Application](#web-application)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [Running the Experiments](#running-the-experiments)
- [Privacy](#privacy)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Project Info](#project-info)
- [License](#license)

---

## Overview

Depression is a significant mental health concern that can affect academic performance, relationships, and overall quality of life. The **Patient Health Questionnaire-9 (PHQ-9)** is a widely used, standardized tool for assessing depressive symptoms.

This project explores whether machine learning can classify PHQ-9 responses into five depression-severity categories — Minimal, Mild, Moderate, Moderately Severe, and Severe — with comparable or complementary results to standard PHQ-9 scoring.

Three algorithms were evaluated: Logistic Regression, Random Forest, and Support Vector Machine (SVM). After model comparison, cross-validation, and hyperparameter tuning, a **Linear SVM (C = 100)** was selected as the final model.

## Objectives

1. Build an AI-based depression-risk screening prototype.
2. Process PHQ-9 responses using machine learning.
3. Compare multiple classification algorithms.
4. Evaluate models using accuracy, precision, recall, and F1-score.
5. Apply cross-validation to assess model stability.
6. Tune hyperparameters to improve SVM performance.
7. Investigate feature importance for interpretability.
8. Build an interactive Streamlit web application.
9. Present predictions in a clear, research-oriented interface.
10. Demonstrate how ML can support questionnaire-based screening research.

## System Architecture

```
User
  │
  ▼
PHQ-9 Questionnaire (9 items) + Context (age, gender, sleep, study & financial pressure)
  │
  ▼
Data Preprocessing (encoding / transformation)
  │
  ▼
Linear SVM (C = 100)
  │
  ▼
Depression Severity Classification
  │
  ▼
Streamlit App
  • PHQ-9 score & category
  • AI prediction & confidence
  • Probability distribution
  • PHQ-9 vs. AI comparison
```

## Dataset

| Property | Value |
|---|---:|
| Total records | 682 |
| Total columns | 16 |
| Severity classes | 5 |
| Missing values | 0 |
| Age range | 17–26 |
| Gender categories | 2 |

**Severity distribution**

| Severity | Records | Percentage |
|---|---:|---:|
| Minimal | 206 | 30.21% |
| Mild | 155 | 22.73% |
| Moderate | 128 | 18.77% |
| Moderately Severe | 125 | 18.33% |
| Severe | 68 | 9.97% |

## PHQ-9 Features

The model uses the nine standard PHQ-9 symptom questions:

1. Little interest or pleasure in doing things
2. Feeling down, depressed, or hopeless
3. Trouble falling or staying asleep, or sleeping too much
4. Feeling tired or having little energy
5. Poor appetite or overeating
6. Feeling bad about yourself, or feeling like a failure
7. Trouble concentrating
8. Moving or speaking unusually slowly, or being unusually restless
9. Thoughts of being better off dead or of self-harm

Each item is scored on a 4-point scale:

| Response | Score |
|---|---:|
| Not at all | 0 |
| Several days | 1 |
| More than half the days | 2 |
| Nearly every day | 3 |

Total PHQ-9 score ranges from **0–27** and maps to severity as follows:

| PHQ-9 Score | Severity |
|---:|---|
| 0–4 | Minimal |
| 5–9 | Mild |
| 10–14 | Moderate |
| 15–19 | Moderately Severe |
| 20–27 | Severe |

## Machine Learning Models

- **Logistic Regression** — used as a baseline linear classifier.
- **Random Forest** — evaluated as a nonlinear, ensemble-based approach.
- **Support Vector Machine** — evaluated for its strength on high-dimensional, questionnaire-encoded feature spaces.

## Model Comparison

Initial results before tuning:

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 83.21% | 83.13% | 83.21% | 82.88% |
| Random Forest | 80.29% | 80.82% | 80.29% | 80.33% |
| SVM | 84.67% | 86.05% | 84.67% | 84.81% |

SVM had the strongest overall performance out of the box.

## Cross-Validation

A 5-fold cross-validation was run to check consistency across splits:

| Model | Mean Accuracy | Accuracy Std | Mean F1 | Macro F1 |
|---|---:|---:|---:|---:|
| SVM | **86.51%** | 2.78% | **86.52%** | **85.90%** |
| Logistic Regression | 82.70% | 2.62% | 82.29% | 80.49% |
| Random Forest | 81.53% | 3.10% | 81.20% | 80.10% |

SVM remained the most consistent and highest-performing model.

## Hyperparameter Tuning

The SVM was tuned via grid search. Best configuration:

| Parameter | Value |
|---|---|
| Kernel | Linear |
| C | 100 |
| Gamma | scale |

Best tuning Macro-F1: **0.8901**

## Final Model

**Linear Support Vector Machine**

| Parameter | Value |
|---|---|
| Algorithm | SVM |
| Kernel | Linear |
| C | 100 |
| Gamma | scale |
| Classes | 5 |

**Test set performance** (137 held-out samples):

| Metric | Score |
|---|---:|
| Accuracy | 91.97% |
| Precision | 92.18% |
| Recall | 91.97% |
| Weighted F1 | 91.92% |
| Macro F1 | 91.12% |

**Per-class performance**

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Mild | 0.96 | 0.84 | 0.90 |
| Minimal | 0.95 | 1.00 | 0.98 |
| Moderate | 0.86 | 0.92 | 0.89 |
| Moderately Severe | 0.92 | 0.88 | 0.90 |
| Severe | 0.87 | 0.93 | 0.90 |

**Confusion matrix**

```
[[26,  2,  3,  0,  0],
 [ 0, 41,  0,  0,  0],
 [ 1,  0, 24,  1,  0],
 [ 0,  0,  1, 22,  2],
 [ 0,  0,  0,  1, 13]]
```

## Explainability

A feature-importance analysis was performed on the final Linear SVM to identify which encoded questionnaire features most strongly influence each severity class, including:

- Feeling bad about yourself
- Poor appetite or overeating
- Trouble concentrating
- Feeling down or hopeless
- Sleep difficulties
- Low energy
- Psychomotor changes
- Self-harm-related response

This analysis is intended to support interpretability and research transparency rather than clinical explanation.

## Web Application

An interactive **Streamlit** app lets users:

- Enter demographic and contextual information
- Answer all nine PHQ-9 questions
- View the calculated PHQ-9 score and severity category
- Get the AI-predicted severity, confidence, and probability distribution
- Compare the PHQ-9 result against the AI prediction

## Project Structure

```
DepressionRiskAI/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── PHQ-9 Student Depression Dataset/
│       └── PHQ-9_Dataset_5th_Edition.csv
│
├── ml/
│   ├── analyze_dataset.py
│   ├── train_baseline.py
│   ├── train_random_forest.py
│   ├── train_svm.py
│   ├── model_comparison.py
│   ├── cross_validation.py
│   ├── tune_svm.py
│   ├── evaluate_final_model.py
│   ├── explain_model.py
│   ├── model_results.csv
│   ├── cross_validation_results.csv
│   ├── svm_tuning_results.csv
│   ├── feature_importance.csv
│   └── final_model.joblib
│
└── pages/
    └── ...
```

> The contents of `pages/` may grow as more Streamlit views are added.

**Tech stack:** Python, scikit-learn, Pandas, NumPy, Matplotlib, Streamlit, Joblib.

## Privacy

This is a research demonstration. Avoid entering personally identifiable information into any experimental deployment. The app is not a clinical record or diagnostic service.

## Limitations

- The dataset contains only 682 records and may not represent the general population.
- Predictions are based solely on questionnaire responses.
- Machine learning predictions can be wrong.
- This is not a clinical diagnostic system.
- Performance may differ on external, unseen datasets.
- The model needs external validation before any real-world or clinical use.
- Model confidence scores are not a measure of clinical certainty.

## Future Work

- Larger, more diverse datasets and external validation
- Additional ML and deep learning approaches
- SHAP-based explainability
- Better probability calibration
- Model monitoring in production
- Secure database integration and user authentication
- Anonymous assessment history
- Mobile app and REST API
- Cloud deployment
- Multilingual questionnaire support
- Professional mental-health referral resources
- Accessibility and UI/UX improvements

## Project Info

| | |
|---|---|
| **Title** | AI-Based Depression Risk Screening System |
| **Type** | Final Year Project / ML Research Prototype |
| **Final algorithm** | Linear SVM (C = 100) |
| **Dataset size** | 682 records |
| **Classes** | 5 |
| **Test accuracy** | 91.97% |
| **Author** | Muhammad Asif, BS Computer Science |

The reported 91.97% accuracy reflects performance on this project's held-out test set only — it is not a claim of clinical diagnostic accuracy. DepressionRiskAI is a machine learning research prototype for educational screening research.

## License

Intended for educational and research use. Before any commercial or clinical use, obtain appropriate licensing, dataset permissions, ethical review, privacy protections, and professional validation. Check the dataset's own redistribution terms before publishing it in a public repository.
