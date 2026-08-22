# Chapter 1: Introduction

## 1.1 Background

Depression is one of the most common mental health conditions and can significantly affect an individual's emotional well-being, academic performance, social relationships, and daily activities.

University students may experience psychological pressure due to academic workload, financial difficulties, sleep problems, social challenges, and other personal factors.

Early identification of depressive symptoms can help individuals recognize potential problems and seek appropriate support.

Traditional depression screening methods generally rely on standardized questionnaires such as the Patient Health Questionnaire-9 (PHQ-9). The PHQ-9 consists of nine questions that evaluate common depressive symptoms experienced during the previous two weeks.

This project explores the use of machine learning to analyze PHQ-9 responses together with additional student-related information and estimate a depression-severity category.

---

## 1.2 Problem Statement

Depressive symptoms among students may remain unnoticed because individuals may not recognize the symptoms or may hesitate to seek professional assistance.

Although standardized screening questionnaires such as PHQ-9 are useful, there is an opportunity to investigate how machine-learning techniques can be applied to questionnaire responses for automated research-based screening.

Therefore, this project proposes DepressionRiskAI, a machine-learning-based research prototype that analyzes PHQ-9 responses and additional information to estimate one of five depression-severity categories:

- Minimal
- Mild
- Moderate
- Moderately severe
- Severe

The system is designed as an educational and research screening tool and does not provide a medical diagnosis.

---

## 1.3 Aim of the Project

The main aim of this project is to develop an AI-powered depression-risk screening prototype using machine-learning techniques and PHQ-9 questionnaire responses.

The project investigates different machine-learning algorithms and selects an appropriate model based on experimental evaluation.

---

## 1.4 Objectives

The main objectives of the project are:

1. To study the use of PHQ-9 responses for depression-severity screening.

2. To analyze a research dataset containing student depression-related information.

3. To preprocess questionnaire and demographic data for machine-learning experiments.

4. To implement multiple machine-learning classification algorithms.

5. To compare Logistic Regression, Random Forest, and Support Vector Machine models.

6. To perform 5-fold cross-validation to evaluate model stability.

7. To perform hyperparameter tuning for the best-performing model.

8. To evaluate the final model using accuracy, precision, recall, F1-score, and confusion matrix.

9. To investigate important model features using explainable machine-learning techniques.

10. To develop a user-friendly Streamlit web application.

11. To compare the standard PHQ-9 severity category with the machine-learning prediction.

12. To provide a research prototype that can demonstrate the practical application of artificial intelligence in mental-health screening research.

---

## 1.5 Research Questions

This project investigates the following research questions:

### RQ1

Can machine-learning algorithms classify depression-severity categories using PHQ-9 responses and additional student-related information?

### RQ2

Which machine-learning algorithm performs best for the selected research dataset?

### RQ3

Does hyperparameter tuning improve the performance of the selected machine-learning model?

### RQ4

How stable is the selected model when evaluated using 5-fold cross-validation?

### RQ5

Which questionnaire and contextual features have the greatest influence on the model's predictions?

---

## 1.6 Scope of the Project

The scope of DepressionRiskAI includes:

- PHQ-9 questionnaire-based screening.
- Five depression-severity categories.
- Student-related demographic and contextual information.
- Machine-learning classification.
- Model comparison.
- Cross-validation.
- Hyperparameter tuning.
- Explainable AI analysis.
- Interactive web-based demonstration.

The system does not replace psychologists, psychiatrists, doctors, or other qualified healthcare professionals.

The system is intended for educational and research purposes.

---

## 1.7 Dataset

The project uses a PHQ-9 Student Depression Dataset containing:

- 682 records
- 16 columns
- 5 depression-severity classes

The severity classes are:

| Severity | Records |
|---|---:|
| Minimal | 206 |
| Mild | 155 |
| Moderate | 128 |
| Moderately severe | 125 |
| Severe | 68 |

The dataset contains PHQ-9 questionnaire responses as well as additional information such as:

- Age
- Gender
- Sleep Quality
- Study Pressure
- Financial Pressure

---

## 1.8 Machine Learning Models

Three classification algorithms were investigated:

### Logistic Regression

Logistic Regression was implemented as a baseline classification model.

Its test performance was:

- Accuracy: 83.21%
- Precision: 83.13%
- Recall: 83.21%
- Weighted F1-score: 82.88%

### Random Forest

Random Forest was evaluated as a tree-based ensemble classification method.

Its test performance was approximately:

- Accuracy: 80%
- Precision: 80%
- Recall: 80%
- Weighted F1-score: 79%

### Support Vector Machine

Support Vector Machine achieved the best initial performance among the evaluated models.

The initial SVM achieved:

- Accuracy: 84.67%
- Precision: 86.05%
- Recall: 84.67%
- Weighted F1-score: 84.81%

After hyperparameter tuning, the selected Linear SVM configuration used:

- Kernel: Linear
- C: 100
- Gamma: scale

The tuned model achieved:

- Accuracy: 91.97%
- Precision: 92.18%
- Recall: 91.97%
- Weighted F1-score: 91.92%
- Macro F1-score: 91.12%

---

## 1.9 Cross-Validation

Five-fold cross-validation was performed to evaluate the stability of the machine-learning models.

The cross-validation results showed that SVM achieved the strongest performance:

| Model | Accuracy | Weighted F1 | Macro F1 |
|---|---:|---:|---:|
| SVM | 86.51% | 86.52% | 85.90% |
| Logistic Regression | 82.70% | 82.29% | 80.49% |
| Random Forest | 81.53% | 81.20% | 80.10% |

These results supported the selection of SVM for further optimization.

---

## 1.10 Proposed System

The proposed DepressionRiskAI system follows the following workflow:

1. User opens the web application.
2. User provides demographic and contextual information.
3. User completes the nine PHQ-9 questions.
4. The system calculates the PHQ-9 score.
5. The PHQ-9 score is converted into a standard severity category.
6. The same information is provided to the trained machine-learning model.
7. The model predicts a depression-severity category.
8. The system calculates prediction probabilities.
9. The application displays the PHQ-9 result and AI prediction.
10. The system displays a probability distribution.
11. The results are presented with research-oriented guidance.

---

## 1.11 Ethical Considerations

Mental-health applications require careful consideration because incorrect predictions may have serious consequences.

Therefore, DepressionRiskAI is explicitly presented as a research and educational screening prototype.

The system does not claim to diagnose depression.

Users should not make medical decisions solely based on the model prediction.

The application should be evaluated by qualified researchers and healthcare professionals before any real-world clinical deployment.

---

## 1.12 Limitations

The current prototype has several limitations.

First, the dataset contains only 682 records, which limits the amount of training data available to the machine-learning algorithms.

Second, the dataset represents a specific student population and may not generalize to all populations.

Third, the model predicts categories based on patterns learned from the available dataset and may therefore produce incorrect predictions.

Fourth, the system has not undergone clinical validation.

Finally, the application should not be considered a replacement for professional mental-health assessment.

---

## 1.13 Significance of the Project

This project demonstrates how artificial intelligence and machine learning can be integrated with a standardized mental-health screening questionnaire.

The project provides practical experience in:

- Data preprocessing
- Exploratory data analysis
- Machine learning
- Classification
- Model evaluation
- Hyperparameter optimization
- Cross-validation
- Explainable AI
- Streamlit application development
- Research methodology

The resulting prototype can also serve as a foundation for future research involving larger datasets, additional features, improved validation methods, and expert clinical evaluation.

---

## 1.14 Chapter Summary

This chapter introduced DepressionRiskAI, an AI-powered research prototype for depression-severity screening using PHQ-9 responses and additional student-related information.

The project investigates multiple machine-learning algorithms and identifies a tuned Linear SVM as the strongest-performing model in the current experiments.

The following chapters will discuss the existing literature, research methodology, system design, implementation, experimental results, discussion, limitations, and conclusions.