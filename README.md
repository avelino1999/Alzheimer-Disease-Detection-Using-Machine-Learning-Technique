# 🧠 Alzheimer's Disease Risk Analysis

> **Predicting Alzheimer's diagnosis using machine learning on clinical, lifestyle, and demographic data — with 94% classification accuracy.**

---

## 📌 Project Overview

This project applies end-to-end machine learning to a dataset of **2,000+ patient records** to identify the most significant risk factors for Alzheimer's disease. The goal is to support **early detection research** by building interpretable, high-performing classification models.

The work spans four interconnected notebooks — from raw data exploration all the way to hyperparameter-tuned model deployment.

---

## 🗂️ Repository Structure

| Notebook | Description |
|---|---|
| `EDA.ipynb` | Exploratory Data Analysis — distributions, correlations, and visual profiling of all features |
| `Project.ipynb` | Extended EDA with deeper visualisation of behavioural, cognitive, and clinical risk factors |
| `ML_Model.ipynb` | Multi-model training and evaluation — KNN, Naïve Bayes, Logistic Regression, SVM, Decision Tree |
| `Gradient_Boosting_hyper_tuning.ipynb` | Gradient Boosting with GridSearchCV hyperparameter optimisation |

---

## 📊 Dataset

- **Source:** `alzheimers_disease_data.csv`
- **Records:** 2,000+ patient entries
- **Target Variable:** `Diagnosis` (binary — Disease Present / No Disease)

**Feature Categories:**

| Category | Features |
|---|---|
| Demographics | Age, Gender, Ethnicity, EducationLevel |
| Lifestyle | BMI, Smoking, AlcoholConsumption, PhysicalActivity, DietQuality, SleepQuality |
| Clinical | SystolicBP, DiastolicBP, CholesterolTotal, CholesterolLDL, CholesterolHDL, CholesterolTriglycerides |
| Cognitive & Functional | MMSE, FunctionalAssessment, ADL |
| Behavioural Symptoms | MemoryComplaints, BehavioralProblems, Forgetfulness, Disorientation, PersonalityChanges |

---

## 🔍 Exploratory Data Analysis

The EDA phase produces a comprehensive visual profile of the dataset, examining how each feature relates to Alzheimer's diagnosis:

- **Demographic breakdowns** — Age distribution vs. diagnosis, Gender, Education level
- **Lifestyle factors** — Smoking habits, BMI violin plots, Sleep quality box plots
- **Clinical markers** — Blood pressure and cholesterol pair plots by diagnosis group
- **Cognitive scores** — MMSE, Functional Assessment, and ADL box plots
- **Behavioural symptoms** — Forgetfulness, Disorientation, and Personality Changes as percentage heatmaps

---

## 🤖 Machine Learning Models

### Data Preprocessing

- Irrelevant columns (`PatientID`, `DoctorInCharge`) dropped
- 60 / 20 / 20 train / validation / test split on a shuffled dataset
- **StandardScaler** applied for feature normalisation
- **RandomOverSampler** applied to the training set to address class imbalance

### Models Trained (`ML_Model.ipynb`)

| Model | Notes |
|---|---|
| K-Nearest Neighbours (KNN) | k=3, probabilistic predictions for ROC evaluation |
| Gaussian Naïve Bayes | Baseline probabilistic classifier |
| Logistic Regression | Linear decision boundary |
| Support Vector Machine (SVM) | RBF kernel with probability calibration |
| Decision Tree | Interpretable, unpruned baseline |

Every model is evaluated with:
- **Classification Report** (Precision, Recall, F1-Score)
- **Confusion Matrix**
- **ROC Curve** — overlaid Validation vs. Test AUC plots

### Gradient Boosting + Hyperparameter Tuning (`Gradient_Boosting_hyper_tuning.ipynb`)

A `GradientBoostingClassifier` was optimised using **5-fold cross-validated GridSearchCV** across the following parameter grid:

```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
}
```

The tuned model achieved **94% classification accuracy** on the test set.

---

## ✅ Key Results

- **Best Model:** Gradient Boosting (tuned) — **94% accuracy**
- **Top Predictive Features:** FunctionalAssessment, ADL, MMSE, MemoryComplaints, BehavioralProblems
- **Key Insight:** Cognitive and functional assessment scores are the strongest predictors of diagnosis, followed by lifestyle markers such as sleep quality and physical activity

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-lightgrey?logo=pandas)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualisation-blue)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualisation-teal)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-Oversampling-yellow)

```
pandas · numpy · matplotlib · seaborn
scikit-learn · imbalanced-learn
```

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/avelino1999/alzheimers-risk-analysis.git
   cd alzheimers-risk-analysis
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
   ```

3. **Add the dataset**  
   Place `alzheimers_disease_data.csv` in the root directory.

4. **Run notebooks in order**
   ```
   1. EDA.ipynb
   2. Project.ipynb
   3. ML_Model.ipynb
   4. Gradient_Boosting_hyper_tuning.ipynb
   ```

---

## 👤 Author

**Avelino Monteiro**  
MSc Data Science — Loughborough University  
📧 avelinomonteiro02@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/avelinomonteiro1999) · [GitHub](https://github.com/avelino1999)

---

*This project was developed as part of an MSc Data Science dissertation at Loughborough University, with a focus on applying machine learning to real-world clinical datasets for healthcare insight.*
