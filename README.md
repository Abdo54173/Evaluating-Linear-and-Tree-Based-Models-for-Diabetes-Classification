# Comparative Analysis of Machine Learning Models for Diabetes Prediction

This project implements a complete **end-to-end machine learning pipeline** for predicting diabetes using clinical and lifestyle data.  
It includes **EDA, preprocessing, handling imbalanced data, training multiple ML models, model comparison, threshold tuning, and selecting the best final model**.

The work is organized into two Jupyter Notebooks:

- **01_EDA + Preprocessing.ipynb** → Data understanding, cleaning, encoding, scaling, and exporting processed datasets  
- **02_Training.ipynb** → Model training, evaluation, comparison, tuning, and final model selection

- 
---

# 📌 Objective

To build and evaluate multiple machine learning models for **early diabetes prediction**, compare their performance, and optimize the best-performing model using class-weight adjustments and probability threshold tuning.

---

# 📊 Dataset Overview

- Total samples: ~100,000  
- Target variable: `diabetes` (0 = Non-diabetic, 1 = Diabetic)
- Key features used:
  - `age`
  - `gender`
  - `hypertension`
  - `heart_disease`
  - `smoking_history`
  - `bmi`
  - `HbA1c_level`
  - `blood_glucose_level`

---

# 🔍 Notebook 1 — EDA & Preprocessing

### ✔ Exploratory Data Analysis (EDA)  
- Summary statistics  
- Distribution plots for all numerical features  
- Boxplots for outlier inspection  
- Class imbalance visualization  
- Correlation inspection  

### ✔ Preprocessing Steps  
#### **1. Handling categorical features**
- `gender`: Rare category `"Other"` (18 samples) removed  
- `smoking_history`: Multiple categories kept as-is (retain medical meaning)

#### **2. Creating two data versions**
- **Tree-based version (`data_preprocessed_tree.csv`)**
  - Label Encoding for categories  
  - No scaling  

- **Scaled version (`data_preprocessed_scaled.csv`)**
  - One-Hot Encoding  
  - StandardScaler applied to numeric columns  
  - Used for Logistic Regression, SVM, etc.

#### **3. Outliers**
- Medical outliers (e.g., BMI > 60, glucose up to 300) kept intentionally  
  → They represent real conditions, not noise.

#### **4. Splitting**
Used stratified train–test split to preserve class distribution:


---

# 🤖 Notebook 2 — Model Training & Evaluation

The following machine learning models were trained and compared:

### **Linear Models (Scaled Data)**
- Logistic Regression  
- Linear SVM  
- SVM (RBF kernel)

### **Tree-Based Models**
- Decision Tree  
- Random Forest  
- XGBoost (Baseline)  
- XGBoost (Tuned + Threshold Adjustment)

---

# ⚠ Challenges Encountered

### **1. Severe class imbalance**
- Majority class ≈ 92% non-diabetic
- Minority class ≈ 8% diabetic

Impact:
- High accuracy can be misleading  
- Linear models tended to predict many false positives (high recall, low precision)

Solutions used:
- `class_weight='balanced'`
- `scale_pos_weight` for XGBoost  
- Probability threshold tuning  
- Stratified splitting  

---

### **2. Non-linear relationships**
Linear models (Logistic, Linear SVM) performed poorly:

- **F1 Score ≈ 0.57–0.59**
- Very high recall but extremely low precision

Reason:  
The dataset contains **non-linear medical relationships** → tree-based models fit them better.

---

### **3. Small rare categories (e.g., “Other” in gender)**
- Only 18 samples  
- Removed to avoid noise and unstable encoding

---

# 📈 Model Performance Summary (Class 1 – Diabetic)

| Model                     | Precision | Recall | F1 Score |
|---------------------------|-----------|--------|----------|
| Logistic Regression       | 0.43      | 0.89   | 0.58     |
| Linear SVM                | 0.42      | 0.90   | 0.57     |
| SVM (RBF)                 | 0.44      | 0.92   | 0.59     |
| Decision Tree             | 0.71      | 0.74   | 0.72     |
| Random Forest             | 0.94      | 0.69   | 0.80     |
| **XGBoost (baseline)**    | 0.85      | 0.75   | 0.80     |
| **XGBoost (threshold 0.45)** | 0.85   | 0.75   | 0.80     |

---

# 🏆 Final Best Model

### ⭐ **XGBoost with `scale_pos_weight = 2` and threshold = 0.45**

Why this model?

- Best balance between **precision** and **recall**  
- High **F1 score (0.80)**  
- High **macro average**  
- Robust to class imbalance  
- Stable performance across different thresholds  

---
