# 📘 Academic Burnout Detection and Performance Forecasting

## Overview
Academic burnout is a growing concern among students and often goes unnoticed until academic performance drops significantly. This project focuses on **early detection of academic burnout risk** using student learning behavior data and provides **actionable insights** to support timely interventions.

The system analyzes patterns such as quiz performance, assignment submission delays, attendance trends, and study habits to identify students who may be at risk of burnout. In addition, it estimates the potential short-term academic impact to support proactive decision-making.

---

## Problem Statement
Educational institutions typically rely on manual observation or delayed performance indicators to identify student burnout. By the time issues are detected, academic decline has often already occurred. There is a need for a **data-driven, early-warning system** that can analyze learning behavior trends over time and flag potential burnout risks before they become critical.

---

## Key Objectives
- Detect students at risk of academic burnout using behavioral data  
- Analyze trends rather than isolated values  
- Provide interpretable model explanations  
- Forecast short-term academic performance decline  
- Enable intervention-oriented insights  

---

## Dataset
- **Type:** Synthetic (generated for academic use)
- **Structure:**  
  - 400 students × 12 weeks (≈ 4800 records)
- **Key Features:**
  - Quiz scores and variability
  - Assignment submission delays
  - Attendance percentage
  - Study session frequency
  - Deadline workload
- **Target Variables:**
  - `burnout_risk` (Binary classification)
  - `quiz_score_next_week` (Regression)

Synthetic data was chosen to ensure privacy while maintaining realistic learning behavior patterns.

---

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Distribution analysis of learning behaviors
- Trend comparison between at-risk and normal students
- Temporal behavior visualization over weeks

### 2. Feature Engineering
To capture **direction and momentum**, several engineered features were created:
- Rolling averages (3-week window)
- Trend-based features (week-over-week changes)
- Performance stability (rolling standard deviation)

---

## Models Used

### 🔹 Burnout Risk Classification
- **Model:** Random Forest Classifier  
- **Output:** Probability of burnout risk  
- **Reason:** Robust to non-linearity and feature interactions  

### 🔹 Performance Forecasting
- **Model:** Linear Regression  
- **Output:** Predicted quiz score for the next week  
- **Purpose:** Estimate short-term academic impact (supporting insight)

---

## Model Explainability (SHAP)
To ensure transparency and trust:
- **Global SHAP Summary:** Shows which learning behaviors most influence burnout risk overall  
- **Local SHAP Explanation:** Explains why a specific student was flagged as high risk  

These explanations help educators understand *why* a model makes certain predictions, not just *what* it predicts.

---

## Risk Stratification & Intervention Logic
Based on predicted risk probability:
- **Low Risk**
- **Moderate Risk**
- **High Risk**

Each category is mapped to **recommended academic interventions**, such as study plan adjustments, mentoring support, or workload balancing.

---

## Project Structure
```
Academic Burnout Detection/
│
├── app.py                            <-- Premium Streamlit Dashboard UI
├── Acadamic_Burnout_Detection.ipynb
├── Dataset generation.ipynb
│
├── models/
│   ├── rf_clf.pkl
│   ├── lin_reg.pkl
│   ├── scaler.pkl
│   └── feature_cols.pkl
│
├── assets/
│   ├── shap_global_summary.png
│   └── shap_local_explanation.png
│
└── data/
    └── synthetic_student_burnout_data.csv
```

---

## Interactive Dashboard (Streamlit)
A full-stack premium interactive web application has been built using **Streamlit** to demonstrate the predictive modeling in real-time. Highlights include:
- **Radar Charts**: Visual breakdown of habit patterns.
- **Risk Gauge**: A smooth, color-coded speedometer indicating Burnout probability.
- **AI Analytics**: Explains *why* you received your risk score using contextual feature importance.

> Run the dashboard locally with:
> `streamlit run app.py`

---

## Deployment Plan
- Models are serialized using `pickle`
- Live Interactive Dashboard built using **Streamlit**
- Designed to integrate seamlessly with a backend API (FastAPI) when scaling out
- SHAP and Plotly visualizations natively render AI insight
 
---

## Tools & Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- SHAP
- Matplotlib, Seaborn
- FastAPI (planned deployment)

---

## Key Takeaway
This project demonstrates how learning behavior analytics can be transformed into a **practical early-warning system** for academic burnout. By combining predictive modeling with interpretability and intervention logic, the system emphasizes not just accuracy, but **responsible and actionable AI** in education.
