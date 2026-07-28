# Credit Risk Model for Buy-Now-Pay-Later (BNPL)

[![CI](https://github.com/semegn19/credit-risk-model/actions/workflows/ci.yml/badge.svg)](https://github.com/semegn19/credit-risk-model/actions/workflows/ci.yml)

An end-to-end Machine Learning and MLOps project that predicts customer credit risk for Bati Bank's Buy-Now-Pay-Later (BNPL) product using behavioural transaction data. The project covers data engineering, feature engineering, proxy target creation, model training, experiment tracking with MLflow, REST API deployment using FastAPI, an interactive Streamlit dashboard, and SHAP explainability.

---

# Project Overview

Traditional credit scoring relies heavily on historical lending behaviour. For many financially excluded customers, these labels are unavailable. This project addresses that challenge by constructing a behavioural proxy target from customer transaction history using RFM (Recency, Frequency and Monetary) analysis and customer clustering.

The resulting model estimates the probability that a customer belongs to a high-risk behavioural segment, enabling risk assessment for BNPL lending while maintaining transparency through explainable AI techniques.

---

# Project Architecture

```
Raw Transaction Data
        │
        ▼
Feature Engineering
        │
        ▼
Proxy Target Engineering (RFM + KMeans)
        │
        ▼
Processed Dataset
        │
        ▼
Model Training
(Logistic Regression + Random Forest)
        │
        ▼
MLflow Experiment Tracking
        │
        ▼
Model Registry
        │
        ▼
FastAPI Prediction Service
        │
        ▼
Streamlit Dashboard
        │
        ▼
SHAP Explainability
```

---

# Repository Structure

```
credit-risk-model/

├── data/
│   ├── raw/
│   └── processed/
│
├── dashboard/
│   ├── pages/
│   │   ├── 1_Credit_Assessment.py
│   │   ├── 2_Model_Performance.py
│   │   ├── 3_Portfolio_Analytics.py
│   │   └── 4_Model_Explainability.py
│   ├── api_client.py
│   ├── mlflow_utils.py
│   ├── shap_utils.py
│   └── app.py
│
├── src/
│   ├── api/
│   ├── config/
│   ├── features/
│   ├── pipelines/
│   └── training/
│
├── tests/
├── mlruns/
├── requirements.txt
└── README.md
```

---

# Features

The project includes:

- Behavioural feature engineering
- RFM-based proxy target creation
- Customer clustering using KMeans
- Logistic Regression baseline
- Random Forest classifier
- Hyperparameter tuning using GridSearchCV
- MLflow experiment tracking and model registry
- FastAPI prediction API
- Streamlit dashboard
- SHAP global and local explainability
- Automated testing using PyTest
- Continuous Integration using GitHub Actions

---

# Model Performance

The best-performing model is a Random Forest classifier.

| Metric | Score |
|---------|-------|
| Accuracy | *(Automatically tracked in MLflow)* |
| Precision | *(Automatically tracked in MLflow)* |
| Recall | *(Automatically tracked in MLflow)* |
| F1-score | *(Automatically tracked in MLflow)* |
| ROC-AUC | **0.856** |

The dashboard retrieves the latest evaluation metrics directly from MLflow, ensuring that displayed values remain synchronized with the most recent registered model.

---

# Streamlit Dashboard

The Streamlit application provides an interactive interface for model exploration and credit risk assessment.

The dashboard contains four pages:

### Credit Assessment

- Customer data entry
- Real-time prediction
- Risk probability
- High-risk / Low-risk classification

---

### Model Performance

- Live MLflow metrics
- ROC-AUC
- Precision
- Recall
- F1-score
- Accuracy

---

### Portfolio Analytics

- Portfolio overview
- Transaction summaries
- Customer statistics
- Risk distribution
- Business KPIs

---

### Model Explainability

- SHAP Summary Plot
- SHAP Waterfall Plot
- Global Feature Importance
- Individual Prediction Explanation

---

# API

The project exposes a FastAPI endpoint.

### Prediction Endpoint

```
POST /predict
```

Input:

```json
{
  "num__Total_Transaction_Amount": 5000,
  "num__Average_Transaction_Amount": 500,
  ...
}
```

Output:

```json
{
  "risk_probability": 0.27,
  "risk_label": 0
}
```

---

# Installation

Clone the repository.

```bash
git clone https://github.com/semegn19/credit-risk-model.git

cd credit-risk-model
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

# Run the Preprocessing Pipeline

```bash
python -m src.pipelines.preprocess
```

---

# Train the Models

```bash
python -m src.training.train
```

---

# Launch the FastAPI Server

```bash
uvicorn src.api.main:app --reload
```

Swagger documentation:

```
http://127.0.0.1:8000/docs
```

---

# Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

---

# Running Tests

Run all unit tests.

```bash
pytest
```

Run with coverage.

```bash
pytest --cov=src --cov-report=term-missing
```

---

# Continuous Integration

The repository uses GitHub Actions to automatically:

- Install dependencies
- Run Flake8 linting
- Execute unit tests
- Verify project integrity on every push and pull request

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- MLflow
- FastAPI
- Streamlit
- SHAP
- PyTest
- GitHub Actions

---

# Business Context

This project was developed for Bati Bank to support Buy-Now-Pay-Later (BNPL) lending using behavioural transaction data.

Because historical default labels were unavailable, a proxy target was engineered using RFM analysis and KMeans clustering. The resulting machine learning pipeline predicts behavioural credit risk while maintaining transparency through SHAP explainability, supporting responsible and interpretable lending decisions.
