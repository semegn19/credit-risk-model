# Credit Risk Probability Model using Alternative Data

[![CI](https://github.com/semegn19/credit-risk-model/actions/workflows/ci.yml/badge.svg)](https://github.com/semegn19/credit-risk-model/actions/workflows/ci.yml)

## Project Overview

This project develops an end-to-end credit risk prediction system for **Bati Bank** to support Buy-Now-Pay-Later (BNPL) lending decisions using alternative transaction data provided by Xente. Since traditional credit histories are unavailable for many customers, the project constructs a proxy target using customer transaction behaviour through **Recency, Frequency, and Monetary (RFM)** analysis.

The system includes data preprocessing, feature engineering, proxy target generation, machine learning model training, experiment tracking with MLflow, a REST API built with FastAPI, automated testing, and continuous integration.

---

# Repository Structure

```text
credit-risk-model
│
├── data/
│   ├── raw/
│   └── processed/
│
├── dashboard/
│
├── notebooks/
│
├── src/
│   ├── api/
│   ├── config/
│   ├── features/
│   ├── pipelines/
│   ├── training/
│   └── utils/
│
├── tests/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── mlruns/
├── requirements.txt
└── README.md
```

---

# Project Workflow

The project follows the complete machine learning lifecycle:

```
Raw Transaction Data
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Feature Engineering
        │
        ▼
RFM Target Engineering
        │
        ▼
Preprocessing Pipeline
        │
        ▼
Model Training
        │
        ▼
MLflow Experiment Tracking
        │
        ▼
FastAPI Deployment
        │
        ▼
Prediction API
```

---

# Features

The current implementation includes:

- Exploratory Data Analysis
- Customer-level feature engineering
- Temporal feature extraction
- Automatic preprocessing pipeline
- RFM-based proxy target engineering
- Customer clustering using K-Means
- Logistic Regression model
- Random Forest model
- Hyperparameter optimisation using GridSearchCV
- MLflow experiment tracking
- MLflow Model Registry
- FastAPI prediction service
- Automated unit testing
- GitHub Actions Continuous Integration

---

# Installation

Clone the repository

```bash
git clone https://github.com/semegn19/credit-risk-model.git

cd credit-risk-model
```

Create a virtual environment

```bash
python -m venv .venv
```

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Data Preprocessing

Generate the processed dataset

```bash
python -m src.pipelines.preprocess
```

This pipeline performs

- Customer aggregation
- Feature engineering
- Missing value handling
- Standardisation
- One-hot encoding
- RFM calculation
- Customer clustering
- Proxy target generation

The processed dataset is saved to

```
data/processed/processed_data_with_target.csv
```

---

# Model Training

Train the models

```bash
python -m src.training.train
```

The training pipeline

- Splits train/test data
- Performs hyperparameter tuning
- Evaluates models
- Logs metrics to MLflow
- Registers the best model

Current models include

- Logistic Regression
- Random Forest

Evaluation metrics include

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

# Experiment Tracking

Start the MLflow UI

```bash
mlflow ui
```

Open

```
http://127.0.0.1:5000
```

MLflow records

- parameters
- metrics
- trained models
- model registry versions

---

# REST API

Start the API

```bash
uvicorn src.api.main:app --reload
```

Swagger documentation

```
http://127.0.0.1:8000/docs
```

Prediction endpoint

```
POST /predict
```

Example request

```json
{
  "num__Total_Transaction_Amount": 5000,
  "num__Average_Transaction_Amount": 500,
  "num__Transaction_Count": 15,
  "num__Std_Transaction_Amount": 100,
  "num__Max_Transaction_Amount": 1000,
  "num__Min_Transaction_Amount": 50,
  "num__Total_Transaction_Value": 6000,
  "num__CountryCode": 256,
  "num__PricingStrategy": 2,

  "cat__CurrencyCode_UGX": 1,

  "cat__ProviderId_ProviderId_1": 0,
  "cat__ProviderId_ProviderId_2": 1,
  "cat__ProviderId_ProviderId_3": 0,
  "cat__ProviderId_ProviderId_4": 0,
  "cat__ProviderId_ProviderId_5": 0,
  "cat__ProviderId_ProviderId_6": 0,

  "cat__ProductCategory_airtime": 0,
  "cat__ProductCategory_data_bundles": 0,
  "cat__ProductCategory_financial_services": 1,
  "cat__ProductCategory_movies": 0,
  "cat__ProductCategory_other": 0,
  "cat__ProductCategory_ticket": 0,
  "cat__ProductCategory_transport": 0,
  "cat__ProductCategory_tv": 0,
  "cat__ProductCategory_utility_bill": 0,

  "cat__ChannelId_ChannelId_1": 1,
  "cat__ChannelId_ChannelId_2": 0,
  "cat__ChannelId_ChannelId_3": 0,
  "cat__ChannelId_ChannelId_5": 0
}
```

Example response

```json
{
  "risk_probability": 0.81,
  "risk_label": 1
}
```

---

# Testing

The repository includes automated unit tests covering

- Feature engineering
- Target engineering
- Data transformers
- API functionality

Run tests

```bash
pytest
```

Generate a coverage report

```bash
pytest --cov=src --cov-report=term-missing
```

---

# Continuous Integration

Every push and pull request automatically executes

- dependency installation
- code linting
- unit testing

using **GitHub Actions**.

The current build status is shown at the top of this README.

---

# Credit Scoring Business Understanding

## Basel II and the Importance of Interpretability

The Basel II Accord establishes an international framework for risk-sensitive banking regulation. Under the Internal Ratings-Based (IRB) approach, financial institutions are responsible for estimating the Probability of Default (PD) of borrowers while demonstrating that their models are transparent, reliable, and well governed.

This creates several important requirements for machine learning models used in credit scoring:

- Models must be explainable to regulators.
- Credit decisions must be auditable.
- Institutions must actively manage model risk.
- Lending decisions should provide understandable reasons for approval or rejection.

## Proxy Target Engineering

Traditional credit histories are unavailable for many BNPL customers. Consequently, this project constructs a behavioural proxy for default risk using customer transaction patterns.

Customers are segmented using **Recency, Frequency, and Monetary (RFM)** metrics before clustering them with K-Means. The least engaged customer cluster is labelled as high risk and used as the binary target variable for supervised learning.

## Model Trade-offs

| Aspect | Logistic Regression | Random Forest |
|----------|--------------------|---------------|
| Interpretability | High | Moderate |
| Predictive Performance | Moderate | High |
| Regulatory Transparency | Excellent | Requires Explainability |
| Risk of Overfitting | Low | Moderate |

This project evaluates both approaches to balance predictive performance with regulatory transparency.

---

# Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- MLflow
- Pytest
- GitHub Actions

---

# Current Status

Completed

- Exploratory Data Analysis
- Feature Engineering
- Target Engineering
- Machine Learning Pipeline
- MLflow Tracking
- FastAPI Deployment
- Automated Testing
- Continuous Integration

In Progress

- Streamlit Dashboard
- SHAP Explainability


Addis Ababa Unive
Software Engineering
