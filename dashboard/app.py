import streamlit as st

st.set_page_config(
    page_title="Bati Bank Credit Risk Dashboard",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Bati Bank Credit Risk Decision Support System")

st.markdown("""
Welcome to the **Bati Bank Buy-Now-Pay-Later Credit Risk Dashboard**.

This application demonstrates a complete machine learning pipeline for
predicting customer credit risk using transaction behaviour from the
Xente dataset.

Use the navigation menu on the left to explore the system.
""")

st.divider()

col1, col2, col3 = st.columns(3)

col1.metric(
    "Model",
    "Random Forest"
)

col2.metric(
    "ROC-AUC",
    "0.856"
)

col3.metric(
    "API",
    "Online ✅"
)

st.subheader("Available Pages")

st.info("""
**Credit Assessment**

Predict the risk of a new customer.

**Model Performance**

View evaluation metrics and business interpretation.

**Portfolio Analytics**
        
Overview of the customer portfolio used to train the credit risk model.

**Model Explainability (SHAP)**
View how each feature contributes to the predicted credit risk.
""")