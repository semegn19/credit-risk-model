import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Portfolio Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Credit Portfolio Analytics")

st.write(
    """
    This page provides an overview of the customer portfolio used to train
    the credit risk model. It helps business stakeholders understand customer
    behaviour, transaction activity, and the distribution of predicted risk.
    """
)

# -------------------------
# Load processed dataset
# -------------------------

df = pd.read_csv("data/processed/processed_data_with_target.csv")

# -------------------------
# KPI Cards
# -------------------------

total_customers = len(df)

high_risk = int(df["is_high_risk"].sum())

low_risk = total_customers - high_risk

risk_rate = high_risk / total_customers * 100

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Customers",
    f"{total_customers:,}"
)

c2.metric(
    "High Risk",
    f"{high_risk:,}"
)

c3.metric(
    "Low Risk",
    f"{low_risk:,}"
)

c4.metric(
    "High Risk %",
    f"{risk_rate:.2f}%"
)

st.divider()

# -------------------------
# Risk Distribution
# -------------------------

left, right = st.columns(2)

with left:

    fig = px.pie(
        values=[low_risk, high_risk],
        names=["Low Risk", "High Risk"],
        title="Portfolio Risk Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with right:

    fig = px.histogram(
        df,
        x="num__Transaction_Count",
        nbins=30,
        title="Transaction Count Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.divider()

# -------------------------
# Feature Distributions
# -------------------------

left, right = st.columns(2)

with left:

    fig = px.box(
        df,
        y="num__Total_Transaction_Value",
        color="is_high_risk",
        title="Transaction Value by Risk Group"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with right:

    fig = px.scatter(
        df,
        x="num__Transaction_Count",
        y="num__Total_Transaction_Value",
        color="is_high_risk",
        title="Transactions vs Value"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.divider()

st.subheader("Dataset Preview")

st.dataframe(df.head(20), use_container_width=True)