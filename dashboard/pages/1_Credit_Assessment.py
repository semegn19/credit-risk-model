import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api_client import predict

st.set_page_config(layout="wide")

st.title("Credit Assessment")

st.caption(
    "Assess the predicted credit risk for a new BNPL applicant."
)

st.sidebar.header("Applicant Information")

# -------------------------
# Numeric Inputs
# -------------------------

amount = st.sidebar.number_input(
    "Total Transaction Amount",
    value=5000.0
)

avg_amount = st.sidebar.number_input(
    "Average Transaction Amount",
    value=500.0
)

transaction_count = st.sidebar.number_input(
    "Transaction Count",
    value=15.0
)

std_amount = st.sidebar.number_input(
    "Transaction Std Dev",
    value=100.0
)

max_amount = st.sidebar.number_input(
    "Maximum Transaction",
    value=1000.0
)

min_amount = st.sidebar.number_input(
    "Minimum Transaction",
    value=50.0
)

total_value = st.sidebar.number_input(
    "Total Transaction Value",
    value=6000.0
)

country = st.sidebar.number_input(
    "Country Code",
    value=256.0
)

pricing = st.sidebar.number_input(
    "Pricing Strategy",
    value=2.0
)

provider = st.sidebar.selectbox(
    "Provider",
    [1,2,3,4,5,6]
)

category = st.sidebar.selectbox(
    "Product Category",
    [
        "airtime",
        "data_bundles",
        "financial_services",
        "movies",
        "other",
        "ticket",
        "transport",
        "tv",
        "utility_bill"
    ]
)

channel = st.sidebar.selectbox(
    "Channel",
    [1,2,3,5]
)

payload = {

    "num__Total_Transaction_Amount": amount,
    "num__Average_Transaction_Amount": avg_amount,
    "num__Transaction_Count": transaction_count,
    "num__Std_Transaction_Amount": std_amount,
    "num__Max_Transaction_Amount": max_amount,
    "num__Min_Transaction_Amount": min_amount,
    "num__Total_Transaction_Value": total_value,
    "num__CountryCode": country,
    "num__PricingStrategy": pricing,

    "cat__CurrencyCode_UGX": 1,

    **{
        f"cat__ProviderId_ProviderId_{i}": float(provider==i)
        for i in range(1,7)
    },

    **{
        f"cat__ProductCategory_{c}": float(category==c)
        for c in [
            "airtime",
            "data_bundles",
            "financial_services",
            "movies",
            "other",
            "ticket",
            "transport",
            "tv",
            "utility_bill"
        ]
    },

    "cat__ChannelId_ChannelId_1": float(channel==1),
    "cat__ChannelId_ChannelId_2": float(channel==2),
    "cat__ChannelId_ChannelId_3": float(channel==3),
    "cat__ChannelId_ChannelId_5": float(channel==5),
}

left,right = st.columns([2,1])

summary = pd.DataFrame({

    "Feature":[
        "Transaction Amount",
        "Average Amount",
        "Transaction Count",
        "Provider",
        "Category",
        "Channel"
    ],

    "Value":[
        amount,
        avg_amount,
        transaction_count,
        provider,
        category,
        channel
    ]
})

left.subheader("Application Summary")
left.dataframe(summary,use_container_width=True,hide_index=True)

right.subheader("Model")

right.metric("Algorithm","Random Forest")
right.metric("ROC-AUC","0.856")

if st.button("Predict",use_container_width=True):

    result = predict(payload)

    risk = result["risk_probability"]

    c1,c2,c3 = st.columns(3)

    if risk < .30:

        decision="Approve"

        limit="UGX 20,000"

        duration="90 Days"

    elif risk < .60:

        decision="Manual Review"

        limit="UGX 10,000"

        duration="30 Days"

    else:

        decision="Reject"

        limit="N/A"

        duration="N/A"

    c1.metric("Risk",f"{risk:.2%}")
    c2.metric("Decision",decision)
    c3.metric("Credit Limit",limit)

    fig=go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk*100,
        number={"suffix":"%"},
        title={"text":"Risk Score"},
        gauge={
            "axis":{"range":[0,100]},
            "steps":[
                {"range":[0,30],"color":"green"},
                {"range":[30,60],"color":"gold"},
                {"range":[60,100],"color":"red"},
            ]
        }
    ))

    st.plotly_chart(fig,use_container_width=True)

    st.info(
        f"""
### Recommended Action

**Decision:** {decision}

**Suggested Credit Limit:** {limit}

**Suggested Loan Duration:** {duration}
"""
    )