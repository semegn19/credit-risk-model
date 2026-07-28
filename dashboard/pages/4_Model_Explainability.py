import matplotlib.pyplot as plt
import shap
import streamlit as st

from dashboard.shap_utils import compute_shap

st.set_page_config(
    page_title="Model Explainability",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Model Explainability")

st.write(
    """
    This page explains the trained Random Forest model using
    SHAP (SHapley Additive exPlanations). SHAP identifies how each
    feature contributes to the predicted credit risk, improving
    transparency and supporting explainable lending decisions.
    """
)

with st.spinner("Generating SHAP explanations..."):

    explanation, X = compute_shap()


tab1, tab2 = st.tabs(
    [
        "Global Importance",
        "Individual Prediction"
    ]
)

# ===================================================
# GLOBAL IMPORTANCE
# ===================================================

with tab1:

    st.subheader("Global Feature Importance")

    positive = explanation[:, :, 1]

    fig, ax = plt.subplots(figsize=(10,6))

    shap.plots.beeswarm(
        positive,
        max_display=15,
        show=False
    )

    st.pyplot(fig)
    plt.close(fig)

    st.success(
        """
        Features at the top contribute most strongly to the model's
        credit risk predictions across the entire customer portfolio.
        """
    )

# ===================================================
# LOCAL EXPLANATION
# ===================================================

with tab2:

    st.subheader("Explain One Customer")

    customer = st.slider(
        "Customer Index",
        0,
        len(X)-1,
        0
    )

    sample = positive[0]

    fig, ax = plt.subplots(figsize=(8,6))

    shap.plots.waterfall(
        sample,
        max_display=15,
        show=False
    )

    st.pyplot(fig)
    plt.close(fig)

st.divider()

st.subheader("Business Interpretation")

st.markdown(
"""
### Global Feature Importance

The SHAP summary plot ranks the variables that most strongly influence
credit risk across the customer portfolio. Features appearing near the
top of the chart contribute the greatest predictive power.

### Individual Explanation

The waterfall plot explains a single prediction by showing which
features increase or decrease the estimated probability of default.

This level of explainability supports:

- transparent lending decisions
- regulatory compliance
- model validation
- customer-level decision justification
"""
)