import streamlit as st
import pandas as pd

from dashboard.mlflow_utils import get_latest_metrics

st.set_page_config(
    page_title="Model Performance",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Credit Risk Model Performance")

metrics = get_latest_metrics()

if metrics is None:
    st.error("No MLflow experiment found.")
    st.stop()

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
col2.metric("Precision", f"{metrics['Precision']:.3f}")
col3.metric("Recall", f"{metrics['Recall']:.3f}")
col4.metric("F1 Score", f"{metrics['F1 Score']:.3f}")
col5.metric("ROC-AUC", f"{metrics['ROC-AUC']:.3f}")

st.divider()

st.subheader("Model Metrics")

df = pd.DataFrame(
    {
        "Metric": metrics.keys(),
        "Value": [round(v, 4) for v in metrics.values()],
    }
)

st.dataframe(df, use_container_width=True)