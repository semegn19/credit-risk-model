import streamlit as st


def show_prediction(probability: float):

    if probability >= 0.7:
        st.error("High Credit Risk")

    elif probability >= 0.4:
        st.warning("Medium Credit Risk")

    else:
        st.success("Low Credit Risk")

    st.metric(
        "Risk Probability",
        f"{probability:.2%}"
    )