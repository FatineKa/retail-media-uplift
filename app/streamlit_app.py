"""Budget-allocation dashboard. Run after training:
streamlit run app/streamlit_app.py"""
import streamlit as st

st.title("Retail Media Budget Allocator")
st.caption("Uplift targeting vs. propensity targeting — incremental profit")

budget = st.slider("% of audience to target", 2, 100, 20)
st.write(
    "TODO (Week 4): load holdout scores, plot profit curves for "
    "(a) target everyone, (b) propensity targeting, (c) uplift targeting, "
    f"and show expected incremental conversions at {budget}% targeting."
)
