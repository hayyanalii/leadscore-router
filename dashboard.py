"""
dashboard.py
Streamlit UI for LeadScore Router.
Self-contained — loads the model directly, no API needed.
Works on Streamlit Community Cloud.
"""

import streamlit as st
import pandas as pd
import pickle
import os
from datetime import datetime, timezone

# --- Load model once at startup ---
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.error(f"⚠️ Failed to load model: {e}")


# --- Page config ---
st.set_page_config(page_title="LeadScore Router", page_icon="📞", layout="centered")

st.title("📞 LeadScore Router")
st.markdown("**AI-powered lead scoring and routing for call centers.**")
st.markdown("---")

if not MODEL_LOADED:
    st.stop()

st.subheader("Enter caller information")

col1, col2 = st.columns(2)

with col1:
    call_duration_sec = st.slider("Call duration (seconds)", 0, 1200, 250, step=10)
    time_of_day = st.selectbox(
        "Time of day", [0, 1, 2],
        format_func=lambda x: ["Morning", "Afternoon", "Evening"][x],
    )
    ivr_option = st.selectbox(
        "IVR option selected", [0, 1, 2],
        format_func=lambda x: ["Sales", "Support", "Billing"][x],
    )

with col2:
    agent_experience_yrs = st.slider("Agent experience (years)", 0, 20, 8)
    previous_calls = st.slider("Previous calls from customer", 0, 10, 1)
    wait_time_sec = st.slider("Queue wait time (seconds)", 0, 600, 60, step=10)

st.markdown("---")

if st.button("🔮 Predict & Route", use_container_width=True, type="primary"):
    # Build DataFrame with same column names as training
    features = pd.DataFrame([{
        "call_duration_sec": call_duration_sec,
        "time_of_day": time_of_day,
        "ivr_option": ivr_option,
        "agent_experience_yrs": agent_experience_yrs,
        "previous_calls": previous_calls,
        "wait_time_sec": wait_time_sec,
    }])

    prob = float(model.predict_proba(features)[0][1])
    score = int(round(prob * 100))

    if score >= 70:
        route = "HIGH_VALUE_QUEUE"
        message = "Route to senior sales agent immediately."
    elif score >= 40:
        route = "STANDARD_QUEUE"
        message = "Route to standard sales queue."
    else:
        route = "SELF_SERVICE"
        message = "Offer self-service options to reduce agent load."

    st.success("✅ Prediction complete")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Conversion Score", f"{score}/100")
    with col_b:
        st.metric("Predicted Probability", f"{prob*100:.1f}%")

    if route == "HIGH_VALUE_QUEUE":
        st.success(f"🎯 **{route}** — {message}")
    elif route == "STANDARD_QUEUE":
        st.info(f"📋 **{route}** — {message}")
    else:
        st.warning(f"🤖 **{route}** — {message}")

    st.caption(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")

st.markdown("---")
st.caption("Built with Streamlit + scikit-learn. Part of the LeadScore Router ML project.")