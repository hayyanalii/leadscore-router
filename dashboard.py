"""
dashboard.py
Streamlit UI for the LeadScore Router API.
Lets non-technical users enter caller data and see predictions instantly.
"""

import streamlit as st
import requests
from datetime import datetime

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="LeadScore Router", page_icon="📞", layout="centered")

st.title("📞 LeadScore Router")
st.markdown("**AI-powered lead scoring and routing for call centers.**")
st.markdown("---")

st.subheader("Enter caller information")

col1, col2 = st.columns(2)

with col1:
    call_duration_sec = st.slider("Call duration (seconds)", 0, 1200, 250, step=10)
    time_of_day = st.selectbox("Time of day", [0, 1, 2],
                                format_func=lambda x: ["🌅 Morning", "☀️ Afternoon", "🌙 Evening"][x])
    ivr_option = st.selectbox("IVR option selected", [0, 1, 2],
                               format_func=lambda x: ["💰 Sales", "🛠️ Support", "💳 Billing"][x])

with col2:
    agent_experience_yrs = st.slider("Agent experience (years)", 0, 20, 8)
    previous_calls = st.slider("Previous calls from customer", 0, 10, 1)
    wait_time_sec = st.slider("Queue wait time (seconds)", 0, 600, 60, step=10)

st.markdown("---")

if st.button("🔮 Predict & Route", use_container_width=True, type="primary"):
    payload = {
        "call_duration_sec": call_duration_sec,
        "time_of_day": time_of_day,
        "ivr_option": ivr_option,
        "agent_experience_yrs": agent_experience_yrs,
        "previous_calls": previous_calls,
        "wait_time_sec": wait_time_sec,
    }

    try:
        with st.spinner("Calling the AI model..."):
            resp = requests.post(API_URL, json=payload, timeout=10)

        if resp.status_code == 200:
            result = resp.json()
            score = result["score"]
            route = result["route"]
            message = result["message"]

            st.success("✅ Prediction complete")

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Conversion Score", f"{score}/100")
            with col_b:
                st.metric("Predicted Probability", f"{result['conversion_probability']*100:.1f}%")

            if route == "HIGH_VALUE_QUEUE":
                st.success(f"🎯 **{route}** — {message}")
            elif route == "STANDARD_QUEUE":
                st.info(f"📋 **{route}** — {message}")
            else:
                st.warning(f"🤖 **{route}** — {message}")

            st.caption(f"Timestamp: {result['timestamp']}")
        else:
            st.error(f"API error: {resp.status_code} — {resp.text}")

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach API. Is the FastAPI server running? Run: uvicorn main:app --reload")
    except Exception as e:
        st.error(f"Something went wrong: {e}")

st.markdown("---")
st.caption("Built with FastAPI + scikit-learn + Streamlit. Predictions logged to predictions_log.csv.")