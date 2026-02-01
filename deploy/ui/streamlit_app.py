import streamlit as st
import requests
import pandas as pd
from datetime import date
import plotly.express as px

API_URL = "http://127.0.0.1:8000/predict"  # your FastAPI endpoint

# -------------------------------
# Page config & styles
# -------------------------------
st.set_page_config(
    page_title="MedOptix Forecast",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
    <style>
        .main { padding-top: 1rem; }
        .stButton > button {
            width: 100%;
            border-radius: 8px;
            height: 3rem;
            font-size: 1.1rem;
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("⚙ Settings")
st.sidebar.markdown("Configure forecasting options.")

steps = st.sidebar.slider("Forecast horizon (days)", 1, 30, 7)
start_date = st.sidebar.date_input("Forecast start date", value=date.today())
st.sidebar.markdown("---")
st.sidebar.caption("MedOptix Analytics © 2025")

st.title("🩺 MedOptix Admissions Forecast")
st.markdown("Predict hospital admissions using operational context and ward-level features.")
st.markdown("### 🔧 Input Data")

# -------------------------------
# Form for input
# -------------------------------
with st.form("prediction_form", border=True):
    col1, col2 = st.columns(2)
    with col1:
        hospital = st.selectbox(
            "Hospital",
            [
                "Helsinki Central Hospital",
                "Tampere City Hospital",
                "Turku University Hospital",
                "Oulu Regional Hospital",
                "Kuopio Medical Center",
            ],
        )
    with col2:
        ward_code = st.selectbox("Ward", ["ED", "ICU", "MED", "SURG"])

    st.markdown("### 🧮 Operational Indicators (Lagged Features)")
    c1, c2, c3 = st.columns(3)
    with c1:
        occupancy_rate = st.number_input("Occupancy rate (yesterday)", value=0.60, min_value=0.0, max_value=1.0, step=0.01)
    with c2:
        overflow_lag = st.number_input("Overflow patients (yesterday)", value=42.0, step=1.0)
    with c3:
        avg_wait_lag = st.number_input("Average wait time (mins, yesterday)", value=227.0, step=1.0)

    st.markdown("### 🛏 Capacity & Staffing")
    colA, colB, colC = st.columns(3)
    with colA:
        base_beds = st.number_input("Base beds", value=30, step=1)
    with colB:
        effective_capacity = st.number_input("Effective capacity", value=34, step=1)
    with colC:
        staffing_index = st.number_input("Staffing index", value=0.927)

    st.markdown("---")
    submitted = st.form_submit_button("🚀 Run Forecast")

# -------------------------------
# Send request & display results
# -------------------------------
if submitted:
    payload = {
        "steps": int(steps),
        "features": {
            "occupancy_rate_lag1": occupancy_rate,
            "overflow_lag1": overflow_lag,
            "avg_wait_minutes_lag1": avg_wait_lag,
            "base_beds": base_beds,
            "effective_capacity": effective_capacity,
            "staffing_index": staffing_index
        },
        # optional scenario trends can be added later
        "scenario": {}
    }

    with st.spinner("Generating forecast…"):
        try:
            response = requests.post(API_URL, json=payload)
            data = response.json()

            if response.status_code == 200:
                forecast_values = data.get("predictions", [])

                st.success("Forecast generated successfully! ✅")
                st.markdown(f"## 📊 Forecast for **{hospital} — {ward_code}**")

                # Show big number if single-step forecast
                if len(forecast_values) == 1:
                    st.markdown(
                        f"""
                        <div style="
                            background-color:#f0f2f6;
                            padding:30px;
                            border-radius:10px;
                            text-align:center;
                            margin-bottom:20px;">
                            <h1 style="font-size:64px; margin:0;">{forecast_values[0]}</h1>
                            <p style="font-size:20px; margin-top:10px;">
                                Admissions forecast for <b>{start_date.strftime("%Y-%m-%d")}</b>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    # Multi-step forecast chart
                    dates = pd.date_range(start=start_date, periods=len(forecast_values))
                    df = pd.DataFrame({"Date": dates, "Admissions forecast": forecast_values})

                    fig = px.line(df, x="Date", y="Admissions forecast", markers=True,
                                  title="Admissions Forecast Over Time")
                    fig.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Forecasted Admissions",
                        template="plotly_white",
                        paper_bgcolor="white",
                        plot_bgcolor="white",
                        height=450,
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(df, use_container_width=True)

            else:
                st.error(f"API error {response.status_code}: {data.get('detail', response.text)}")

        except Exception as e:
            st.error(f"Request failed: {e}")
