import streamlit as st
import pandas as pd
import plotly.express as px
import random
from pathlib import Path
from datetime import date

# --------------------------------------------------
# CONFIG & CSS FOR COMPACT "NO-SCROLL" LAYOUT
# --------------------------------------------------
DATA_PATH = Path("DataCleaning/cleaned_data.csv")

st.set_page_config(
    page_title="MedOptix Analytics | The HealInsight Initiative",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to eliminate whitespace while keeping the banner visible
st.markdown("""
<style>
.main { background:#f7f9fc; }
.block-container { 
    padding-top: 2rem !important; 
    padding-bottom: 1rem !important; 
    max-width: 98% !important; 
}
.main-header {
    text-align: center;
    padding: 15px 0;
    background: linear-gradient(90deg, #1f4fd8, #4fa3ff);
    color: white;
    border-radius: 8px;
    margin-bottom: 20px;
}
.main-header h2 { 
    margin: 0; 
    padding: 0; 
    font-size: 1.8rem; 
    color: white;
}
div[data-testid="stMetricValue"] { font-size: 1.4rem !important; }
div[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# VISUAL BANNER
# --------------------------------------------------
st.markdown("""
<div class="main-header">
<h2>🩺 MedOptix Analytics | The HealInsight Initiative</h2>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data(path):
    if not path.exists():
        return pd.DataFrame(columns=["date", "hospital_id", "occupancy_rate_lag1", 
                                     "staffing_index", "admissions", "wait_per_triage", 
                                     "outcome_readmit_30d", "ward_code", "age",
                                     "effective_capacity", "arrival_source_ambulance",
                                     "arrival_source_referral", "arrival_source_self",
                                     "arrival_source_transfer", "outcome_discharged",
                                     "outcome_transferred", "outcome_death", "outcome_unknown"])
    
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data(DATA_PATH)

HOSPITAL_NAME_MAP = {
    1: "Helsinki Central Hospital",
    2: "Tampere City Hospital",
    3: "Turku University Hospital",
    4: "Oulu Regional Hospital"
}

if not df.empty:
    df["hospital_name"] = df["hospital_id"].map(HOSPITAL_NAME_MAP)
    df["year"] = df["date"].dt.year

# --------------------------------------------------
# SIDEBAR (MULTI-SELECT CONTROLS)
# --------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Controls")
    
    if not df.empty:
        all_hospitals = sorted(df["hospital_name"].dropna().unique())
        selected_hospitals = st.multiselect("Hospital", all_hospitals, default=all_hospitals)
        
        all_years = sorted(df["year"].unique())
        selected_years = st.multiselect("Year", all_years, default=all_years)
        
        all_wards = sorted(df["ward_code"].unique())
        selected_wards = st.multiselect("Ward (For Arrival Source)", all_wards, default=all_wards)
    else:
        st.warning("No data available.")

# --------------------------------------------------
# TABS SETUP
# --------------------------------------------------
tab_dash, tab_fore = st.tabs(["📊 Executive Dashboard", "📈 Forecast Tool"])

# ==================================================
# TAB 1: DASHBOARD (NO SCROLLING)
# ==================================================
with tab_dash:
    if not df.empty:
        if not selected_hospitals or not selected_years:
            st.warning("⚠️ Please select at least one Hospital and one Year from the sidebar to view data.")
        else:
            # ---- FILTER DATA (Using .isin() for multiple selections) ----
            yf = df[df["hospital_name"].isin(selected_hospitals) & df["year"].isin(selected_years)]

            if yf.empty:
                st.warning("No data matches the current filter combination.")
            else:
                # ---- KPI LOGIC ----
                bed_util = yf["occupancy_rate_lag1"].mean()
                bed_status = "Low utilisation" if bed_util < 0.85 else "Over capacity" if bed_util > 0.95 else "Optimal range"
                
                staffing_value = yf["staffing_index"].mean()
                staffing_status = "Understaffed" if staffing_value < 0.95 else "Overstaffed" if staffing_value > 1.05 else "Adequate"

                # Formatting year string for the metric title
                year_label = f"{len(selected_years)} Years" if len(selected_years) > 2 else ", ".join(map(str, selected_years))

                # ---- EXECUTIVE METRICS (TOP ROW) ----
                k1, k2, k3, k4, k5 = st.columns(5)
                k1.metric(f"Total Admissions ({year_label})", f"{yf['admissions'].sum()/1000:.1f}K")
                k2.metric("Avg Bed Utilisation", f"{bed_util:.0%}", bed_status)
                k3.metric("Avg Wait Time", f"{yf['wait_per_triage'].mean():.0f} mins")
                k4.metric("Readmission Rate", f"{yf['outcome_readmit_30d'].mean():.2%}")
                k5.metric("Staffing Level", f"{staffing_value:.2f}", staffing_status)

                st.markdown("<div style='margin-top:-15px'></div>", unsafe_allow_html=True)

                # ---- CHARTS GRID (2 ROWS x 3 COLUMNS) ----
                CHART_HEIGHT = 240
                MARGINS = dict(t=30, b=10, l=10, r=10)

                # Row 1
                r1c1, r1c2, r1c3 = st.columns(3)
                
                # 1. Admissions Trend
                with r1c1:
                    monthly = yf.groupby(pd.Grouper(key="date", freq="ME"))["admissions"].sum().reset_index()
                    fig_month = px.line(monthly, x="date", y="admissions", markers=True, height=CHART_HEIGHT, color_discrete_sequence=["#1f4fd8"])
                    fig_month.update_layout(title="Total Monthly Admissions", margin=MARGINS, xaxis_title="", yaxis_title="")
                    fig_month.update_xaxes(tickformat="%b", dtick="M1")
                    st.plotly_chart(fig_month, use_container_width=True)

                # 2. Admissions by Ward
                with r1c2:
                    ward_bar = yf.groupby("ward_code")["admissions"].sum().reset_index()
                    ward_bar["label"] = ward_bar["admissions"].apply(lambda x: f"{int(x/1000)}k")
                    fig_ward = px.bar(ward_bar, x="ward_code", y="admissions", text="label", color="ward_code", height=CHART_HEIGHT, color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig_ward.update_layout(title="Admissions by Ward", margin=MARGINS, showlegend=False, xaxis_title="", yaxis_title="")
                    fig_ward.update_traces(textposition="outside")
                    st.plotly_chart(fig_ward, use_container_width=True)

                # 3. Capacity vs Occupancy
                with r1c3:
                    cap_df = yf.copy()
                    cap_df["month_num"] = cap_df["date"].dt.month
                    cap_df["month"] = cap_df["date"].dt.strftime("%b")
                    cap_monthly = cap_df.groupby(["month_num", "month"]).agg(capacity=("effective_capacity", "mean"), occupancy_rate=("occupancy_rate_lag1", "mean")).reset_index().sort_values("month_num")
                    cap_monthly["occupancy_pct"] = cap_monthly["occupancy_rate"] * 100
                    
                    fig_cap = px.bar(cap_monthly, x="month", y="capacity", height=CHART_HEIGHT, color_discrete_sequence=["#62b6cb"])
                    fig_cap.add_scatter(x=cap_monthly["month"], y=cap_monthly["occupancy_pct"], mode="lines+markers", name="Occ (%)", yaxis="y2", line=dict(color="#d9ed92", width=2))
                    fig_cap.update_layout(
                        title="Capacity vs Occupancy", margin=MARGINS, showlegend=False, xaxis_title="", yaxis_title="",
                        yaxis2=dict(overlaying="y", side="right", showgrid=False)
                    )
                    st.plotly_chart(fig_cap, use_container_width=True)

                # Row 2
                r2c1, r2c2, r2c3 = st.columns(3)

                # 4. Arrival Source Distribution (Tied to Sidebar Ward Select)
                with r2c1:
                    if not selected_wards:
                        st.info("⚠️ Select a ward from the sidebar to view arrivals.")
                    else:
                        wf = yf[yf["ward_code"].isin(selected_wards)]
                        arrivals = wf[["arrival_source_ambulance", "arrival_source_referral", "arrival_source_self", "arrival_source_transfer"]].sum()
                        if arrivals.sum() > 0:
                            adf = arrivals.reset_index()
                            adf.columns = ["Source", "Count"]
                            adf["Source"] = adf["Source"].str.replace("arrival_source_", "").str.capitalize()
                            fig_arrival = px.pie(adf, names="Source", values="Count", hole=0.6, height=CHART_HEIGHT, color_discrete_sequence=px.colors.qualitative.Safe)
                            fig_arrival.update_layout(title="Arrival Source (Selected Wards)", margin=MARGINS, showlegend=False)
                            fig_arrival.update_traces(textposition='inside', textinfo='percent+label')
                            st.plotly_chart(fig_arrival, use_container_width=True)
                        else:
                            st.info("No arrival data for selected wards.")

                # 5. Wait Time by Age Group
                with r2c2:
                    age_df = yf.copy()
                    age_df["age_band"] = pd.cut(age_df["age"], bins=[0, 39, 59, 120], labels=["20–39", "40–59", "60+"])
                    age_wait = age_df.groupby("age_band", observed=True)["wait_per_triage"].mean().reset_index()
                    fig_age = px.bar(age_wait, x="age_band", y="wait_per_triage", color="age_band", height=CHART_HEIGHT, color_discrete_sequence=px.colors.qualitative.Set2)
                    fig_age.update_layout(title="Triage Wait by Age", margin=MARGINS, showlegend=False, xaxis_title="", yaxis_title="Mins")
                    st.plotly_chart(fig_age, use_container_width=True)

                # 6. Wait Time by Outcome
                with r2c3:
                    outcomes = {"Discharged": "outcome_discharged", "Readmitted": "outcome_readmit_30d", "Transferred": "outcome_transferred", "Death": "outcome_death"}
                    rows = [{"Outcome": label, "Avg Wait": yf[yf[col] == 1]["wait_per_triage"].mean()} for label, col in outcomes.items() if col in yf.columns and len(yf[yf[col] == 1]) > 0]
                    if rows:
                        out_df = pd.DataFrame(rows).dropna()
                        fig_out = px.bar(out_df, y="Outcome", x="Avg Wait", orientation="h", color="Outcome", height=CHART_HEIGHT, color_discrete_sequence=px.colors.qualitative.Set3)
                        fig_out.update_layout(title="Avg Wait by Outcome", margin=MARGINS, showlegend=False, xaxis_title="Mins", yaxis_title="")
                        st.plotly_chart(fig_out, use_container_width=True)

    else:
        st.warning("No data found in DataCleaning/cleaned_data.csv. Please ensure the file exists.")

# ==================================================
# TAB 2: FORECAST TOOL (STANDALONE)
# ==================================================
with tab_fore:
    st.subheader("Admission Forecasting Tool")
    
    # Keep controls horizontal
    fc1, fc2, fc3, fc4, fc5, fc6, fc7 = st.columns(7)
    with fc1: steps = st.slider("Horizon", 1, 30, 7)
    with fc2: occ = st.number_input("Occupancy", 0.0, 1.0, 0.6)
    with fc3: staff = st.number_input("Staffing", 0.0, 2.0, 0.9)
    with fc4: over = st.number_input("Overflow", 0.0, 500.0, 40.0)
    with fc5: waitlag = st.number_input("Wait lag", 0.0, 1000.0, 200.0)
    with fc6: beds = st.number_input("Beds", 1, 500, 30)
    with fc7: cap = st.number_input("Capacity", 1, 500, 34)

    run_forecast = st.button("🚀 Run Forecast", type="primary")

    if run_forecast:
        # --- MOCK PREDICTION LOGIC ---
        base_value = beds + (waitlag * 0.1) + over - (cap * 0.5)
        preds = [round(max(0, base_value + random.uniform(-5.0, 5.0)), 2) for _ in range(steps)]
        
        forecast_df = pd.DataFrame({
            "Date": pd.date_range(date.today(), periods=len(preds)),
            "Predicted Admissions": preds
        })

        chart_col, table_col = st.columns([2.0, 1.0])

        with chart_col:
            fig_fore = px.line(
                forecast_df, x="Date", y="Predicted Admissions", markers=True, 
                title="Daily Admission Forecast", height=420
            )
            fig_fore.update_traces(marker=dict(size=6))
            st.plotly_chart(fig_fore, use_container_width=True)

        with table_col:
            st.markdown("*Forecast values*")
            st.dataframe(
                forecast_df, use_container_width=True, height=420,
                column_config={
                    "Date": st.column_config.DateColumn("Date", width="small"),
                    "Predicted Admissions": st.column_config.NumberColumn("Predicted", format="%.2f", width="small")
                }
            )