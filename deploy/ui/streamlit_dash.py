import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from pathlib import Path
from datetime import date

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
API_URL = "http://127.0.0.1:8000/predict"
DATA_PATH = Path("assets/cleaned_data.csv")

st.set_page_config(
    page_title="MedOptix Analytics | The HealInsight Initiative",
    page_icon="🩺",
    layout="wide"
)

# --------------------------------------------------
# THEME
# --------------------------------------------------
st.markdown("""
<style>
.main { background:#f7f9fc; }
.main-header {
 text-align:center;
 padding:28px 0;
 background: linear-gradient(90deg,#1f4fd8,#4fa3ff);
 color:white;
 margin:-40px -40px 25px -40px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
<h1>🩺 MedOptix Analytics | The HealInsight Initiative</h1>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data(DATA_PATH)

# --------------------------------------------------
# HOSPITAL MAP
# --------------------------------------------------
HOSPITAL_NAME_MAP = {
    1: "Helsinki Central Hospital",
    2: "Tampere City Hospital",
    3: "Turku University Hospital",
    4: "Oulu Regional Hospital"
}

df["hospital_name"] = df["hospital_id"].map(HOSPITAL_NAME_MAP)
df["year"] = df["date"].dt.year

# --------------------------------------------------
# CONTEXT CONTROLS
# --------------------------------------------------
st.subheader("Context Controls")

c1, c2 = st.columns(2)

hospital = c1.selectbox(
    "Hospital",
    sorted(df["hospital_name"].dropna().unique())
)

year = c2.selectbox(
    "Year",
    sorted(df["year"].unique()),
    index=len(df["year"].unique()) - 1
)

# --------------------------------------------------
# FILTERED DATA
# --------------------------------------------------
hdf = df[df["hospital_name"] == hospital]
yf = hdf[hdf["year"] == year]

# --------------------------------------------------
# KPI STATUS LOGIC
# --------------------------------------------------
bed_util = yf["occupancy_rate_lag1"].mean()

if bed_util < 0.85:
    bed_status = "Low utilisation"
elif bed_util > 0.95:
    bed_status = "Over capacity"
else:
    bed_status = "Optimal range"

staffing_value = yf["staffing_index"].mean()

if staffing_value < 0.95:
    staffing_status = "Understaffed"
elif staffing_value > 1.05:
    staffing_status = "Overstaffed"
else:
    staffing_status = "Adequately staffed"

# --------------------------------------------------
# EXECUTIVE OVERVIEW
# --------------------------------------------------
st.subheader("Executive Overview")

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric(
    f"Total Admissions ({year})",
    f"{yf['admissions'].sum()/1000:.0f}K"
)

k2.metric(
    "Avg Bed Utilisation",
    f"{bed_util:.0%}",
    bed_status
)

k3.metric(
    "Avg Wait Time",
    f"{yf['wait_per_triage'].mean():.0f} mins"
)

k4.metric(
    "Readmission Rate",
    f"{yf['outcome_readmit_30d'].mean():.2%}"
)

k5.metric(
    "Staffing Level",
    f"{staffing_value:.2f}",
    staffing_status
)

# --------------------------------------------------
# IMAGE + STRATEGIC CHARTS
# --------------------------------------------------
img_col, charts_col = st.columns([0.8, 2.2])
#img_col.image("assets/dashpic.png", use_container_width=True)
img_col.image("assets/dashpic.png", width=260)
img_col.markdown("<div style='height:0px'></div>", unsafe_allow_html=True)

trend_col, ward_col = charts_col.columns([1.45, 1.05])

# ---- MONTHLY ADMISSIONS TREND ----
monthly = (
    yf.groupby(pd.Grouper(key="date", freq="M"))["admissions"]
    .sum()
    .reset_index()
)

fig_month = px.line(
    monthly,
    x="date",
    y="admissions",
    markers=True,
    height=330,
    color_discrete_sequence=["#1f4fd8"]
)

fig_month.update_traces(marker=dict(size=7))
fig_month.update_layout(
    title={
        "text": f"Total Monthly Admissions<br><sup>{hospital} · {year}</sup>",
        "x": 0.0
    },
    showlegend=False
)
fig_month.update_xaxes(tickformat="%b", tickangle=-45, dtick="M1")
trend_col.plotly_chart(fig_month, use_container_width=True)

# ---- ADMISSIONS BY WARD ----
ward_bar = (
    yf.groupby("ward_code")["admissions"]
    .sum()
    .reset_index()
)

ward_bar["label"] = ward_bar["admissions"].apply(lambda x: f"{int(x/1000)}k")
ymax = ward_bar["admissions"].max() * 1.15

fig_ward = px.bar(
    ward_bar,
    x="ward_code",
    y="admissions",
    text="label",
    color="ward_code",
    height=380,
    color_discrete_sequence=px.colors.qualitative.Pastel
)

fig_ward.update_layout(
    title={
        "text": f"Total Admissions by Ward<br><sup>{hospital} · {year}</sup>",
        "x": 0.0
    },
    xaxis_title="Ward",
    yaxis_title="Admissions",
    margin=dict(t=120),
    yaxis=dict(range=[0, ymax])
)
fig_ward.update_traces(textposition="outside")
fig_ward.update_xaxes(tickangle=-45)
ward_col.plotly_chart(fig_ward, use_container_width=True)


# --------------------------------------------------
# OPERATIONAL ANALYTICS (COMPACT, HORIZONTAL BARS)
# --------------------------------------------------

st.markdown("<div style='margin-top:-35px'></div>", unsafe_allow_html=True)
st.subheader("Operational Analytics")

op_df = yf.copy()
op_df["day_of_week"] = op_df["date"].dt.day_name()

dow_order = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

# ---- 2x2 GRID
r1c1, r1c2 = st.columns([1.2, 0.8])
r2c1, r2c2 = st.columns(2)

# --------------------------------------------------
# 1️⃣ ARRIVAL SOURCE DISTRIBUTION (DONUT)
# --------------------------------------------------
with r1c1:
    # Narrow column so slicer stays compact
    left_col, right_col = st.columns([0.45, 1.55])

    with left_col:
        ward = st.selectbox(
            "Ward",
            sorted(op_df["ward_code"].unique()),
            key="ward_arrival"
        )

    wf = op_df[op_df["ward_code"] == ward]

    arrival_cols = [
        "arrival_source_ambulance",
        "arrival_source_referral",
        "arrival_source_self",
        "arrival_source_transfer"
    ]

    arrivals = wf[arrival_cols].sum()

    if arrivals.sum() == 0:
        st.info("No arrival source data available for this ward.")
    else:
        adf = arrivals.reset_index()
        adf.columns = ["Source", "Count"]
        adf["Source"] = (
            adf["Source"]
            .str.replace("arrival_source_", "", regex=False)
            .str.capitalize()
        )

        fig_arrival = px.pie(
            adf,
            names="Source",
            values="Count",
            hole=0.6,
            title="Arrival Source Distribution",
            color_discrete_sequence=px.colors.qualitative.Safe,
            height=240
        )

        # Layout: vertical legend on the LEFT, donut on the RIGHT
        fig_arrival.update_layout(
            legend_title_text="",
            legend=dict(
            orientation="v",
            xanchor="left",
            x=0.02,      # very close to donut
            yanchor="middle",
            y=0.5
            ),
            margin=dict(t=40, b=10, l=0, r=10)
        )

        # Shift donut right to make room for legend (key fix)
        fig_arrival.update_traces(
            domain=dict(x=[0.22, 0.98], y=[0.0, 1.0])
        )

        st.plotly_chart(fig_arrival, use_container_width=True)



# --------------------------------------------------
# 2️⃣ AVG TRIAGE WAIT TIME BY AGE GROUP (VERTICAL, REFINED)
# --------------------------------------------------
with r1c2:
    bins = [0, 39, 59, 120]
    labels = ["20–39", "40–59", "60+"]

    age_df = op_df.copy()
    age_df["age_band"] = pd.cut(age_df["age"], bins=bins, labels=labels)

    age_wait = (
        age_df.groupby("age_band")["wait_per_triage"]
        .mean()
        .round(2)
        .reset_index()
    )

    ymax_age = age_wait["wait_per_triage"].max() * 1.2

    fig_age = px.bar(
        age_wait,
        x="age_band",
        y="wait_per_triage",
        color="age_band",
        color_discrete_sequence=px.colors.qualitative.Set2,
        height=300
    )

    fig_age.update_layout(
        title="Average Triage Wait Time by Age Group",
        xaxis_title="Age Group",
        yaxis_title="Time (mins)",
        yaxis=dict(range=[0, ymax_age]),
        showlegend=False,
        bargap=0.10
    )

    fig_age.update_traces(
        width=0.35   # slimmer bars, but not skinny
    )

    fig_age.update_xaxes(tickangle=-30)

    st.plotly_chart(fig_age, use_container_width=True)

# --------------------------------------------------
# 3️⃣ CAPACITY VS OCCUPANCY (MONTHLY)
# --------------------------------------------------
with r2c1:
    cap_df = op_df.copy()

    # Extract month number and name
    cap_df["month_num"] = cap_df["date"].dt.month
    cap_df["month"] = cap_df["date"].dt.strftime("%b")

    # Aggregate by month (hospital + year already filtered)
    cap_monthly = (
        cap_df.groupby(["month_num", "month"])
        .agg(
            capacity=("effective_capacity", "mean"),
            occupancy_rate=("occupancy_rate_lag1", "mean")
        )
        .reset_index()
    )

    # Ensure all 12 months appear
    all_months = pd.DataFrame({
        "month_num": list(range(1, 13)),
        "month": [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
        ]
    })

    cap_monthly = (
        all_months.merge(cap_monthly, on=["month_num", "month"], how="left")
        .sort_values("month_num")
    )

    # Convert occupancy to percentage
    cap_monthly["occupancy_pct"] = cap_monthly["occupancy_rate"] * 100

    # Dynamic zoom for secondary axis
    min_occ = cap_monthly["occupancy_pct"].min()
    max_occ = cap_monthly["occupancy_pct"].max()

    lower = max(0, min_occ - 1)
    upper = min(100, max_occ + 1)

    # Capacity bars
    fig_cap = px.bar(
        cap_monthly,
        x="month",
        y="capacity",
        title="Capacity vs Occupancy (Monthly)",
        labels={"capacity": "Capacity (Beds)"},
        color_discrete_sequence=["#62b6cb"],
        height=340
    )

    fig_cap.update_traces(
        name="Capacity",
        showlegend=True,
        text=cap_monthly["capacity"].apply(
            lambda x: f"{x/1000:.0f}k" if pd.notna(x) and x >= 1000 else ""
        ),
        textposition="outside"
    )

    # Occupancy line (secondary axis)
    fig_cap.add_scatter(
        x=cap_monthly["month"],
        y=cap_monthly["occupancy_pct"],
        mode="lines+markers",
        name="Occupancy (%)",
        yaxis="y2",
        line=dict(color="#d9ed92", width=3),
        marker=dict(size=7)
    )

    # Layout: remove gridlines + tidy legend
    fig_cap.update_layout(
        xaxis=dict(
            title="Month",
            showgrid=False
        ),
        yaxis=dict(
            title="Capacity (Beds)",
            showgrid=False,
            tickformat="~s"
        ),
        yaxis2=dict(
            title="Occupancy (%)",
            overlaying="y",
            side="right",
            showgrid=False,
            range=[lower, upper],
            dtick=0.5,
            tickformat=".1f"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.12,
            xanchor="right",
            x=0.98
        ),
        bargap=0.25
        
    )

    # Force all month labels to show
    fig_cap.update_xaxes(
        tickmode="array",
        tickvals=cap_monthly["month"],
        tickangle=-30
    )

    st.plotly_chart(fig_cap, use_container_width=True)


# --------------------------------------------------
# 4️⃣ AVG WAIT TIME BY OUTCOME (HORIZONTAL)
# --------------------------------------------------
with r2c2:
    outcome_cols = {
        "Discharged": "outcome_discharged",
        "Readmitted (30d)": "outcome_readmit_30d",
        "Transferred": "outcome_transferred",
        "Death": "outcome_death",
        "Unknown": "outcome_unknown"
    }

    rows = []
    for label, col in outcome_cols.items():
        subset = op_df[op_df[col] == 1]
        if len(subset) > 0:
            rows.append({
                "Outcome": label,
                "Avg Wait (mins)": subset["wait_per_triage"].mean()
            })

    outcome_wait = pd.DataFrame(rows)
    outcome_wait["Avg Wait (mins)"] = outcome_wait["Avg Wait (mins)"].round(2)

    fig_outcome = px.bar(
        outcome_wait,
        y="Outcome",
        x="Avg Wait (mins)",
        orientation="h",
        color="Outcome",
        color_discrete_sequence=px.colors.qualitative.Set3,
        height=300
    )

    fig_outcome.update_layout(
        title="Average Wait Time by Outcome",
        xaxis_title="Time (mins)",
        yaxis_title=None,
        bargap=0.15,
        showlegend=False
    )

    st.plotly_chart(fig_outcome, use_container_width=True)


# --------------------------------------------------
# FORECAST TOOL (HORIZONTAL CONTROLS + SIDE-BY-SIDE OUTPUT)
# --------------------------------------------------
st.subheader("Forecast Tool")

# -------- CONTROLS (HORIZONTAL)
fc1, fc2, fc3, fc4, fc5, fc6, fc7 = st.columns(7)

with fc1:
    steps = st.slider("Horizon", 1, 30, 7)

with fc2:
    occ = st.number_input("Occupancy", 0.0, 1.0, 0.6)

with fc3:
    staff = st.number_input("Staffing", 0.0, 2.0, 0.9)

with fc4:
    over = st.number_input("Overflow", 0.0, 500.0, 40.0)

with fc5:
    waitlag = st.number_input("Wait lag", 0.0, 1000.0, 200.0)

with fc6:
    beds = st.number_input("Beds", 1, 500, 30)

with fc7:
    cap = st.number_input("Capacity", 1, 500, 34)

run_forecast = st.button("🚀 Run Forecast")

# -------- OUTPUTS (SIDE-BY-SIDE)
if run_forecast:
    preds = requests.post(
        API_URL,
        json={
            "steps": steps,
            "features": {
                "occupancy_rate_lag1": occ,
                "overflow_lag1": over,
                "avg_wait_minutes_lag1": waitlag,
                "base_beds": beds,
                "effective_capacity": cap,
                "staffing_index": staff
            }
        }
    ).json()["predictions"]

    forecast_df = pd.DataFrame({
        "Date": pd.date_range(date.today(), periods=len(preds)),
        "Predicted Admissions": preds
    })

    chart_col, table_col = st.columns([2.0, 1.0])

    # ---- Forecast chart (primary)
    with chart_col:
        fig_fore = px.line(
            forecast_df,
            x="Date",
            y="Predicted Admissions",
            markers=True,
            title="Daily Admission Forecast",
            height=420
        )
        fig_fore.update_traces(marker=dict(size=6))
        st.plotly_chart(fig_fore, use_container_width=True)

    # ---- Forecast table (secondary, decimals preserved)
    with table_col:
        st.markdown("**Forecast values**")
        st.dataframe(
            forecast_df,
            use_container_width=True,
            height=420,
            column_config={
                "Date": st.column_config.DateColumn(
                    "Date",
                    width="small"
                ),
                "Predicted Admissions": st.column_config.NumberColumn(
                    "Predicted",
                    format="%.2f",   # ✅ decimals restored
                    width="small"
                )
            }
        )


