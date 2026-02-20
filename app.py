# ===============================================
# SMART ELEVATOR MOVEMENT VISUALIZATION APP
# ===============================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="Smart Elevator Dashboard", layout="wide")

st.title("🏢 Smart Elevator Predictive Maintenance Dashboard")
st.markdown("### Elevator Sensor Data Analysis")

st.markdown("""
This dashboard analyzes elevator sensor data to study how revolutions 
and environmental conditions influence vibration levels.

Vibration is considered the health indicator of the elevator system.
""")

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("Elevator predictive-maintenance-dataset.csv")

    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing vibration values
    df["vibration"] = df["vibration"].fillna(df["vibration"].mean())

    return df

df = load_data()

# ------------------------------------------------
# SIDEBAR FILTER
# ------------------------------------------------

st.sidebar.header("🔎 Filter Data")

min_rev = float(df["revolutions"].min())
max_rev = float(df["revolutions"].max())

rev_range = st.sidebar.slider(
    "Select Revolutions Range",
    min_value=float(min_rev),
    max_value=float(max_rev),
    value=(float(min_rev), float(max_rev))
)

df_filtered = df[
    (df["revolutions"] >= rev_range[0]) &
    (df["revolutions"] <= rev_range[1])
]

# ------------------------------------------------
# DATA PREVIEW
# ------------------------------------------------

st.subheader("📊 Dataset Preview")
st.dataframe(df_filtered.head())

# ------------------------------------------------
# 1️⃣ VIBRATION OVER TIME
# ------------------------------------------------

st.subheader("📈 Vibration Over Time")

fig1 = px.line(
    df_filtered,
    x="ID",
    y="vibration",
    title="Vibration Over Time"
)

st.plotly_chart(fig1, use_container_width=True)

# ------------------------------------------------
# 2️⃣ DISTRIBUTION ANALYSIS
# ------------------------------------------------

st.subheader("📊 Distribution Analysis")

col1, col2 = st.columns(2)

with col1:
    fig2 = px.histogram(
        df_filtered,
        x="humidity",
        nbins=40,
        title="Humidity Distribution"
    )
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    fig3 = px.histogram(
        df_filtered,
        x="revolutions",
        nbins=40,
        title="Revolutions Distribution"
    )
    st.plotly_chart(fig3, use_container_width=True)

# ------------------------------------------------
# 3️⃣ REVOLUTIONS VS VIBRATION
# ------------------------------------------------

st.subheader("🔎 Revolutions vs Vibration")

fig4 = px.scatter(
    df_filtered,
    x="revolutions",
    y="vibration",
    title="Revolutions vs Vibration",
    trendline="ols"
)

st.plotly_chart(fig4, use_container_width=True)

# ------------------------------------------------
# 4️⃣ SENSOR BOX PLOT
# ------------------------------------------------

st.subheader("📦 Sensor Health Check (x1 - x5)")

sensor_data = df_filtered[["x1", "x2", "x3", "x4", "x5"]]

fig5, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=sensor_data, ax=ax)
ax.set_title("Sensor Reading Distribution")
st.pyplot(fig5)

# ------------------------------------------------
# 5️⃣ CORRELATION HEATMAP
# ------------------------------------------------

st.subheader("🔥 Correlation Heatmap")

corr = df_filtered.corr()

fig6, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig6)

# ------------------------------------------------
# INSIGHTS SECTION
# ------------------------------------------------

st.subheader("💡 Key Insights")

st.markdown("""
- Higher revolutions generally increase vibration.
- Humidity may influence mechanical performance.
- Certain sensors show variation that could indicate stress.
- Monitoring vibration trends can help prevent elevator failure.
""")

st.success("Dashboard Ready! Use sidebar filters to explore the data.")
