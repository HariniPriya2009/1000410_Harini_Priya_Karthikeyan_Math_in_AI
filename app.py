# ===============================
# ROCKET LAUNCH VISUALIZATION APP
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="Rocket Launch Dashboard", layout="wide")

st.title("🚀 Rocket Launch Path Visualization Dashboard")
st.markdown("Mathematics for AI - Streamlit Project")

# =====================================
# LOAD & CLEAN DATA
# =====================================

@st.cache_data
def load_data():
    df = pd.read_csv("rocket_missions.csv")

    # Convert date
    df["Launch Date"] = pd.to_datetime(df["Launch Date"], errors="coerce")

    # Numeric columns
    numeric_columns = [
        "Distance from Earth",
        "Mission Duration",
        "Mission Cost",
        "Scientific Yield",
        "Crew Size",
        "Fuel Consumption",
        "Payload Weight"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates()
    df = df.dropna()

    return df, numeric_columns


df, numeric_columns = load_data()

# =====================================
# SIDEBAR FILTERS
# =====================================

st.sidebar.header("Filter Missions")

mission_type = st.sidebar.selectbox(
    "Select Mission Type",
    ["All"] + list(df["Mission Type"].unique())
)

if mission_type != "All":
    df = df[df["Mission Type"] == mission_type]

# =====================================
# DATA VISUALIZATION SECTION
# =====================================

st.header("📊 Real Mission Data Analysis")

col1, col2 = st.columns(2)

# 1️⃣ Scatter Plot
with col1:
    fig1 = px.scatter(
        df,
        x="Payload Weight",
        y="Fuel Consumption",
        color="Mission Success",
        title="Payload Weight vs Fuel Consumption"
    )
    st.plotly_chart(fig1, use_container_width=True)

# 2️⃣ Bar Chart
with col2:
    cost_data = df.groupby("Mission Success")["Mission Cost"].mean().reset_index()

    fig2 = px.bar(
        cost_data,
        x="Mission Success",
        y="Mission Cost",
        title="Average Mission Cost: Success vs Failure"
    )
    st.plotly_chart(fig2, use_container_width=True)

# 3️⃣ Line Chart
col3, col4 = st.columns(2)

with col3:
    fig3 = px.line(
        df.sort_values("Distance from Earth"),
        x="Distance from Earth",
        y="Mission Duration",
        title="Mission Duration vs Distance from Earth"
    )
    st.plotly_chart(fig3, use_container_width=True)

# 4️⃣ Box Plot
with col4:
    fig4 = px.box(
        df,
        x="Mission Success",
        y="Crew Size",
        title="Crew Size vs Mission Success"
    )
    st.plotly_chart(fig4, use_container_width=True)

# 5️⃣ Correlation Heatmap
st.subheader("Correlation Heatmap")

corr = df[numeric_columns].corr()

fig5, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig5)

# =====================================
# ROCKET SIMULATION SECTION
# =====================================

st.header("🧮 Rocket Launch Physics Simulation")

st.markdown("""
This simulation applies Newton's Second Law:

Acceleration = (Thrust − Gravity − Drag) / Mass

Fuel decreases over time, reducing mass and changing acceleration.
""")

def rocket_simulation(thrust, payload_weight, fuel_mass, drag_factor):

    g = 9.81
    time_step = 1
    total_time = 120

    mass = payload_weight + fuel_mass + 10000
    velocity = 0
    altitude = 0

    results = []

    for t in range(total_time):

        if fuel_mass > 0:
            burn_rate = 50
            fuel_mass -= burn_rate
            mass -= burn_rate
        else:
            thrust = 0

        drag = drag_factor * velocity**2

        acceleration = (thrust - (mass * g) - drag) / mass

        velocity += acceleration * time_step
        altitude += velocity * time_step

        if altitude < 0:
            altitude = 0

        results.append([t, altitude, velocity, acceleration])

    return pd.DataFrame(results, columns=["Time", "Altitude", "Velocity", "Acceleration"])


# =====================================
# SIMULATION CONTROLS
# =====================================

st.sidebar.header("Simulation Controls")

thrust = st.sidebar.slider("Thrust (N)", 500000, 2000000, 1000000)
payload = st.sidebar.slider("Payload Weight (kg)", 1000, 20000, 5000)
fuel = st.sidebar.slider("Fuel Mass (kg)", 10000, 100000, 50000)
drag = st.sidebar.slider("Drag Factor", 0.0, 0.5, 0.02)

sim_df = rocket_simulation(thrust, payload, fuel, drag)

# =====================================
# SIMULATION PLOTS
# =====================================

col5, col6 = st.columns(2)

with col5:
    fig_alt = px.line(
        sim_df,
        x="Time",
        y="Altitude",
        title="Rocket Altitude Over Time"
    )
    st.plotly_chart(fig_alt, use_container_width=True)

with col6:
    fig_vel = px.line(
        sim_df,
        x="Time",
        y="Velocity",
        title="Rocket Velocity Over Time"
    )
    st.plotly_chart(fig_vel, use_container_width=True)

st.success("🚀 Simulation Complete! Adjust sliders to test different launch scenarios.")
