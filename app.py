import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_FILE = os.path.join(DATA_DIR, "aqi_5_years.csv")
MODEL_FILE = os.path.join(BASE_DIR, "aqi_model.pkl")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Smart City AQI", layout="wide")
st.title("🌍 Smart City AQI Monitoring & Prediction System")

# =========================
# LOAD DATA
# =========================
if not os.path.exists(DATA_FILE):
    st.error("❌ data/aqi_5_years.csv not found")
    st.stop()

df = pd.read_csv(DATA_FILE)

# Detect date column automatically
date_col = None
for c in df.columns:
    if c.lower() in ["date", "datetime", "timestamp"]:
        date_col = c
        break

if date_col is None:
    st.error("❌ No date column found in dataset")
    st.stop()

df.rename(columns={date_col: "date"}, inplace=True)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df.dropna(subset=["date"], inplace=True)

# =========================
# LOAD MODEL
# =========================
if not os.path.exists(MODEL_FILE):
    st.error("❌ aqi_model.pkl not found")
    st.stop()

model = joblib.load(MODEL_FILE)

# =========================
# USER INPUT
# =========================
st.header("🔢 Enter Current Pollution Values")

c1, c2, c3, c4 = st.columns(4)

with c1:
    pm25 = st.number_input("PM2.5 (µg/m³)", 0.0, 500.0, 55.0)
with c2:
    pm10 = st.number_input("PM10 (µg/m³)", 0.0, 500.0, 80.0)
with c3:
    no2 = st.number_input("NO2 (µg/m³)", 0.0, 300.0, 25.0)
with c4:
    co = st.number_input("CO (µg/m³)", 0.0, 30000.0, 600.0)

# =========================
# AQI PREDICTION
# =========================
input_df = pd.DataFrame([[pm25, pm10, no2, co]],
                        columns=["PM2.5", "PM10", "NO2", "CO"])

prediction = round(float(model.predict(input_df)[0]), 2)

st.subheader("📊 Current AQI Prediction")
st.metric("Predicted AQI", prediction)

# =========================
# ALERT MESSAGE (FIXED)
# =========================
if prediction <= 1.5:
    st.success("😊 Air Quality: GOOD")
elif prediction <= 2.5:
    st.info("🙂 Air Quality: FAIR")
elif prediction <= 3.5:
    st.warning("😐 Air Quality: MODERATE")
elif prediction <= 4.5:
    st.error("😷 Air Quality: POOR")
else:
    st.error("☠️ VERY POOR – HEALTH EMERGENCY")

# =========================
# 5 YEAR HISTORY GRAPH
# =========================
st.subheader("📈 AQI History (Last 5 Years)")

fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(df["date"], df["AQI"])
ax1.set_ylabel("AQI")
ax1.set_xlabel("Year")
ax1.grid(True)
st.pyplot(fig1)

# =========================
# DATA TABLE
# =========================
st.subheader("📋 Historical AQI Records")
st.dataframe(df.sort_values("date", ascending=False).head(20))

# =========================
# 7 DAY FORECAST
# =========================
st.subheader("📅 AQI Forecast (Next 7 Days)")

future_dates = []
future_aqi = []

for i in range(1, 8):
    future_dates.append(datetime.today() + timedelta(days=i))
    future_aqi.append(prediction + (i * 0.2))

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted AQI": future_aqi
})

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(forecast_df["Date"], forecast_df["Predicted AQI"], marker="o")
ax2.set_ylabel("AQI")
ax2.grid(True)
st.pyplot(fig2)

# =========================
# POLLUTION BAR CHART
# =========================
st.subheader("📉 Current Pollution Levels")

pollution_df = pd.DataFrame({
    "Pollutant": ["PM2.5", "PM10", "NO2", "CO"],
    "Value": [pm25, pm10, no2, co]
})

fig3, ax3 = plt.subplots(figsize=(6, 4))
ax3.bar(pollution_df["Pollutant"], pollution_df["Value"])
ax3.set_ylabel("Concentration")
st.pyplot(fig3)

st.markdown("---")
st.markdown("✅ **Smart City AQI Monitoring & Prediction System**")
