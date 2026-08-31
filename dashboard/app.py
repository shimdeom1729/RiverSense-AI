import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd

st.set_page_config(page_title="RiverSense AI Dashboard", layout="wide")

st.title("🌊 RiverSense AI: Riverbed Garbage & Silt Command Center")
st.markdown("Automated Remote Sensing & Hydrodynamic Drift Prediction")

# Top KPI Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Active Plastic Clusters", "12 Zones", "+2 today")
col2.metric("Surface Flow Speed", "0.48 m/s", "Normal Flow")
col3.metric("Predicted 24h Choke Risk", "HIGH", "Bridge Pier #3")

st.sidebar.header("Control Panel")
basin = st.sidebar.selectbox("Select River Basin", ["Indrayani River (Alandi)", "Mula-Mutha River", "Ganga Basin"])
forecast_hours = st.sidebar.slider("Forecast Drift Horizon (Hours)", 3, 48, 12)

# Interactive Map
st.subheader("📍 Live River Map & 12h Debris Trajectory")
m = folium.Map(location=[18.6780, 73.8980], zoom_start=15, tiles="OpenStreetMap")

# Live Detected Debris (Orange Circle)
folium.CircleMarker(
    location=[18.6782, 73.8975],
    radius=9,
    color="orange",
    fill=True,
    fill_color="orange",
    popup="Current Cluster: Plastic Debris (8.5 kg/m²)"
).add_to(m)

# Predicted Future Accumulation Choke Point (Red Warning Pin)
folium.Marker(
    location=[18.6765, 73.9015],
    popup=f"PREDICTED TRAP ZONE in {forecast_hours} Hours!",
    icon=folium.Icon(color="red", icon="exclamation-sign")
).add_to(m)

# Trajectory Line
folium.PolyLine(
    locations=[[18.6782, 73.8975], [18.6775, 73.8995], [18.6765, 73.9015]],
    color="blue",
    weight=3,
    dash_array="5, 10"
).add_to(m)

# Render Map in Streamlit
st_folium(m, width=1100, height=450)

# Work Order Dispatch Table
st.subheader("📋 Automated Municipal Skimmer Dispatch Orders")
orders_df = pd.DataFrame([
    {"Order ID": "WO-2026-01", "Target Lat": 18.67650, "Target Lon": 73.90150, "Waste Type": "PET Bottles & Bags", "Est. Volume": "3.5 m³", "Priority": "CRITICAL"},
    {"Order ID": "WO-2026-02", "Target Lat": 18.67902, "Target Lon": 73.89901, "Waste Type": "Water Hyacinth Mat", "Est. Volume": "7.2 m³", "Priority": "MODERATE"}
])
st.dataframe(orders_df, use_container_width=True)

if st.button("🚀 Dispatch Work Orders to Municipal Cleanup Crew"):
    st.success("Work orders dispatched successfully to Sanitation and Irrigation division APIs.")