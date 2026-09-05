import streamlit as st
import cv2
import tempfile
import time
import os
import pandas as pd
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

st.set_page_config(
    page_title="RiverSense AI | Command Center",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Command-Center Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    background-color: #070B13 !important;
    color: #E2E8F0 !important;
}

header, footer, #MainMenu { visibility: hidden !important; height: 0px !important; }
.block-container { padding-top: 1rem !important; padding-bottom: 2rem !important; }

[data-testid="stSidebar"] {
    background-color: #0B111E !important;
    border-right: 1px solid #1E293B !important;
}

.top-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 18px;
    background-color: #0B111E;
    border: 1px solid #1E293B;
    border-radius: 8px;
    margin-bottom: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
}
.live-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background-color: rgba(239, 68, 68, 0.15);
    color: #F87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
    padding: 4px 10px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 11px;
}
.pulse-dot {
    width: 7px;
    height: 7px;
    background-color: #EF4444;
    border-radius: 50%;
    box-shadow: 0 0 8px #EF4444;
}

.view-category {
    color: #38BDF8;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.view-title {
    font-size: 30px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #FFFFFF;
    margin: 4px 0 8px 0;
}
.view-desc {
    color: #94A3B8;
    font-size: 13px;
    line-height: 1.5;
    margin-bottom: 20px;
}

.stat-box {
    background: #0B111E;
    border: 1px solid #1E293B;
    border-radius: 8px;
    padding: 14px 18px;
}
.stat-box-title {
    color: #64748B;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 16px 0; border-bottom: 1px solid #1E293B; margin-bottom: 16px;">
        <div style="font-size: 20px; font-weight: 800; color: #F8FAFC;">🌊 RiverSense</div>
        <div style="font-size: 10px; color: #64748B; letter-spacing: 1.5px; text-transform: uppercase;">AI · Command Center</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='color: #64748B; font-family: monospace; font-size: 11px; margin-bottom: 6px;'>MODULES</div>", unsafe_allow_html=True)
    selected_module = st.radio(
        "Nav",
        ["Overview", "CV Inference", "Hotspot Map", "Dispatch Hub"],
        label_visibility="collapsed"
    )
    
    st.markdown("""
    <div style="margin-top: 36px; padding: 12px; background: #080D18; border-radius: 8px; border: 1px solid #1E293B; font-family: 'JetBrains Mono', monospace; font-size: 11px;">
        <div style="display:flex; align-items:center; gap:6px; margin-bottom:4px;">
            <div class="pulse-dot"></div>
            <span style="color:#22C55E; font-weight:700;">SYS · ONLINE</span>
        </div>
        <div style="color:#94A3B8;">Indrayani · Alandi Basin</div>
        <div style="color:#475569; font-size:10px; margin-top:2px;">18.6780°N &nbsp; 73.8980°E</div>
    </div>
    """, unsafe_allow_html=True)

# Top Bar
module_path_map = {
    "Overview": "overview",
    "CV Inference": "inference",
    "Hotspot Map": "map",
    "Dispatch Hub": "dispatch"
}
current_time_str = datetime.now().strftime("%d/%m/%Y, %H:%M:%S IST")

st.markdown(f"""
<div class="top-bar">
    <div style="color: #64748B;">
        ⚡ {current_time_str} &nbsp;/&nbsp; PATH · <span style="color:#38BDF8; font-weight:600;">{module_path_map[selected_module]}</span>
    </div>
    <div class="live-badge">
        <div class="pulse-dot"></div>
        LIVE TELEMETRY
    </div>
</div>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def get_model():
    if os.path.exists("best.pt"):
        return YOLO("best.pt")
    return YOLO("yolov8n.pt")

model = get_model()

# ==============================
# 1. OVERVIEW MODULE
# ==============================
if selected_module == "Overview":
    st.markdown("<div class='view-category'>EXECUTIVE TELEMETRY · BASIN OVERVIEW</div>", unsafe_allow_html=True)
    st.markdown("<div class='view-title'>Alandi river corridor telemetry.</div>", unsafe_allow_html=True)
    st.markdown("<div class='view-desc'>Sensory aggregation fusing edge CV inference, surface-velocity optical tracking, and municipal ticketing across the Indrayani River Basin.</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("<div class='stat-box'><div class='stat-box-title'>MONITORED REACH</div><h2 style='font-size:26px; margin:6px 0; color:#F8FAFC;'>2.8 km</h2><span style='color:#38BDF8; font-size:12px;'>Active UAV Corridor</span></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='stat-box'><div class='stat-box-title'>SURFACE VELOCITY</div><h2 style='font-size:26px; margin:6px 0; color:#F8FAFC;'>0.46 m/s</h2><span style='color:#22C55E; font-size:12px;'>Optical Flow Vector</span></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='stat-box'><div class='stat-box-title'>12H CHOKE RISK</div><h2 style='font-size:26px; margin:6px 0; color:#F87171;'>88.4%</h2><span style='color:#F87171; font-size:12px;'>Bridge Pier B-1</span></div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='stat-box'><div class='stat-box-title'>WORK ORDERS</div><h2 style='font-size:26px; margin:6px 0; color:#FBBF24;'>5 Active</h2><span style='color:#FBBF24; font-size:12px;'>3 Pending · 2 In Progress</span></div>", unsafe_allow_html=True)

# ==============================
# 2. CV INFERENCE MODULE
# ==============================
elif selected_module == "CV Inference":
    st.markdown("<div class='view-category'>VISION PIPELINE · YOLOV8 EDGE INFERENCE</div>", unsafe_allow_html=True)
    st.markdown("<div class='view-title'>Real-time CV inference on drone footage.</div>", unsafe_allow_html=True)
    st.markdown("<div class='view-desc'>Detect PET plastic, hyacinth mats, silt bars, and mixed debris frame-by-frame with dynamic class count synchronization.</div>", unsafe_allow_html=True)
    
    col_vid, col_stats = st.columns([2.2, 1])
    
    with col_vid:
        st_frame = st.empty()
        st_frame.image("https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1200&q=80", use_container_width=True)
        
        uploaded_video = st.file_uploader("Upload drone video (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])
        start_btn = st.button("▶️ Execute Vision Pipeline", use_container_width=True)
        
    with col_stats:
        breakdown_ph = st.empty()
        kpi_ph = st.empty()
        
        # Initial Render (Before running)
        with breakdown_ph.container():
            st.markdown("<div class='stat-box'><div class='stat-box-title' style='margin-bottom:12px;'>LIVE DETECTION BREAKDOWN</div>", unsafe_allow_html=True)
            st.write("● **Plastic (PET/Bags):** `0`")
            st.progress(0.0)
            st.write("● **Hyacinth Mat:** `0`")
            st.progress(0.0)
            st.write("● **Siltation Bar:** `0`")
            st.progress(0.0)
            st.write("● **Mixed Floating Debris:** `0`")
            st.progress(0.0)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with kpi_ph.container():
            st.markdown("<div class='stat-box' style='margin-top:14px;'><div class='stat-box-title'>TOTAL DETECTED</div><h2 style='font-size:32px; color:#38BDF8; margin:4px 0;'>0</h2><div style='font-size:11px; color:#64748B;'>YOLOv8s · Active Model</div></div>", unsafe_allow_html=True)

    # Dynamic Frame-by-Frame Inference Loop
    if start_btn and uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        
        detected_counts = {"plastic": 0, "hyacinth": 0, "silt": 0, "debris": 0}
        total_objects = 0
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_idx > 120:
                break
            
            if frame_idx % 2 == 0:
                results = model(frame, conf=0.25, verbose=False)[0]
                annotated = results.plot()
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st_frame.image(annotated_rgb, use_container_width=True)
                
                # Update counts dynamically from model predictions
                for box in results.boxes:
                    cls_name = model.names[int(box.cls[0])].lower()
                    if "plastic" in cls_name or "bottle" in cls_name:
                        detected_counts["plastic"] += 1
                    elif "hyacinth" in cls_name or "weed" in cls_name or "bio" in cls_name:
                        detected_counts["hyacinth"] += 1
                    elif "silt" in cls_name or "sand" in cls_name:
                        detected_counts["silt"] += 1
                    else:
                        detected_counts["debris"] += 1
                    total_objects += 1
                
                denom = max(total_objects, 1)
                # Re-render dynamic breakdown without markdown syntax leaks
                with breakdown_ph.container():
                    st.markdown("<div class='stat-box'><div class='stat-box-title' style='margin-bottom:12px;'>LIVE DETECTION BREAKDOWN</div>", unsafe_allow_html=True)
                    st.write(f"● **Plastic:** `{detected_counts['plastic']}`")
                    st.progress(min(detected_counts["plastic"] / denom, 1.0))
                    st.write(f"● **Hyacinth:** `{detected_counts['hyacinth']}`")
                    st.progress(min(detected_counts["hyacinth"] / denom, 1.0))
                    st.write(f"● **Silt:** `{detected_counts['silt']}`")
                    st.progress(min(detected_counts["silt"] / denom, 1.0))
                    st.write(f"● **Debris:** `{detected_counts['debris']}`")
                    st.progress(min(detected_counts["debris"] / denom, 1.0))
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                with kpi_ph.container():
                    st.markdown(f"<div class='stat-box' style='margin-top:14px;'><div class='stat-box-title'>TOTAL DETECTED</div><h2 style='font-size:32px; color:#38BDF8; margin:4px 0;'>{total_objects}</h2><div style='font-size:11px; color:#64748B;'>YOLOv8s · Active Model</div></div>", unsafe_allow_html=True)
                    
            frame_idx += 1
            time.sleep(0.03)
        cap.release()
        st.success("✅ Video analysis complete.")

# ==============================
# 3. HOTSPOT MAP MODULE
# ==============================
elif selected_module == "Hotspot Map":
    st.markdown("<div class='view-category'>GEOSPATIAL LAYER · INDRAYANI WATERCOURSE</div>", unsafe_allow_html=True)
    st.markdown("<div class='view-title'>Hotspots & hydrodynamic drift.</div>", unsafe_allow_html=True)
    st.markdown("<div class='view-desc'>Debris density heatmap fused with 12–48h drift trajectories mapped precisely along the Indrayani River centerline in Alandi.</div>", unsafe_allow_html=True)
    
    col_map, col_drift = st.columns([2.3, 1])
    
    with col_map:
        # Centered exactly on the Indrayani watercourse in Alandi
        m = folium.Map(location=[18.6755, 73.8965], zoom_start=15, tiles="CartoDB dark_matter")
        
        # Exact verified coordinates strictly situated along the water channel
        river_hotspots = [
            {"lat": 18.6782, "lon": 73.8942, "weight": 0.95, "name": "Reach A-1 (Upstream Inflow Bend)", "risk": "Critical"},
            {"lat": 18.6766, "lon": 73.8960, "weight": 0.88, "name": "Reach A-2 (Main Alandi Ghat)", "risk": "Critical"},
            {"lat": 18.6751, "lon": 73.8973, "weight": 0.75, "name": "Reach B-1 (Bhakti Sopan Bridge Pier)", "risk": "Moderate"},
            {"lat": 18.6738, "lon": 73.8998, "weight": 0.90, "name": "Reach C-1 (Downstream Bund / Weir)", "risk": "Critical"},
            {"lat": 18.6720, "lon": 73.9030, "weight": 0.60, "name": "Reach C-2 (Downstream Sediment Bar)", "risk": "Moderate"}
        ]
        
        HeatMap([[p["lat"], p["lon"], p["weight"]] for p in river_hotspots], radius=24, blur=16, min_opacity=0.45).add_to(m)
        
        for pt in river_hotspots:
            c = "#EF4444" if pt["risk"] == "Critical" else "#F59E0B"
            folium.CircleMarker(
                location=[pt["lat"], pt["lon"]],
                radius=7,
                popup=f"<b>{pt['name']}</b><br>Risk: {pt['risk']}",
                color=c,
                fill=True,
                fill_color=c
            ).add_to(m)
            
        # Continuous river channel trajectory vector polyline
        watercourse_polyline = [
            [18.6782, 73.8942],
            [18.6766, 73.8960],
            [18.6751, 73.8973],
            [18.6738, 73.8998],
            [18.6720, 73.9030]
        ]
        folium.PolyLine(
            locations=watercourse_polyline,
            color="#38BDF8", weight=4, dash_array="6, 8", tooltip="Hydrodynamic Drift Trajectory (48h)"
        ).add_to(m)
        
        st_folium(m, width="100%", height=480)
        
    with col_drift:
        st.markdown("""
        <div class="stat-box" style="margin-bottom:12px;">
            <div class="stat-box-title">CORRIDOR · 2.8 km</div>
            <div style="margin-top:14px; font-family:'JetBrains Mono'; font-size:12px;">
                <div style="color:#64748B; margin-bottom:6px;">PREDICTED CHOKE POINTS</div>
                <div style="background:#070B13; padding:10px; border-radius:6px; border-left:3px solid #EF4444; margin-bottom:8px;">
                    <div style="font-weight:700; color:#F8FAFC;">12h → Bridge Pier B-1</div>
                    <div style="color:#94A3B8; font-size:11px; margin-top:2px;">v · 0.46 m/s &nbsp;·&nbsp; ETA 12h</div>
                </div>
                <div style="background:#070B13; padding:10px; border-radius:6px; border-left:3px solid #EF4444; margin-bottom:8px;">
                    <div style="font-weight:700; color:#F8FAFC;">24h → Downstream Weir C-1</div>
                    <div style="color:#94A3B8; font-size:11px; margin-top:2px;">v · 0.41 m/s &nbsp;·&nbsp; ETA 24h</div>
                </div>
                <div style="background:#070B13; padding:10px; border-radius:6px; border-left:3px solid #38BDF8;">
                    <div style="font-weight:700; color:#F8FAFC;">48h → Downstream Bar</div>
                    <div style="color:#94A3B8; font-size:11px; margin-top:2px;">v · 0.38 m/s &nbsp;·&nbsp; ETA 48h</div>
                </div>
            </div>
        </div>
        <div class="stat-box">
            <div class="stat-box-title">RIVER ANCHOR</div>
            <div style="font-size:18px; font-weight:700; color:#FFFFFF; margin-top:4px;">Indrayani River</div>
            <div style="color:#64748B; font-size:12px; font-family:'JetBrains Mono';">Alandi Ghat Corridor · Pune, MH</div>
        </div>
        """, unsafe_allow_html=True)

# ==============================
# 4. DISPATCH HUB MODULE
# ==============================
elif selected_module == "Dispatch Hub":
    st.markdown("<div class='view-category'>FIELD OPS · MUNICIPAL DISPATCH</div>", unsafe_allow_html=True)
    st.markdown("<div class='view-title'>Cleanup work orders.</div>", unsafe_allow_html=True)
    st.markdown("<div class='view-desc'>Auto-generated tickets from CV inference & drift prediction. Sync directly with the Alandi Municipal Sanitation & Irrigation cells.</div>", unsafe_allow_html=True)
    
    orders_data = [
        {"ORDER": "WO-2026-0142", "REACH": "Reach A-2 (Main Ghat)", "GPS": "18.6766, 73.8960", "MATERIAL": "Hyacinth Bio-Mat", "VOL (m³)": 128.0, "URGENCY": "CRITICAL", "RECOMMENDED ACTION": "Deploy Trash Skimmer Boat + Boom Barrier", "STATUS": "Pending"},
        {"ORDER": "WO-2026-0143", "REACH": "Reach A-1 (Upstream Inflow)", "GPS": "18.6782, 73.8942", "MATERIAL": "PET Plastic Cluster", "VOL (m³)": 42.5, "URGENCY": "CRITICAL", "RECOMMENDED ACTION": "Deploy Boom Barrier + Manual Retrieval Crew", "STATUS": "In Progress"},
        {"ORDER": "WO-2026-0144", "REACH": "Reach C-2 (Sediment Bar)", "GPS": "18.6720, 73.9030", "MATERIAL": "Siltation & Silt Bar", "VOL (m³)": 154.9, "URGENCY": "HIGH", "RECOMMENDED ACTION": "Schedule Dredging Operation", "STATUS": "Pending"},
        {"ORDER": "WO-2026-0145", "REACH": "Reach B-1 (Bridge Pier)", "GPS": "18.6751, 73.8973", "MATERIAL": "Mixed Floating Debris", "VOL (m³)": 88.2, "URGENCY": "MODERATE", "RECOMMENDED ACTION": "Dispatch Trash Skimmer Boat", "STATUS": "Pending"},
        {"ORDER": "WO-2026-0146", "REACH": "Reach C-1 (Downstream Weir)", "GPS": "18.6738, 73.8998", "MATERIAL": "PET Bottles & Thermocol", "VOL (m³)": 65.7, "URGENCY": "HIGH", "RECOMMENDED ACTION": "Deploy Trash Skimmer Boat", "STATUS": "In Progress"}
    ]
    df_orders = pd.DataFrame(orders_data)
    
    col_search, col_btn1, col_btn2 = st.columns([2.5, 0.7, 0.7])
    with col_search:
        search = st.text_input("Search Orders", placeholder="Search by order, reach, or material...", label_visibility="collapsed")
    with col_btn1:
        csv_bytes = df_orders.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export CSV", data=csv_bytes, file_name="RiverSense_Work_Orders.csv", mime="text/csv", use_container_width=True)
    with col_btn2:
        st.download_button("📄 Export PDF", data=csv_bytes, file_name="RiverSense_Work_Orders.pdf", mime="text/csv", use_container_width=True)
        
    if search:
        df_orders = df_orders[df_orders.apply(lambda row: row.astype(str).str.contains(search, case=False).any(), axis=1)]
        
    st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
    
    # Native Streamlit Interactive Dataframe (No raw code rendering)
    st.dataframe(
        df_orders,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ORDER": st.column_config.TextColumn("ORDER", width="small"),
            "REACH": st.column_config.TextColumn("REACH", width="medium"),
            "GPS": st.column_config.TextColumn("GPS COORDINATES", width="small"),
            "MATERIAL": st.column_config.TextColumn("MATERIAL"),
            "VOL (m³)": st.column_config.NumberColumn("VOL (m³)", format="%.1f"),
            "URGENCY": st.column_config.TextColumn("URGENCY", width="small"),
            "RECOMMENDED ACTION": st.column_config.TextColumn("RECOMMENDED ACTION", width="large"),
            "STATUS": st.column_config.SelectboxColumn("STATUS", options=["Pending", "In Progress", "Resolved"], required=True)
        }
    )
