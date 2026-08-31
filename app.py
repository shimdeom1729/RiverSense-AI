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

# Page Setup
st.set_page_config(
    page_title="RiverSense AI | Command Center",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Command-Center Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');
    
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
        background-color: #070B13 !important;
        color: #E2E8F0 !important;
    }
    
    /* Hide Default Header/Footer */
    header, footer, #MainMenu { visibility: hidden !important; height: 0px !important; }
    .block-container { padding-top: 1rem !important; padding-bottom: 2rem !important; }
    
    /* Top Bar Styling */
    .top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 18px;
        background-color: #0B111E;
        border-bottom: 1px solid #1E293B;
        border-radius: 8px;
        margin-bottom: 20px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
    }
    .top-bar-path { color: #64748B; }
    .top-bar-path span { color: #38BDF8; font-weight: 600; }
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
        letter-spacing: 0.5px;
    }
    .pulse-dot {
        width: 7px;
        height: 7px;
        background-color: #EF4444;
        border-radius: 50%;
        box-shadow: 0 0 8px #EF4444;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0B111E !important;
        border-right: 1px solid #1E293B !important;
    }
    .brand-box {
        padding: 10px 0 20px 0;
        border-bottom: 1px solid #1E293B;
        margin-bottom: 20px;
    }
    .brand-title {
        font-size: 20px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #F8FAFC;
        margin: 0;
    }
    .brand-sub {
        font-size: 10px;
        color: #64748B;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-top: 2px;
    }
    .sys-status-box {
        margin-top: 40px;
        padding: 14px;
        background: #080D18;
        border-radius: 8px;
        border: 1px solid #1E293B;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
    }
    
    /* Content Headings */
    .view-category {
        color: #38BDF8;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .view-title {
        font-size: 32px;
        font-weight: 800;
        letter-spacing: -1px;
        color: #FFFFFF;
        margin: 0 0 8px 0;
    }
    .view-desc {
        color: #94A3B8;
        font-size: 14px;
        line-height: 1.5;
        margin-bottom: 24px;
        max-width: 850px;
    }
    
    /* Metric & Breakdown Cards */
    .stat-card {
        background: #0B111E;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 18px;
    }
    .card-label {
        color: #64748B;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    /* Progress Bars for CV Inference */
    .breakdown-row {
        margin-bottom: 14px;
    }
    .breakdown-meta {
        display: flex;
        justify-content: space-between;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 5px;
    }
    .progress-track {
        height: 6px;
        background: #1E293B;
        border-radius: 4px;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        border-radius: 4px;
    }
    
    /* Badges */
    .badge-critical { background: rgba(239, 68, 68, 0.2); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.4); padding: 3px 8px; border-radius: 4px; font-weight: 700; font-size: 11px; }
    .badge-high { background: rgba(245, 158, 11, 0.2); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.4); padding: 3px 8px; border-radius: 4px; font-weight: 700; font-size: 11px; }
    .badge-moderate { background: rgba(56, 189, 248, 0.2); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.4); padding: 3px 8px; border-radius: 4px; font-weight: 700; font-size: 11px; }
    
    /* Table Styling */
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }
    .custom-table th {
        text-align: left;
        padding: 12px 14px;
        background: #0B111E;
        color: #64748B;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        border-bottom: 1px solid #1E293B;
    }
    .custom-table td {
        padding: 14px;
        border-bottom: 1px solid #131D31;
        color: #E2E8F0;
    }
    .custom-table tr:hover {
        background-color: #0F172A;
    }
    .order-id {
        color: #38BDF8;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.markdown("""
    <div class="brand-box">
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="background:#0284C7; border-radius:6px; padding:6px; display:flex;">🌊</div>
            <div>
                <div class="brand-title">RiverSense</div>
                <div class="brand-sub">AI · Command Center</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='card-label' style='margin-bottom:8px;'>MODULES</div>", unsafe_allow_html=True)
    selected_module = st.radio(
        "Navigation",
        ["Overview", "CV Inference", "Hotspot Map", "Dispatch Hub"],
        label_visibility="collapsed"
    )
    
    st.markdown("""
    <div class="sys-status-box">
        <div style="display:flex; align-items:center; gap:6px; margin-bottom:4px;">
            <div class="pulse-dot"></div>
            <span style="color:#22C55E; font-weight:700;">SYS · ONLINE</span>
        </div>
        <div style="color:#94A3B8;">Indrayani · Alandi Sector</div>
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
    <div class="top-bar-path">
        ⚡ {current_time_str} &nbsp; / &nbsp; PATH · <span>{module_path_map[selected_module]}</span>
    </div>
    <div class="live-badge">
        <div class="pulse-dot"></div>
        LIVE TELEMETRY
    </div>
</div>
""", unsafe_allow_html=True)

# Load Model Checkpoint
@st.cache_resource
def load_model():
    if os.path.exists("best.pt"):
        return YOLO("best.pt")
    return YOLO("yolov8n.pt")

model = load_model()

# MODULE 1: OVERVIEW
if selected_module == "Overview":
    st.markdown("<div class='view-category'>EXECUTIVE TELEMETRY · REACH OVERVIEW</div>", unsafe_allow_html=True)
    st.markdown("<div class='view-title'>Alandi river corridor telemetry.</div>", unsafe_allow_html=True)
    st.markdown("<div class='view-desc'>Real-time sensory aggregation fusing edge CV inference, surface-velocity optical tracking, and municipal response ticketing across the Indrayani River Basin.</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="card-label">MONITORED REACH</div>
            <h2 style="font-size:26px; font-weight:800; margin:8px 0; color:#F8FAFC;">2.8 km</h2>
            <div style="color:#38BDF8; font-size:12px; font-family:'JetBrains Mono';">Active Drone Corridor</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="card-label">MEAN SURFACE SPEED</div>
            <h2 style="font-size:26px; font-weight:800; margin:8px 0; color:#F8FAFC;">0.46 m/s</h2>
            <div style="color:#22C55E; font-size:12px; font-family:'JetBrains Mono';">Optical Flow Vector</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="stat-card">
            <div class="card-label">12H CHOKE RISK</div>
            <h2 style="font-size:26px; font-weight:800; margin:8px 0; color:#F87171;">88.4%</h2>
            <div style="color:#F87171; font-size:12px; font-family:'JetBrains Mono';">Bridge Pier #2</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="stat-card">
            <div class="card-label">ACTIVE WORK ORDERS</div>
            <h2 style="font-size:26px; font-weight:800; margin:8px 0; color:#FBBF24;">6 Orders</h2>
            <div style="color:#FBBF24; font-size:12px; font-family:'JetBrains Mono';">3 Pending · 3 In Progress</div>
        </div>
        """, unsafe_allow_html=True)

# MODULE 2: CV INFERENCE
elif selected_module == "CV Inference":
    st.markdown("<div class='view-category'>VISION PIPELINE · YOLOV8 SIMULATOR</div>", unsafe_allow_html=True)
    st.markdown("<div class='view-title'>Real-time CV inference on drone footage.</div>", unsafe_allow_html=True)
    st.markdown("<div class='view-desc'>Detect PET plastic, hyacinth mats, silt bars and mixed debris frame-by-frame. Latency and FPS mirror an edge-deployed T4 inference node.</div>", unsafe_allow_html=True)
    
    col_vid, col_breakdown = st.columns([2.2, 1])
    
    with col_vid:
        st.markdown("""
        <div style="display:flex; justify-content:space-between; align-items:center; background:#0B111E; padding:8px 14px; border-radius:8px 8px 0 0; border:1px solid #1E293B; border-bottom:none;">
            <div style="display:flex; align-items:center; gap:8px;">
                <span class="live-badge" style="padding:2px 8px;"><div class="pulse-dot"></div> LIVE · YOLOv8s</span>
            </div>
            <div style="font-family:'JetBrains Mono'; font-size:12px; color:#94A3B8;">60 FPS &nbsp;·&nbsp; 25 ms</div>
        </div>
        """, unsafe_allow_html=True)
        
        st_frame = st.empty()
        # Default placeholder image matching drone feed
        st_frame.image("https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1200&q=80", use_container_width=True)
        
        st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload new footage (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"], label_visibility="collapsed")
        
        btn_col1, btn_col2 = st.columns([1, 1])
        start_analysis = btn_col1.button("▶️ Run Vision Pipeline", use_container_width=True)
        
    with col_breakdown:
        st.markdown("""
        <div class="stat-card" style="margin-bottom:14px;">
            <div class="card-label" style="margin-bottom:14px;">LIVE DETECTION BREAKDOWN</div>
            
            <div class="breakdown-row">
                <div class="breakdown-meta">
                    <span style="color:#F87171;">● PLASTIC</span>
                    <span id="cnt-plastic">270</span>
                </div>
                <div class="progress-track"><div class="progress-fill" style="width:75%; background:#EF4444;"></div></div>
            </div>
            
            <div class="breakdown-row">
                <div class="breakdown-meta">
                    <span style="color:#4ADE80;">● HYACINTH</span>
                    <span id="cnt-hyacinth">135</span>
                </div>
                <div class="progress-track"><div class="progress-fill" style="width:40%; background:#22C55E;"></div></div>
            </div>
            
            <div class="breakdown-row">
                <div class="breakdown-meta">
                    <span style="color:#FBBF24;">● SILT</span>
                    <span id="cnt-silt">0</span>
                </div>
                <div class="progress-track"><div class="progress-fill" style="width:5%; background:#F59E0B;"></div></div>
            </div>
            
            <div class="breakdown-row">
                <div class="breakdown-meta">
                    <span style="color:#38BDF8;">● DEBRIS</span>
                    <span id="cnt-debris">135</span>
                </div>
                <div class="progress-track"><div class="progress-fill" style="width:40%; background:#0284C7;"></div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="stat-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div class="card-label">TOTAL OBJECTS</div>
                    <div style="font-size:32px; font-weight:800; color:#FFFFFF; margin-top:4px;">540</div>
                </div>
                <div style="text-align:right;">
                    <div class="card-label">MODEL</div>
                    <div style="font-size:13px; font-family:'JetBrains Mono'; color:#38BDF8; font-weight:600; margin-top:4px;">YOLOv8s · fine-tuned</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Real Inference Loop if Video Uploaded
    if start_analysis and uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_count > 150:
                break
            if frame_count % 3 == 0:
                results = model(frame, conf=0.25, verbose=False)[0]
                annotated = results.plot()
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st_frame.image(annotated_rgb, use_container_width=True)
            frame_count += 1
        cap.release()

# MODULE 3: HOTSPOT MAP
elif selected_module == "Hotspot Map":
    st.markdown("<div class='view-category'>GEOSPATIAL LAYER · INDRAYANI RIVER BASIN</div>", unsafe_allow_html=True)
    st.markdown("<div class='view-title'>Hotspots & hydrodynamic drift.</div>", unsafe_allow_html=True)
    st.markdown("<div class='view-desc'>Debris density heatmap fused with 12–48h drift trajectories predicted from surface-flow optical velocity toward downstream weirs and bridge piers.</div>", unsafe_allow_html=True)
    
    col_map, col_drift = st.columns([2.3, 1])
    
    with col_map:
        # Layer toggles & Legend Bar
        st.markdown("""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
            <div style="display:flex; gap:8px;">
                <span class="live-badge" style="background:#0F172A; color:#38BDF8; border-color:#1E293B;">Hotspot Pins</span>
                <span class="live-badge" style="background:#0F172A; color:#38BDF8; border-color:#1E293B;">Drift Vectors</span>
                <span class="live-badge" style="background:#0F172A; color:#38BDF8; border-color:#1E293B;">Density Heatmap</span>
            </div>
            <div style="font-family:'JetBrains Mono'; font-size:12px;">
                <span style="color:#F87171;">● Critical · 3</span> &nbsp;
                <span style="color:#FBBF24;">● Moderate · 3</span> &nbsp;
                <span style="color:#22C55E;">● Cleared · 1</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Dark Matter Map Centered on Alandi/Indrayani
        m = folium.Map(location=[18.6780, 73.8980], zoom_start=14, tiles="CartoDB dark_matter")
        
        hotspots = [
            {"lat": 18.6795, "lon": 73.8930, "weight": 0.95, "desc": "Reach A-1 (Alandi Ghat)", "type": "Critical"},
            {"lat": 18.6772, "lon": 73.8965, "weight": 0.90, "desc": "Reach A-2 (Temple Bend)", "type": "Critical"},
            {"lat": 18.6760, "lon": 73.9010, "weight": 0.65, "desc": "Reach B-1 (Bridge Pier)", "type": "Moderate"},
            {"lat": 18.6740, "lon": 73.9050, "weight": 0.85, "desc": "Reach C-1 (Downstream Weir)", "type": "Critical"},
            {"lat": 18.6725, "lon": 73.9100, "weight": 0.60, "desc": "Reach C-2 (Sediment Bar)", "type": "Moderate"}
        ]
        
        HeatMap([[p["lat"], p["lon"], p["weight"]] for p in hotspots], radius=25, blur=18, min_opacity=0.4).add_to(m)
        
        for pt in hotspots:
            color = "#EF4444" if pt["type"] == "Critical" else "#F59E0B"
            folium.CircleMarker(
                location=[pt["lat"], pt["lon"]],
                radius=7,
                popup=f"<b>{pt['desc']}</b><br>Risk: {pt['type']}",
                color=color,
                fill=True,
                fill_color=color
            ).add_to(m)
            
        # Predicted Hydrodynamic Trajectory Polyline
        folium.PolyLine(
            locations=[[18.6795, 73.8930], [18.6772, 73.8965], [18.6760, 73.9010], [18.6740, 73.9050], [18.6725, 73.9100]],
            color="#38BDF8", weight=3, dash_array="6, 8", tooltip="Hydrodynamic Drift Trajectory (48h)"
        ).add_to(m)
        
        st_folium(m, width="100%", height=460)
        
    with col_drift:
        st.markdown("""
        <div class="stat-card" style="margin-bottom:12px;">
            <div class="card-label">CORRIDOR · 2.8 km</div>
            <div style="margin-top:14px; font-family:'JetBrains Mono'; font-size:12px;">
                <div style="color:#64748B; margin-bottom:4px;">DRIFT PREDICTIONS</div>
                <div style="background:#070B13; padding:10px; border-radius:6px; border-left:3px solid #38BDF8; margin-bottom:8px;">
                    <div style="font-weight:700; color:#F8FAFC;">12h drift → Bridge Pier B-1</div>
                    <div style="color:#94A3B8; font-size:11px; margin-top:2px;">v · 0.46 m/s &nbsp;·&nbsp; ETA 12h</div>
                </div>
                <div style="background:#070B13; padding:10px; border-radius:6px; border-left:3px solid #38BDF8; margin-bottom:8px;">
                    <div style="font-weight:700; color:#F8FAFC;">24h drift → Weir C-1</div>
                    <div style="color:#94A3B8; font-size:11px; margin-top:2px;">v · 0.41 m/s &nbsp;·&nbsp; ETA 24h</div>
                </div>
                <div style="background:#070B13; padding:10px; border-radius:6px; border-left:3px solid #38BDF8;">
                    <div style="font-weight:700; color:#F8FAFC;">48h drift → Downstream Bar</div>
                    <div style="color:#94A3B8; font-size:11px; margin-top:2px;">v · 0.38 m/s &nbsp;·&nbsp; ETA 48h</div>
                </div>
            </div>
        </div>
        
        <div class="stat-card">
            <div class="card-label">RIVER ANCHOR</div>
            <div style="font-size:18px; font-weight:700; color:#FFFFFF; margin-top:4px;">Indrayani</div>
            <div style="color:#64748B; font-size:12px; font-family:'JetBrains Mono';">Alandi Ghat · Pune, MH</div>
        </div>
        """, unsafe_allow_html=True)

# MODULE 4: DISPATCH HUB
elif selected_module == "Dispatch Hub":
    st.markdown("<div class='view-category'>FIELD OPS · MUNICIPAL DISPATCH</div>", unsafe_allow_html=True)
    st.markdown("<div class='view-title'>Cleanup work orders.</div>", unsafe_allow_html=True)
    st.markdown("<div class='view-desc'>Auto-generated tickets from CV inference & drift prediction. Sync directly with the Alandi Municipal Sanitation & Irrigation cells.</div>", unsafe_allow_html=True)
    
    # Export Actions & Filters Bar
    col_filter, col_export = st.columns([2.5, 1])
    with col_filter:
        search_query = st.text_input("🔍 Filters", placeholder="Search order, reach, material...", label_visibility="collapsed")
    with col_export:
        btn_c1, btn_c2 = st.columns(2)
        btn_c1.button("📥 Export CSV", use_container_width=True)
        btn_c2.button("📄 Export PDF", use_container_width=True)
        
    orders_data = [
        {"order": "WO-2026-0142", "reach": "Reach A-2 (Temple Bend)", "gps": "18.6772, 73.8965", "material": "Hyacinth Mat", "vol": 128.0, "urgency": "CRITICAL", "action": "Deploy Trash Skimmer Boat + Boom Barrier", "status": "Pending"},
        {"order": "WO-2026-0143", "reach": "Reach A-1 (Alandi Ghat)", "gps": "18.6795, 73.8930", "material": "PET Plastic Cluster", "vol": 42.5, "urgency": "CRITICAL", "action": "Deploy Boom Barrier + Manual Retrieval Crew", "status": "In Progress"},
        {"order": "WO-2026-0144", "reach": "Reach C-2 (Sediment Bar)", "gps": "18.6725, 73.9100", "material": "Siltation", "vol": 154.9, "urgency": "HIGH", "action": "Schedule Dredging Operation", "status": "Pending"},
        {"order": "WO-2026-0145", "reach": "Reach B-1 (Bridge Pier)", "gps": "18.6760, 73.9010", "material": "Mixed Debris", "vol": 88.2, "urgency": "MODERATE", "action": "Dispatch Trash Skimmer Boat", "status": "Pending"},
        {"order": "WO-2026-0146", "reach": "Reach C-1 (Downstream Weir)", "gps": "18.6740, 73.9050", "material": "PET Plastic & Thermocol", "vol": 65.7, "urgency": "HIGH", "action": "Deploy Trash Skimmer Boat", "status": "In Progress"}
    ]
    
    # Render Work Orders Table
    table_html = """
    <div style="background:#0B111E; border:1px solid #1E293B; border-radius:8px; overflow:hidden; margin-top:16px;">
    <table class="custom-table">
        <thead>
            <tr>
                <th>ORDER</th>
                <th>REACH</th>
                <th>GPS</th>
                <th>MATERIAL</th>
                <th>VOL M³</th>
                <th>URGENCY</th>
                <th>RECOMMENDED ACTION</th>
                <th>STATUS</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for row in orders_data:
        urg_badge = f"<span class='badge-{row['urgency'].lower()}'>{row['urgency']}</span>"
        status_color = "#F87171" if row['status'] == "Pending" else "#FBBF24"
        table_html += f"""
        <tr>
            <td class="order-id">{row['order']}</td>
            <td style="font-weight:600;">{row['reach']}</td>
            <td style="font-family:'JetBrains Mono'; font-size:12px; color:#94A3B8;">{row['gps']}</td>
            <td>{row['material']}</td>
            <td style="font-family:'JetBrains Mono'; font-weight:700;">{row['vol']}</td>
            <td>{urg_badge}</td>
            <td style="font-size:12px; color:#CBD5E1;">{row['action']}</td>
            <td><span style="color:{status_color}; font-family:'JetBrains Mono'; font-size:12px; font-weight:700;">{row['status']} ▾</span></td>
        </tr>
        """
        
    table_html += "</tbody></table></div>"
    st.markdown(table_html, unsafe_allow_html=True)
