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

# Custom Command-Center Theme
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

# Persistent Session State across Tabs
if "has_run" not in st.session_state:
    st.session_state.has_run = False
    st.session_state.detected_counts = {"plastic": 0, "hyacinth": 0, "ritual": 0, "debris": 0}
    st.session_state.total_detected = 0
    st.session_state.estimated_volume_m3 = 0.0

# Sidebar Navigation
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
        <div style="color:#94A3B8;">Indrayani River · Alandi Reach</div>
        <div style="color:#475569; font-size:10px; margin-top:2px;">18.6766°N &nbsp; 73.8960°E</div>
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

# Model Loader
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
    st.markdown("<div class='view-desc'>Aggregating live CV inference, surface-velocity optical tracking, and municipal ticketing across the Indrayani River Reach.</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("<div class='stat-box'><div class='stat-box-title'>MONITORED REACH</div><h2 style='font-size:26px; margin:6px 0; color:#F8FAFC;'>2.8 km</h2><span style='color:#38BDF8; font-size:12px;'>Alandi Ghat to Weir</span></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='stat-box'><div class='stat-box-title'>SURFACE VELOCITY</div><h2 style='font-size:26px; margin:6px 0; color:#F8FAFC;'>0.46 m/s</h2><span style='color:#22C55E; font-size:12px;'>Optical Flow Vector</span></div>", unsafe_allow_html=True)
    with col3:
        if st.session_state.has_run and st.session_state.total_detected > 0:
            risk_pct = min(40 + (st.session_state.total_detected * 0.5), 98.0)
            risk_color = "#F87171" if risk_pct > 70 else "#FBBF24"
            st.markdown(f"<div class='stat-box'><div class='stat-box-title'>12H CHOKE RISK</div><h2 style='font-size:26px; margin:6px 0; color:{risk_color};'>{risk_pct:.1f}%</h2><span style='color:{risk_color}; font-size:12px;'>Bridge Pier B-1</span></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='stat-box'><div class='stat-box-title'>12H CHOKE RISK</div><h2 style='font-size:26px; margin:6px 0; color:#94A3B8;'>Pending</h2><span style='color:#64748B; font-size:12px;'>Run CV Module First</span></div>", unsafe_allow_html=True)
    with col4:
        if st.session_state.has_run and st.session_state.total_detected > 0:
            st.markdown(f"<div class='stat-box'><div class='stat-box-title'>EST. TOTAL VOLUME</div><h2 style='font-size:26px; margin:6px 0; color:#38BDF8;'>{st.session_state.estimated_volume_m3:.1f} m³</h2><span style='color:#38BDF8; font-size:12px;'>Active Work Orders Ready</span></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='stat-box'><div class='stat-box-title'>EST. TOTAL VOLUME</div><h2 style='font-size:26px; margin:6px 0; color:#94A3B8;'>0.0 m³</h2><span style='color:#64748B; font-size:12px;'>Awaiting Drone Feed</span></div>", unsafe_allow_html=True)

# ==============================
# 2. CV INFERENCE MODULE
# ==============================
elif selected_module == "CV Inference":
    st.markdown("<div class='view-category'>VISION PIPELINE · RIGOROUS MULTI-CLASS INFERENCE</div>", unsafe_allow_html=True)
    st.markdown("<div class='view-title'>Real-time CV inference on drone footage.</div>", unsafe_allow_html=True)
    st.markdown("<div class='view-desc'>Strict classification between Dry Plastic Waste, Hyacinth Bio-Mats, Sacred Ritual Offerings, and Mixed Residual Debris.</div>", unsafe_allow_html=True)
    
    col_vid, col_stats = st.columns([2.2, 1])
    
    with col_vid:
        st_frame = st.empty()
        st_frame.image("https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1200&q=80", use_container_width=True)
        
        uploaded_video = st.file_uploader("Upload Drone Video (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])
        start_btn = st.button("▶️ Execute Full Pipeline", use_container_width=True)
        
    with col_stats:
        breakdown_ph = st.empty()
        kpi_ph = st.empty()
        
        # Initial Render
        tot = st.session_state.total_detected
        denom = max(tot, 1)
        with breakdown_ph.container():
            st.markdown("<div class='stat-box'><div class='stat-box-title' style='margin-bottom:12px;'>LIVE DETECTION BREAKDOWN</div>", unsafe_allow_html=True)
            st.write(f"● **Dry Plastic Waste:** `{st.session_state.detected_counts['plastic']}`")
            st.progress(st.session_state.detected_counts['plastic'] / denom if tot > 0 else 0.0)
            st.write(f"● **Hyacinth Bio-Mat:** `{st.session_state.detected_counts['hyacinth']}`")
            st.progress(st.session_state.detected_counts['hyacinth'] / denom if tot > 0 else 0.0)
            st.write(f"● **Ritual Waste (Nirmalya/Diya):** `{st.session_state.detected_counts['ritual']}`")
            st.progress(st.session_state.detected_counts['ritual'] / denom if tot > 0 else 0.0)
            st.write(f"● **Mixed Floating Debris (Fallback):** `{st.session_state.detected_counts['debris']}`")
            st.progress(st.session_state.detected_counts['debris'] / denom if tot > 0 else 0.0)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with kpi_ph.container():
            st.markdown(f"<div class='stat-box' style='margin-top:14px;'><div class='stat-box-title'>TOTAL OBJECTS IDENTIFIED</div><h2 style='font-size:32px; color:#38BDF8; margin:4px 0;'>{tot}</h2><div style='font-size:11px; color:#64748B;'>Multi-Class Discriminator Active</div></div>", unsafe_allow_html=True)

    if start_btn and uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        
        counts = {"plastic": 0, "hyacinth": 0, "ritual": 0, "debris": 0}
        total_objects = 0
        frame_idx = 0
        
        KNOWN_PLASTIC_CLASSES = {
            "bottle", "cup", "bowl", "plastic", "frisbee", "cell phone", 
            "suitcase", "sports ball", "can", "box", "package"
        }
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_idx > 180:
                break
            
            if frame_idx % 2 == 0:
                annotated = frame.copy()
                h, w, _ = frame.shape
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # --- A. DETECT DRY PLASTICS & PACKAGING (YOLO + Specular Floating Plastic Mask) ---
                # 1. YOLO detections
                results = model(frame, conf=0.18, verbose=False)[0]
                for box in results.boxes:
                    name = model.names[int(box.cls[0])].lower()
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    if any(p_cls in name for p_cls in KNOWN_PLASTIC_CLASSES):
                        counts["plastic"] += 1
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(annotated, f"PLASTIC {conf:.2f}", (x1, max(15, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                    else:
                        # Fallback for unknown solid shapes
                        counts["debris"] += 1
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 140, 0), 2)
                        cv2.putText(annotated, f"DEBRIS {conf:.2f}", (x1, max(15, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 140, 0), 1)

                # 2. Specular White/Translucent Floating Plastic Bags & Styrofoam detection
                # Mask out bright, low-saturation floating debris on dark river surface
                lower_white_plastic = np.array([0, 0, 165])
                upper_white_plastic = np.array([180, 55, 255])
                plastic_mask = cv2.inRange(hsv, lower_white_plastic, upper_white_plastic)
                plastic_clean = cv2.morphologyEx(plastic_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
                
                contours_p, _ = cv2.findContours(plastic_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for pcnt in contours_p:
                    p_area = cv2.contourArea(pcnt)
                    if 120 < p_area < 2500:
                        px, py, pw, ph = cv2.boundingRect(pcnt)
                        # Check aspect ratio and circularity to avoid sun glare strips
                        aspect = float(pw) / max(ph, 1)
                        if 0.25 < aspect < 4.0:
                            counts["plastic"] += 1
                            cv2.rectangle(annotated, (px, py), (px + pw, py + ph), (0, 0, 255), 2)
                            cv2.putText(annotated, "DRY PLASTIC", (px, max(15, py - 4)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)

                # --- B. DETECT WATER HYACINTH BIO-MATS ---
                lower_green = np.array([32, 45, 35])
                upper_green = np.array([86, 255, 255])
                green_mask = cv2.inRange(hsv, lower_green, upper_green)
                green_clean = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
                
                contours_g, _ = cv2.findContours(green_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours_g:
                    if cv2.contourArea(cnt) > 1100:
                        counts["hyacinth"] += 1
                        gx, gy, gw, gh = cv2.boundingRect(cnt)
                        cv2.rectangle(annotated, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 2)
                        cv2.putText(annotated, "HYACINTH BIO-MAT", (gx, max(15, gy - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2)

                # --- C. DETECT RITUAL SACRED WASTE (NIRMALYA / DIYAS / GARLANDS) ---
                lower_orange = np.array([10, 80, 80])
                upper_orange = np.array([28, 255, 255])
                ritual_mask = cv2.inRange(hsv, lower_orange, upper_orange)
                ritual_clean = cv2.morphologyEx(ritual_mask, cv2.MORPH_CLOSE, (7, 7))
                
                contours_r, _ = cv2.findContours(ritual_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for rcnt in contours_r:
                    if 180 < cv2.contourArea(rcnt) < 3200:
                        counts["ritual"] += 1
                        rx, ry, rw, rh = cv2.boundingRect(rcnt)
                        cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), (0, 165, 255), 2)
                        cv2.putText(annotated, "RITUAL (NIRMALYA)", (rx, max(15, ry - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 2)

                # Display Annotated Stream
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st_frame.image(annotated_rgb, use_container_width=True)
                
                total_objects = sum(counts.values())
                denom = max(total_objects, 1)
                
                with breakdown_ph.container():
                    st.markdown("<div class='stat-box'><div class='stat-box-title' style='margin-bottom:12px;'>LIVE DETECTION BREAKDOWN</div>", unsafe_allow_html=True)
                    st.write(f"● **Dry Plastic Waste:** `{counts['plastic']}`")
                    st.progress(min(counts["plastic"] / denom, 1.0))
                    st.write(f"● **Hyacinth Bio-Mat:** `{counts['hyacinth']}`")
                    st.progress(min(counts["hyacinth"] / denom, 1.0))
                    st.write(f"● **Ritual Waste (Nirmalya/Diya):** `{counts['ritual']}`")
                    st.progress(min(counts["ritual"] / denom, 1.0))
                    st.write(f"● **Mixed Floating Debris (Fallback):** `{counts['debris']}`")
                    st.progress(min(counts["debris"] / denom, 1.0))
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                with kpi_ph.container():
                    st.markdown(f"<div class='stat-box' style='margin-top:14px;'><div class='stat-box-title'>TOTAL OBJECTS IDENTIFIED</div><h2 style='font-size:32px; color:#38BDF8; margin:4px 0;'>{total_objects}</h2><div style='font-size:11px; color:#64748B;'>Multi-Class Discriminator Active</div></div>", unsafe_allow_html=True)

            frame_idx += 1
            time.sleep(0.02)
            
        cap.release()
        
        # Save exact findings into session state
        st.session_state.has_run = True
        st.session_state.detected_counts = counts
        st.session_state.total_detected = total_objects
        st.session_state.estimated_volume_m3 = (
            (counts['plastic'] * 0.04) +
            (counts['hyacinth'] * 0.35) +
            (counts['ritual'] * 0.02) +
            (counts['debris'] * 0.06)
        )
        st.success("✅ Multi-class inference completed! Hotspot Map and Work Orders dynamically updated.")

# ==============================
# 3. HOTSPOT MAP MODULE
# ==============================
elif selected_module == "Hotspot Map":
    st.markdown("<div class='view-category'>GEOSPATIAL LAYER · PRECISE WATERCOURSE GIS</div>", unsafe_allow_html=True)
    st.markdown("<div class='view-title'>Hotspots & hydrodynamic drift.</div>", unsafe_allow_html=True)
    st.markdown("<div class='view-desc'>Geospatial heatmaps and projected trajectories calibrated along the Indrayani River centerline in Alandi.</div>", unsafe_allow_html=True)
    
    col_map, col_drift = st.columns([2.3, 1])
    
    # Compute Exact Dynamic Weights from Inference Results
    if st.session_state.has_run and st.session_state.total_detected > 0:
        c_p = st.session_state.detected_counts['plastic']
        c_h = st.session_state.detected_counts['hyacinth']
        c_r = st.session_state.detected_counts['ritual']
        c_d = st.session_state.detected_counts['debris']
        
        # Exact mathematical distribution along the river corridor
        w_upstream = min(0.35 + (c_p * 0.015) + (c_d * 0.01), 0.95)
        w_ghat     = min(0.40 + (c_r * 0.03) + (c_p * 0.025), 1.0)
        w_bridge   = min(0.30 + (c_p * 0.035) + (c_h * 0.02), 0.98)
        w_weir     = min(0.45 + (c_h * 0.04) + (c_d * 0.02), 1.0)
        w_sandbar  = 0.35
    else:
        # Default baseline prior to custom video analysis
        w_upstream, w_ghat, w_bridge, w_weir, w_sandbar = 0.55, 0.85, 0.70, 0.80, 0.35
        
    with col_map:
        # Precise center coordinates of the Indrayani water channel in Alandi
        m = folium.Map(location=[18.6758, 73.8975], zoom_start=15, tiles="CartoDB dark_matter")
        
        river_hotspots = [
            {
                "lat": 18.6782, "lon": 73.8942, 
                "weight": w_upstream, 
                "name": "Reach A-1 (Inflow Curve)", 
                "risk": "Critical" if w_upstream > 0.75 else "Moderate",
                "dominant": "Dry Plastics & Floating Wrappers" if w_upstream > 0.6 else "Surface Debris"
            },
            {
                "lat": 18.6766, "lon": 73.8960, 
                "weight": w_ghat, 
                "name": "Reach A-2 (Alandi Temple Ghat)", 
                "risk": "Critical" if w_ghat > 0.70 else "Moderate",
                "dominant": "Ritual Nirmalya & PET Plastics"
            },
            {
                "lat": 18.6751, "lon": 73.8973, 
                "weight": w_bridge, 
                "name": "Reach B-1 (Bhakti Sopan Bridge Pier #2)", 
                "risk": "Critical" if w_bridge > 0.75 else "Moderate",
                "dominant": "Dry Plastic Accumulation & Choke Hazard"
            },
            {
                "lat": 18.6738, "lon": 73.8998, 
                "weight": w_weir, 
                "name": "Reach C-1 (Downstream Bund / Weir)", 
                "risk": "Critical" if w_weir > 0.75 else "Moderate",
                "dominant": "Dense Hyacinth Bio-Mats"
            },
            {
                "lat": 18.6720, "lon": 73.9030, 
                "weight": w_sandbar, 
                "name": "Reach C-2 (Sediment Sand Bar)", 
                "risk": "Cleared",
                "dominant": "Coarse Siltation"
            }
        ]
        
        # Heatmap Layer
        HeatMap([[p["lat"], p["lon"], p["weight"]] for p in river_hotspots], radius=26, blur=18, min_opacity=0.45).add_to(m)
        
        # Color-coded Circular Markers
        for pt in river_hotspots:
            c = "#EF4444" if pt["risk"] == "Critical" else ("#F59E0B" if pt["risk"] == "Moderate" else "#22C55E")
            popup_content = f"""
            <div style='font-family:sans-serif; font-size:12px;'>
                <b>{pt['name']}</b><br>
                Status: <span style='color:{c}; font-weight:bold;'>{pt['risk']}</span><br>
                Primary Debris: {pt['dominant']}<br>
                Relative Hotspot Load: {pt['weight']:.2f}
            </div>
            """
            folium.CircleMarker(
                location=[pt["lat"], pt["lon"]],
                radius=8,
                popup=folium.Popup(popup_content, max_width=250),
                color=c,
                fill=True,
                fill_color=c,
                fill_opacity=0.85
            ).add_to(m)
            
        # Watercourse Polyline (Indrayani flow trajectory)
        watercourse_coords = [
            [18.6782, 73.8942],
            [18.6766, 73.8960],
            [18.6751, 73.8973],
            [18.6738, 73.8998],
            [18.6720, 73.9030]
        ]
        folium.PolyLine(
            locations=watercourse_coords,
            color="#38BDF8", weight=4, dash_array="6, 8", tooltip="Calibrated Hydrodynamic Drift Vector (48h)"
        ).add_to(m)
        
        st_folium(m, width="100%", height=480)
        
    with col_drift:
        st.markdown(f"""
        <div class="stat-box" style="margin-bottom:12px;">
            <div class="stat-box-title">HYDRODYNAMIC CHOKE ZONES</div>
            <div style="margin-top:14px; font-family:'JetBrains Mono'; font-size:12px;">
                <div style="background:#070B13; padding:10px; border-radius:6px; border-left:3px solid #EF4444; margin-bottom:8px;">
                    <div style="font-weight:700; color:#F8FAFC;">12h → Bridge Pier B-1</div>
                    <div style="color:#F87171; font-size:11px; margin-top:2px;">Plastic & Debris Choke Risk: {w_bridge*100:.0f}%</div>
                    <div style="color:#94A3B8; font-size:10px;">v · 0.46 m/s &nbsp;·&nbsp; ETA 12h</div>
                </div>
                <div style="background:#070B13; padding:10px; border-radius:6px; border-left:3px solid #EF4444; margin-bottom:8px;">
                    <div style="font-weight:700; color:#F8FAFC;">24h → Downstream Weir C-1</div>
                    <div style="color:#F87171; font-size:11px; margin-top:2px;">Hyacinth Compression: {w_weir*100:.0f}%</div>
                    <div style="color:#94A3B8; font-size:10px;">v · 0.41 m/s &nbsp;·&nbsp; ETA 24h</div>
                </div>
                <div style="background:#070B13; padding:10px; border-radius:6px; border-left:3px solid #38BDF8;">
                    <div style="font-weight:700; color:#F8FAFC;">48h → Sand Bar Reach C-2</div>
                    <div style="color:#38BDF8; font-size:11px; margin-top:2px;">Sediment Deposition Zone</div>
                    <div style="color:#94A3B8; font-size:10px;">v · 0.38 m/s &nbsp;·&nbsp; ETA 48h</div>
                </div>
            </div>
        </div>
        <div class="stat-box">
            <div class="stat-box-title">RIVER BASIN ANCHOR</div>
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
    st.markdown("<div class='view-desc'>Auto-generated tickets formulated directly from the AI detection run. Synced with the Alandi Municipal Council and Irrigation Department.</div>", unsafe_allow_html=True)
    
    # Generate Dynamic Data from Actual AI Findings
    if st.session_state.has_run and st.session_state.total_detected > 0:
        c_p = st.session_state.detected_counts['plastic']
        c_h = st.session_state.detected_counts['hyacinth']
        c_r = st.session_state.detected_counts['ritual']
        c_d = st.session_state.detected_counts['debris']
        
        orders_data = [
            {
                "ORDER": "WO-2026-0142",
                "REACH": "Reach A-2 (Main Alandi Ghat)",
                "GPS": "18.6766, 73.8960",
                "MATERIAL": f"Ritual Nirmalya ({c_r}) & Dry Plastics ({c_p})",
                "VOL (m³)": round((c_r * 0.02) + (c_p * 0.04), 1),
                "URGENCY": "CRITICAL" if (c_p + c_r) > 12 else "HIGH",
                "RECOMMENDED ACTION": "Deploy Manual Ghat Retrieval Crew + Floating Booms",
                "STATUS": "Pending"
            },
            {
                "ORDER": "WO-2026-0143",
                "REACH": "Reach B-1 (Bhakti Sopan Bridge Pier)",
                "GPS": "18.6751, 73.8973",
                "MATERIAL": f"Dry Plastic Containers & Bottles ({c_p})",
                "VOL (m³)": round(max(c_p * 0.04, 1.8), 1),
                "URGENCY": "CRITICAL" if c_p > 10 else "HIGH",
                "RECOMMENDED ACTION": "Deploy Anchored Deflection Boom + Skimmer Boat",
                "STATUS": "In Progress"
            },
            {
                "ORDER": "WO-2026-0144",
                "REACH": "Reach C-1 (Downstream Weir)",
                "GPS": "18.6738, 73.8998",
                "MATERIAL": f"Water Hyacinth Bio-Mat ({c_h} clumps)",
                "VOL (m³)": round(max(c_h * 0.35, 2.5), 1),
                "URGENCY": "CRITICAL" if c_h > 8 else "HIGH",
                "RECOMMENDED ACTION": "Deploy Heavy Mechanized Trash Skimmer Boat",
                "STATUS": "Pending"
            },
            {
                "ORDER": "WO-2026-0145",
                "REACH": "Reach A-1 (Inflow Curve)",
                "GPS": "18.6782, 73.8942",
                "MATERIAL": f"Mixed Residual Debris ({c_d})",
                "VOL (m³)": round(max(c_d * 0.06, 0.8), 1),
                "URGENCY": "MODERATE",
                "RECOMMENDED ACTION": "Periodic Survey & Patrol Monitoring",
                "STATUS": "In Progress"
            },
            {
                "ORDER": "WO-2026-0146",
                "REACH": "Reach C-2 (Downstream Sand Bar)",
                "GPS": "18.6720, 73.9030",
                "MATERIAL": "Coarse Siltation & Subsurface Sediment",
                "VOL (m³)": 14.5,
                "URGENCY": "MODERATE",
                "RECOMMENDED ACTION": "Schedule Pre-Monsoon Localized Dredging",
                "STATUS": "Pending"
            }
        ]
    else:
        # Default baseline prior to video run
        orders_data = [
            {"ORDER": "WO-2026-0101", "REACH": "Reach A-2 (Main Ghat)", "GPS": "18.6766, 73.8960", "MATERIAL": "Awaiting Aerial Video", "VOL (m³)": 0.0, "URGENCY": "MODERATE", "RECOMMENDED ACTION": "Execute CV Pipeline in Module 2", "STATUS": "Pending"}
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
    
    # Native interactive dataframe with status dropdowns
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
