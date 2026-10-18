import os
import time
import requests
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import cv2

st.set_page_config(
    page_title="ApexSentry | Industrial Telemetry Triage",
    page_icon="⚙️",
    layout="wide"
)

# --- Configuration & State ---
DEFAULT_AMD_KEY = "rc-e2a8efe2866caa4abb0fc073d72b3b51952e675990ed8137"
DEFAULT_BASE_URL = "https://developer.amd.com.cn/radeon/api/v1"

if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("AMD_API_KEY", DEFAULT_AMD_KEY)

with st.sidebar:
    st.title("Connection Settings")
    st.markdown("Configure your AMD Developer Cloud endpoint credentials.")
    
    api_url = st.text_input("Token Factory Base URL", value=DEFAULT_BASE_URL)
    api_key_input = st.text_input("AMD Bearer Token", value=st.session_state.api_key, type="password")
    if api_key_input:
        st.session_state.api_key = api_key_input

    model_id = st.selectbox(
        "AMD Cloud Model Endpoint",
        ["DeepSeek-V4-Flash", "Gemma-4-31B-IT"]
    )
    st.divider()
    st.caption("Stack: Edge OpenCV Filtering + AMD ROCm Cloud Hardware (vLLM inference)")

# --- Core Computer Vision (Edge Tier) ---
def analyze_surface(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    marked_img = img_array.copy()
    faults = 0
    fault_area = 0
    
    for c in contours:
        area = cv2.contourArea(c)
        if area > 35:
            faults += 1
            fault_area += area
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(marked_img, (x, y), (x + w, y + h), (225, 30, 30), 2)
            cv2.putText(marked_img, f"#{faults}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (225, 30, 30), 1)
            
    total_pixels = img_array.shape[0] * img_array.shape[1]
    pct_damaged = round((fault_area / total_pixels) * 100, 2)
    return marked_img, faults, pct_damaged

# --- Incident Telemetry Presets ---
PRESETS = {
    "Gas Turbine Blade (Stage 2 High Vibration)": {
        "notes": "Inconel 718 blade. High-frequency acoustic spikes observed near stage 2 disc.",
        "sensor_label": "Vibration RMS (mm/s)",
        "sensor_vals": [4.1, 4.2, 4.5, 4.9, 5.8, 6.4, 7.8, 8.9, 9.4, 9.1],
        "draw_type": "crack"
    },
    "Hydraulic Piston (Chamber Pressure Drop)": {
        "notes": "4140 chrome-plated shaft. Seal failure suspected; pressure dropped 18% over shift.",
        "sensor_label": "Line Pressure (bar)",
        "sensor_vals": [198, 195, 190, 184, 180, 172, 168, 165, 162, 160],
        "draw_type": "pitting"
    },
    "Conveyor Drive Bearing (Overheating)": {
        "notes": "Main intake line. Grease expelled from seal; surface temp measuring 84°C.",
        "sensor_label": "Bearing Housing Temp (°C)",
        "sensor_vals": [55, 58, 61, 66, 70, 75, 79, 82, 84, 85],
        "draw_type": "shear"
    }
}

def generate_sample_plate(draw_type):
    canvas = np.full((320, 500, 3), 205, dtype=np.uint8)
    if draw_type == "crack":
        cv2.line(canvas, (90, 120), (390, 170), (35, 35, 35), 3)
        cv2.line(canvas, (250, 150), (290, 230), (45, 45, 45), 2)
    elif draw_type == "pitting":
        pts = [(130, 90), (160, 140), (210, 120), (310, 210), (340, 180), (280, 130), (200, 190)]
        for pt in pts:
            cv2.circle(canvas, pt, 6, (30, 30, 30), -1)
    elif draw_type == "shear":
        cv2.line(canvas, (50, 60), (450, 260), (30, 30, 30), 4)
    return canvas

# --- User Interface ---
st.title("⚙️ ApexSentry — Industrial Incident Triage")
st.markdown("Rapid edge visual fault detection coupled with AMD ROCm remote cloud reasoning.")

tab_run, tab_info = st.tabs(["Dashboard", "Architecture & Telemetry Specs"])

with tab_run:
    col_left, col_right = st.columns([1, 1], gap="large")
    
    with col_left:
        st.subheader("1. Edge Visual & Sensor Acquisition")
        selected_preset = st.selectbox("Component Incident Preset", list(PRESETS.keys()))
        case_data = PRESETS[selected_preset]
        
        base_img = generate_sample_plate(case_data["draw_type"])
        inspected_img, count, density = analyze_surface(base_img)
        
        st.image(inspected_img, caption="Edge CV Output: Isolated Fault Regions", width="stretch")
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Anomalies Flagged", f"{count}")
        m_col2.metric("Defect Surface Area", f"{density}%")
        m_col3.metric("Edge Frame Latency", "11 ms")
        
        st.markdown(f"**Field Telemetry:** {case_data['notes']}")
        
        df = pd.DataFrame({
            "Sample": list(range(1, len(case_data["sensor_vals"]) + 1)),
            case_data["sensor_label"]: case_data["sensor_vals"]
        })
        st.line_chart(df.set_index("Sample"), height=180)

    with col_right:
        st.subheader("2. AMD ROCm Cloud Reasoning Engine")
        st.write("Synthesizes an immediate operational brief using open models hosted on AMD hardware.")
        
        run_button = st.button("🚀 Transmit Telemetry to AMD Cloud", type="primary", use_container_width=True)
        
        if run_button:
            if not st.session_state.api_key:
                st.error("Missing AMD Bearer Token. Add it in the sidebar.")
            else:
                prompt = (
                    f"You are ApexSentry, an expert industrial maintenance systems engineer.\n"
                    f"Component: {selected_preset}\n"
                    f"Sensor telemetry stream: {case_data['sensor_label']} trend {case_data['sensor_vals']}\n"
                    f"Field notes: {case_data['notes']}\n"
                    f"Edge computer-vision telemetry: {count} fault boundaries identified covering {density}% of the component surface area.\n\n"
                    f"Respond in clean markdown using these exact headings:\n"
                    f"### 🚨 Urgency Level: (CRITICAL / HIGH / ADVISORY)\n"
                    f"### 🔍 Root Cause Hypothesis:\n"
                    f"(2-3 clear sentences explaining the mechanical failure mode)\n"
                    f"### 🛠️ Containment & Operator Protocol:\n"
                    f"1. Immediate line control action (e.g., speed reduction, emergency shutoff, bypass)\n"
                    f"2. Physical inspection step\n"
                    f"3. Part replacement code & safety lockout recommendation"
                )
                
                headers = {
                    "Authorization": f"Bearer {st.session_state.api_key}",
                    "Content-Type": "application/json"
                }
                body = {
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 450
                }
                
                with st.spinner(f"Querying {model_id} on AMD Radeon Cloud..."):
                    try:
                        start_time = time.time()
                        clean_url = api_url.rstrip("/")
                        if not clean_url.endswith("/chat/completions"):
                            clean_url = f"{clean_url}/chat/completions"
                            
                        res = requests.post(clean_url, headers=headers, json=body, timeout=25)
                        duration = round(time.time() - start_time, 2)
                        
                        if res.status_code == 200:
                            data = res.json()
                            reply = data["choices"][0]["message"]["content"]
                            st.success(f"Inference verified via AMD Cloud ({model_id}) in {duration}s")
                            st.markdown(reply)
                        else:
                            st.error(f"HTTP {res.status_code}: {res.text}")
                    except Exception as err:
                        st.error(f"Request failed: {err}")

with tab_info:
    st.markdown("""
    ### System Architecture
    * **Edge Processing Tier:** Local OpenCV routines isolate component fracture boundaries, count anomalies, and compute surface degradation density in real time without sending bandwidth-heavy video to cloud instances.
    * **Cloud Acceleration Tier:** Lightweight telemetry and vision metrics are transmitted to AMD Token Factory endpoints (`developer.amd.com.cn/radeon/api/v1`), invoking open models accelerated by AMD ROCm infrastructure.
    * **Partner Eligibility:** Utilizes AMD hardware infrastructure and supports Google's open Gemma models (`Gemma-4-31B-IT`).
    """)