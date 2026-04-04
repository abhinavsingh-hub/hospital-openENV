import streamlit as st
import pandas as pd
import random
import time
import json
import sys
import os
from PIL import Image

# Add parent directory to path to import env
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.hospital_env import HospitalEnv
from scripts.run_baseline import smart_agent

# Page Config
st.set_page_config(
    page_title="AI Hospital Dashboard",
    page_icon="🏥",
    layout="wide",
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at top left, #1e293b, #0f172a);
        color: #f8fafc;
    }

    /* Sidebar / Navbar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1e1e2d !important;
        background-image: linear-gradient(180deg, #1e1e2d 0%, #161625 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Glassmorphism Containers */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        border-color: rgba(59, 130, 246, 0.4);
    }

    /* Metric Styling */
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3b82f6, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Decision Highlight */
    .decision-bubble {
        border-left: 6px solid #3b82f6;
        background: linear-gradient(90deg, rgba(59, 130, 246, 0.1), transparent);
        padding: 20px;
        border-radius: 12px;
        animation: slideIn 0.5s ease-out;
    }

    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }

    /* Reward Pulse Animation */
    .reward-plus { animation: pulseGreen 1s infinite alternate; }
    .reward-minus { animation: shakeRed 0.5s ease-in-out; }

    @keyframes pulseGreen {
        from { box-shadow: 0 0 0px rgba(16, 185, 129, 0); }
        to { box-shadow: 0 0 20px rgba(16, 185, 129, 0.4); }
    }

    @keyframes shakeRed {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-5px); }
        75% { transform: translateX(5px); }
    }

    /* Patient Card */
    .patient-item {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 8px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Table Styling for Activity Log */
    [data-testid="stDataFrame"] {
        color: #ffffff !important;
    }
    
    [data-testid="stTable"] td {
        color: #ffffff !important;
        font-weight: 400;
    }
    
    /* Hide standard metrics delta */
    div[data-testid="stMetricDelta"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if 'env' not in st.session_state:
    with open('data/patients.json', 'r') as f:
        patients_data = json.load(f)
    st.session_state.env = HospitalEnv(data=patients_data)
    st.session_state.history = []
    st.session_state.total_reward = 0
    st.session_state.patients_processed = 0
    st.session_state.beds_total = 50
    st.session_state.beds_occupied = random.randint(30, 45)
    st.session_state.staff_total = 20
    st.session_state.staff_active = 15
    st.session_state.queue = patients_data[:5]
    st.session_state.latest_reward = 0

# --- SIDEBAR & LOGO ---
with st.sidebar:
    # Use the transparent logo
    if os.path.exists("assets/logo_transparent.png"):
        st.image("assets/logo_transparent.png", use_container_width=True)
    elif os.path.exists("assets/logo.png"):
        st.image("assets/logo.png", use_container_width=True)
    else:
        st.metric("🏥 AI HOSPITAL", "v1.0")
    
    st.markdown("---")
    st.header("🎮 Controls")
    if st.button("Reset Simulation 🔄", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    
    st.markdown("---")
    episodes = st.slider("Max Episodes", 1, 100, 20)
    st.toggle("Auto-step Mode", False)
    st.toggle("High Contrast Mode", True)
    
    st.markdown("---")
    st.caption("Developed with ❤️ for Medical Excellence.")

# --- HEADER ---
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.title("🏥 Smart Hospital RL Dashboard")
    st.markdown("Optimization of clinical triage and resource allocation using Reinforcement Learning.")

# --- METRIC TILES ---
m_col1, m_col2, m_col3, m_col4 = st.columns(4)
avg_reward = st.session_state.total_reward / max(1, st.session_state.patients_processed)

with m_col1:
    st.markdown(f'<div class="glass-card"><div class="metric-label">Total Reward</div><div class="metric-value">{st.session_state.total_reward:.2f}</div></div>', unsafe_allow_html=True)
with m_col2:
    st.markdown(f'<div class="glass-card"><div class="metric-label">Processed</div><div class="metric-value">{st.session_state.patients_processed}</div></div>', unsafe_allow_html=True)
with m_col3:
    st.markdown(f'<div class="glass-card"><div class="metric-label">Avg Reward</div><div class="metric-value">{avg_reward:.3f}</div></div>', unsafe_allow_html=True)
with m_col4:
    throughput = st.session_state.patients_processed / max(1, len(st.session_state.history))
    st.markdown(f'<div class="glass-card"><div class="metric-label">Throughput / Batch</div><div class="metric-value">{throughput:.1f}</div></div>', unsafe_allow_html=True)

# --- MAIN LAYOUT ---
left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    with st.container(border=True):
        st.subheader("🏙️ Hospital Capacity")
        
        # Beds
        bed_free = st.session_state.beds_total - st.session_state.beds_occupied
        st.write(f"**Beds Available**: {bed_free} / {st.session_state.beds_total}")
        st.progress((bed_free) / st.session_state.beds_total)
        
        # Staff
        st.write(f"**Staff Active**: {st.session_state.staff_active} / {st.session_state.staff_total}")
        st.progress(st.session_state.staff_active / st.session_state.staff_total)

    st.markdown("### ⏳ Patient Queue")
    for p in st.session_state.queue:
        st.markdown(f"""
        <div class="patient-item">
            <span style="color: #94a3b8; font-size: 0.8rem;">URGENCY UNKNOWN</span><br>
            👤 <b>Patient (Age {p['age']})</b><br>
            <span style="font-size: 0.9rem;">{p['symptoms']}</span>
        </div>
        """, unsafe_allow_html=True)

with right_col:
    st.subheader("🦾 Agent Workbench")
    
    if st.button("Simulate Next Patient ⏩", type="primary", use_container_width=True):
        # Simulation Logic
        state = st.session_state.env.reset()
        action = smart_agent(state)
        next_state, reward, done, info = st.session_state.env.step(action)
        
        # Update State
        st.session_state.total_reward += reward
        st.session_state.patients_processed += 1
        st.session_state.latest_reward = reward
        
        # Update History
        st.session_state.history.insert(0, {
            "ID": f"#{st.session_state.patients_processed}",
            "Symptoms": state['symptoms'],
            "Agent Priority": action['priority'],
            "Agent Dept": action['department'].capitalize(),
            "Outcome": "Matched" if reward > 0.5 else "Suboptimal",
            "Reward": f"{reward:+.2f}"
        })

    # Decision Display
    if st.session_state.patients_processed > 0:
        reward_class = "reward-plus" if st.session_state.latest_reward > 0.5 else "reward-minus"
        color = "#10b981" if st.session_state.latest_reward > 0.5 else "#ef4444"
        
        st.markdown(f"""
        <div class="glass-card {reward_class}">
            <h4>Latest Agent Decision</h4>
            <div class="decision-bubble">
                <p><b>Recommended Priority:</b> {st.session_state.history[0]['Agent Priority']}</p>
                <p><b>Target Department:</b> {st.session_state.history[0]['Agent Dept']}</p>
                <hr style="opacity: 0.1">
                <p style="color: {color}; font-weight: 700;">
                    IMPACT REWARD: {st.session_state.latest_reward:+.2f}
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Agent standing by. Start the simulation to observe decisions.")

    # History Table
    if st.session_state.history:
        st.markdown("### 📋 Activity Log")
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()
st.caption("Aesthetics and functionality optimized for AI observability. © 2026 Smart Hospital Initiative.")
