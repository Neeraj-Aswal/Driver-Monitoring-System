import streamlit as st
import subprocess
import sys
from pathlib import Path
import os
import time

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
MAIN_SCRIPT = PROJECT_DIR / "main_monitor.py"
LOG_FILE = PROJECT_DIR / "monitor.log"

st.set_page_config(
    page_title="Driver Monitoring System",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# 2. ENHANCED CUSTOM STYLING (MODERN SAAS UI)
# --------------------------------------------------

st.markdown(
    """
    <style>
        /* Global Theme */
        .stApp {
            background-color: #0b0f17;
            color: #e2e8f0;
        }

        .block-container {
            padding-top: 2.5rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #111827;
            border-right: 1px solid #1f2937;
        }

        /* Dashboard Cards */
        .metric-card {
            background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
            border: 1px solid #374151;
            border-radius: 16px;
            padding: 24px 20px;
            min-height: 130px;
            text-align: center;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            border-color: #4f46e5;
            transform: translateY(-2px);
            box-shadow: 0 15px 30px -5px rgba(79, 70, 229, 0.15);
        }

        .metric-label {
            color: #9ca3af;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 1.2px;
            text-transform: uppercase;
            margin-bottom: 10px;
        }

        .metric-value {
            color: #f3f4f6;
            font-size: 22px;
            font-weight: 700;
            margin: 0;
        }

        /* Status Pills */
        .status-pill-active {
            display: inline-flex;
            align-items: center;
            background-color: rgba(16, 185, 129, 0.1);
            color: #34d399;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 15px;
            font-weight: 600;
            border: 1px solid rgba(16, 185, 129, 0.2);
        }

        .status-pill-inactive {
            display: inline-flex;
            align-items: center;
            background-color: rgba(239, 68, 68, 0.1);
            color: #f87171;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 15px;
            font-weight: 600;
            border: 1px solid rgba(239, 68, 68, 0.2);
        }

        .phone-value { color: #60a5fa; }
        .face-value { color: #34d399; }

        /* Section Headings */
        .section-heading {
            font-size: 18px;
            font-weight: 700;
            color: #f3f4f6;
            margin-bottom: 16px;
            letter-spacing: -0.3px;
        }

        /* Terminal Logs Box */
        .terminal-box {
            background-color: #030712;
            border: 1px solid #1f2937;
            border-radius: 12px;
            padding: 16px;
            font-family: 'Courier New', Courier, monospace;
            color: #38bdf8;
            font-size: 13px;
            max-height: 350px;
            overflow-y: auto;
        }

        /* Buttons Customization */
        div.stButton > button {
            border-radius: 10px;
            font-weight: 600;
            letter-spacing: 0.3px;
            transition: all 0.2s ease;
        }

        /* Dividers */
        hr {
            border-color: #1f2937;
            margin-top: 1.5rem;
            margin-bottom: 1.5rem;
        }

        /* Footer */
        .dashboard-footer {
            text-align: center;
            color: #6b7280;
            font-size: 11px;
            font-weight: 500;
            letter-spacing: 1px;
            padding-top: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# 3. SESSION STATE
# --------------------------------------------------

if "monitor_process" not in st.session_state:
    st.session_state.monitor_process = None


# --------------------------------------------------
# 4. MONITORING FUNCTIONS
# --------------------------------------------------

def is_running():
    process = st.session_state.monitor_process
    return process is not None and process.poll() is None


def start_monitoring():
    if not MAIN_SCRIPT.exists():
        st.error(f"Monitoring script not found: {MAIN_SCRIPT}")
        return

    if is_running():
        return

    try:
        log = open(LOG_FILE, "a", encoding="utf-8")

        # Set environment to unbuffered so prints write to log.txt instantly
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        try:
            process = subprocess.Popen(
                [sys.executable, str(MAIN_SCRIPT)],
                cwd=str(PROJECT_DIR),
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,  # Unbuffered environment for live logging
            )
        except Exception:
            log.close()
            raise

        st.session_state.monitor_process = process
        st.session_state.log_handle = log
        st.success("Monitoring started successfully!")

    except Exception as error:
        st.error(f"Could not start monitoring: {error}")


def stop_monitoring():
    process = st.session_state.monitor_process

    if process is not None and process.poll() is None:
        process.terminate()

        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    log = st.session_state.get("log_handle")

    if log is not None:
        log.close()

    st.session_state.monitor_process = None
    st.session_state.log_handle = None
    st.info("Monitoring stopped.")


# --------------------------------------------------
# 5. SIDEBAR CONTROLS
# --------------------------------------------------

with st.sidebar:
    st.markdown("### 🚘 Control Center")
    st.caption("Driver Safety Telemetry Panel")

    st.divider()

    running = is_running()

    if not running:
        if st.button("▶ Start Monitoring", use_container_width=True, type="primary"):
            start_monitoring()
            st.rerun()
    else:
        if st.button("⏹ Stop Monitoring", use_container_width=True):
            stop_monitoring()
            st.rerun()

    st.divider()

    st.markdown("### ⚙️ Engine Specs")
    st.markdown(
        """
        - **Detection:** YOLOv8 + ONNX
        - **Landmarks:** MediaPipe Face
        - **Backend:** OpenCV / Python
        """
    )

    st.divider()

    if running:
        st.markdown('<span style="color: #34d399; font-weight:600;">● System Live</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color: #9ca3af; font-weight:600;">○ System Standby</span>', unsafe_allow_html=True)


# --------------------------------------------------
# 6. MAIN DASHBOARD HEADER
# --------------------------------------------------

st.title("Driver Monitoring Dashboard")
st.markdown("Real-time telemetry tracking driver fatigue, drowsiness ratios, and distraction events.")
st.caption("COMPUTER VISION ENGINE  ·  SAFETY COMPLIANCE  ·  LOW LATENCY")

st.divider()


# --------------------------------------------------
# 7. SYSTEM STATUS CARDS
# --------------------------------------------------

st.markdown('<div class="section-heading">System Overview</div>', unsafe_allow_html=True)

running = is_running()
col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    status_pill = '<span class="status-pill-active">🟢 Active</span>' if running else '<span class="status-pill-inactive">🔴 Inactive</span>'
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">System Status</div>
            <div class="metric-value" style="margin-top: 8px;">{status_pill}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Phone Detection Model</div>
            <div class="metric-value phone-value">YOLOv8 + ONNX</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Facial Landmark Tracking</div>
            <div class="metric-value face-value">MediaPipe Mesh</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# 8. STATUS NOTIFICATIONS
# --------------------------------------------------

st.write("")
if running:
    st.success("**Monitoring Session Active.** The computer vision processing feed is running in the background window.")
else:
    st.info("System is ready. Click **Start Monitoring** in the sidebar to activate the video stream.")


# --------------------------------------------------
# 9. TABS: LOGS & ARCHITECTURE
# --------------------------------------------------

st.divider()

tab_logs, tab_features = st.tabs(["📋 Live Activity Logs", "💡 Architecture Breakdown"])

with tab_logs:
    # Unpack all 3 columns matching the proportions [4, 1, 1]
    log_header, stream_column, refresh_column = st.columns([4, 1, 1])

    with log_header:
        st.markdown("### System Stream Logs")

    with stream_column:
        auto_stream = st.checkbox("Live Stream", value=True)

    with refresh_column:
        st.write("")
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    # Dynamic placeholder container for live log updates
    log_placeholder = st.empty()

    if LOG_FILE.exists():
        try:
            with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as log_file:
                log_lines = log_file.readlines()

            recent_logs = "".join(log_lines[-25:]) if log_lines else "Log file is currently empty."
            
            # Display inside the dynamic placeholder container
            log_placeholder.code(recent_logs, language="text")

        except OSError as error:
            log_placeholder.error(f"Could not read log file: {error}")
    else:
        log_placeholder.info("No activity logs generated yet. Start the monitoring loop to populate logs.")

    # Auto-refresh loop if the system is running and Live Stream is checked
    if running and auto_stream:
        time.sleep(2)
        st.rerun()

# --------------------------------------------------
# 10. FOOTER
# --------------------------------------------------

st.divider()
st.markdown(
    """
    <div class="dashboard-footer">
        DRIVER MONITORING SYSTEM &nbsp;·&nbsp; DESIGNED FOR SAFETY &nbsp;·&nbsp; STREAMLIT DASHBOARD
    </div>
    """,
    unsafe_allow_html=True,
)