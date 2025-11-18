# File: app.py
import streamlit as st
import pandas as pd
import random
import time
from datetime import datetime

# ----------------------------
# 🛰️ Streamlit Page Setup
# ----------------------------
st.set_page_config(
    page_title="Network Anomaly Detection Dashboard",
    layout="wide",
    page_icon="🛰️"
)

st.title("🛰️ Real-Time Network Anomaly & User Behavior Dashboard")
st.caption("Live monitoring with AI-based anomaly alerts")

# ----------------------------
# Session State Initialization
# ----------------------------
if "logs" not in st.session_state:
    st.session_state.logs = pd.DataFrame(columns=["Timestamp", "Event", "Source IP", "Severity"])
if "alert_message" not in st.session_state:
    st.session_state.alert_message = None

# ----------------------------
# Placeholders
# ----------------------------
alert_box = st.empty()
chart_box = st.empty()
stats_box = st.container()
table_box = st.empty()

# ----------------------------
# Simulated Live Stream
# ----------------------------
for i in range(1000):
    # Generate mock network event
    event_type = random.choices(
        ["Normal Activity", "Suspicious Login Detected", "Anomaly in Network Traffic"],
        weights=[0.8, 0.15, 0.05]
    )[0]

    severity = (
        "Critical" if "Anomaly" in event_type
        else "Warning" if "Suspicious" in event_type
        else "Normal"
    )

    new_log = {
        "Timestamp": datetime.now().strftime("%H:%M:%S"),
        "Event": event_type,
        "Source IP": f"192.168.1.{random.randint(2, 254)}",
        "Severity": severity,
    }

    # Add new log entry (keep last 40)
    st.session_state.logs = pd.concat(
        [pd.DataFrame([new_log]), st.session_state.logs]
    ).head(40)

    # ----------------------------
    # 🚨 Alert Bar
    # ----------------------------
    if severity != "Normal":
        alert_color = "#dc2626" if severity == "Critical" else "#f59e0b"
        st.session_state.alert_message = (
            f"🚨 {event_type} detected from {new_log['Source IP']} at {new_log['Timestamp']}"
        )
        alert_box.markdown(
            f"""
            <div style="
                background:{alert_color};
                padding:12px;
                border-radius:10px;
                color:white;
                font-weight:600;
                text-align:center;
                box-shadow:0 0 15px {alert_color}77;">
                {st.session_state.alert_message}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        alert_box.empty()

    # ----------------------------
    # 📊 Chart: Event Type Counts
    # ----------------------------
    df_chart = st.session_state.logs.copy()
    df_chart["count"] = 1
    chart_data = df_chart.groupby("Severity")["count"].count()
    chart_box.bar_chart(chart_data, width="stretch")

    # ----------------------------
    # 📈 Stats Metrics
    # ----------------------------
    normal_count = (st.session_state.logs["Severity"] == "Normal").sum()
    warning_count = (st.session_state.logs["Severity"] == "Warning").sum()
    critical_count = (st.session_state.logs["Severity"] == "Critical").sum()
    total_logs = len(st.session_state.logs)

    col1, col2, col3, col4 = stats_box.columns(4)
    col1.metric("Total Logs", total_logs)
    col2.metric("Normal", normal_count)
    col3.metric("Warnings", warning_count)
    col4.metric("Critical Alerts", critical_count)

    # ----------------------------
    # 🧾 Log Table
    # ----------------------------
    table_box.dataframe(
        st.session_state.logs,
        width="stretch",
        height=400
    )

    time.sleep(1)
