import streamlit as st
import cv2
from ultralytics import YOLO
import time
import plotly.graph_objects as go
import plotly.express as px
from collections import deque
from datetime import datetime
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI CCTV Surveillance",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Dark Professional Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Rajdhani:wght@400;600;700&display=swap');

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1528 50%, #0a1020 100%);
    font-family: 'Rajdhani', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060c1a 0%, #0a1228 100%);
    border-right: 1px solid rgba(0,200,255,0.15);
}

/* Title */
h1, h2, h3 {
    font-family: 'Rajdhani', sans-serif !important;
    color: #00c8ff !important;
    letter-spacing: 2px;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(0,200,255,0.05);
    border: 1px solid rgba(0,200,255,0.2);
    border-radius: 12px;
    padding: 12px;
}

[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: #00c8ff !important;
    font-size: 2rem !important;
}

[data-testid="stMetricLabel"] {
    color: #7a9ab5 !important;
    font-family: 'Rajdhani', sans-serif !important;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-size: 0.75rem !important;
}

/* Buttons */
.stButton > button {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700;
    letter-spacing: 2px;
    border-radius: 8px;
    transition: all 0.3s ease;
}

/* Plotly charts border */
.js-plotly-plot {
    border: 1px solid rgba(0,200,255,0.1);
    border-radius: 12px;
}

/* Info/Success boxes */
.stSuccess {
    background: rgba(0,255,100,0.08) !important;
    border: 1px solid rgba(0,255,100,0.3) !important;
    border-radius: 8px !important;
}
.stInfo {
    background: rgba(0,150,255,0.08) !important;
    border: 1px solid rgba(0,150,255,0.3) !important;
    border-radius: 8px !important;
}
.stWarning {
    background: rgba(255,165,0,0.08) !important;
    border: 1px solid rgba(255,165,0,0.3) !important;
    border-radius: 8px !important;
}
.stError {
    background: rgba(255,50,50,0.08) !important;
    border: 1px solid rgba(255,50,50,0.3) !important;
    border-radius: 8px !important;
}

/* Header strip */
.header-strip {
    background: linear-gradient(90deg, rgba(0,200,255,0.1), rgba(0,100,255,0.05));
    border: 1px solid rgba(0,200,255,0.2);
    border-radius: 12px;
    padding: 14px 24px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 12px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-strip">
  <span style="font-size:2rem">🎯</span>
  <div>
    <div style="font-family:'Rajdhani',sans-serif;font-size:1.6rem;
                font-weight:700;color:#00c8ff;letter-spacing:3px;">
      AI CCTV SURVEILLANCE SYSTEM
    </div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                color:#4a7a95;letter-spacing:1px;">
      YOLOv8 · ByteTrack · Real-Time Analytics · Umar Saeed Jan
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
defaults = {
    "run": False,
    "entered": 0,
    "exited": 0,
    "people_tracker": {},
    "counted_ids": set(),
    "history_time": deque(maxlen=60),
    "history_count": deque(maxlen=60),
    "history_entered": deque(maxlen=60),
    "history_exited": deque(maxlen=60),
    "alert_log": [],
    "peak_count": 0,
    "total_frames": 0,
    "session_start": datetime.now().strftime("%H:%M:%S"),
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ SYSTEM CONTROLS")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("▶ START", type="primary", use_container_width=True):
            st.session_state.run = True
    with col_s2:
        if st.button("⛔ STOP", use_container_width=True):
            st.session_state.run = False

    st.divider()

    # Live metrics
    st.markdown("### 📊 LIVE METRICS")
    m1 = st.empty()
    m2 = st.empty()
    m3 = st.empty()
    m4 = st.empty()
    m5 = st.empty()

    st.divider()

    # Settings
    st.markdown("### 🎛️ SETTINGS")
    line_y_setting = st.slider("Detection Line Y", 100, 400, 240, 10)
    crowd_threshold = st.slider("Crowd Alert Threshold", 1, 20, 5)
    conf_threshold  = st.slider("YOLO Confidence", 0.1, 0.9, 0.4, 0.05)

    st.divider()

    # Alert log
    st.markdown("### 🚨 ALERT LOG")
    alert_box = st.empty()

    st.divider()
    st.markdown("""
    <div style='font-family:"JetBrains Mono",monospace;font-size:0.65rem;
                color:#2a4a5a;text-align:center;line-height:1.8'>
    YOLOv8n · ByteTrack<br>
    Umar Saeed Jan<br>
    AI & ML — Multan
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
vid_col, graph_col = st.columns([3, 2], gap="medium")

with vid_col:
    st.markdown("#### 📹 LIVE FEED")
    frame_placeholder = st.empty()
    status_placeholder = st.empty()

with graph_col:
    st.markdown("#### 📈 REAL-TIME ANALYTICS")
    graph1 = st.empty()   # People over time
    graph2 = st.empty()   # Entry/Exit bar
    graph3 = st.empty()   # Doughnut

# Bottom row — full width graphs
st.divider()
st.markdown("#### 📊 SESSION ANALYTICS")
b1, b2 = st.columns(2)
with b1:
    heatmap_ph = st.empty()
with b2:
    summary_ph = st.empty()

# ─────────────────────────────────────────────────────────────────────────────
# HELPER — Plotly dark template
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono", color="#7a9ab5", size=11),
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
)

def line_chart(times, counts, entered_list, exited_list):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(times), y=list(counts),
        mode="lines+markers",
        name="People",
        line=dict(color="#00c8ff", width=2),
        marker=dict(size=4),
        fill="tozeroy",
        fillcolor="rgba(0,200,255,0.08)"
    ))
    fig.add_trace(go.Scatter(
        x=list(times), y=list(entered_list),
        mode="lines", name="Entered",
        line=dict(color="#00ff88", width=1.5, dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=list(times), y=list(exited_list),
        mode="lines", name="Exited",
        line=dict(color="#ff4466", width=1.5, dash="dot")
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="People Count — Live", font=dict(color="#00c8ff", size=12)),
        xaxis=dict(showgrid=False, color="#2a4a5a"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,200,255,0.05)", color="#2a4a5a"),
        height=200
    )
    return fig

def bar_chart(entered, exited):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Entered", "Exited"],
        y=[entered, exited],
        marker_color=["#00ff88", "#ff4466"],
        marker_line_color=["#00cc66", "#cc2244"],
        marker_line_width=1,
        text=[entered, exited],
        textposition="outside",
        textfont=dict(color=["#00ff88","#ff4466"], size=16, family="JetBrains Mono")
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Entry vs Exit", font=dict(color="#00c8ff", size=12)),
        xaxis=dict(showgrid=False, color="#2a4a5a"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,200,255,0.05)", color="#2a4a5a"),
        height=180
    )
    return fig

def donut_chart(entered, exited):
    total = entered + exited
    if total == 0:
        entered, exited = 1, 1
    fig = go.Figure(go.Pie(
        labels=["Entered", "Exited"],
        values=[entered, exited],
        hole=0.65,
        marker=dict(colors=["#00ff88","#ff4466"],
                    line=dict(color="rgba(0,0,0,0)", width=0)),
        textfont=dict(size=10, family="JetBrains Mono"),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Traffic Split", font=dict(color="#00c8ff", size=12)),
        showlegend=True,
        height=180,
        annotations=[dict(text=f"{total}<br>Total",
                          x=0.5, y=0.5, showarrow=False,
                          font=dict(size=14, color="#00c8ff",
                                    family="JetBrains Mono"))]
    )
    return fig

def summary_table(peak, entered, exited, session_start, frames):
    data = {
        "Metric": ["Session Start","Total Frames","Peak Count",
                   "Total Entered","Total Exited","Net Flow"],
        "Value":  [session_start, frames, peak,
                   entered, exited, entered - exited]
    }
    df = pd.DataFrame(data)
    fig = go.Figure(go.Table(
        header=dict(
            values=["<b>METRIC</b>","<b>VALUE</b>"],
            fill_color="rgba(0,200,255,0.15)",
            font=dict(color="#00c8ff", size=12, family="JetBrains Mono"),
            align="left", height=30,
            line=dict(color="rgba(0,200,255,0.2)", width=1)
        ),
        cells=dict(
            values=[df["Metric"], df["Value"]],
            fill_color=[["rgba(0,200,255,0.03)","rgba(0,200,255,0.06)"]*3],
            font=dict(color=["#7a9ab5","#e0eeff"], size=12,
                      family="JetBrains Mono"),
            align="left", height=28,
            line=dict(color="rgba(0,200,255,0.1)", width=1)
        )
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title=dict(
        text="Session Summary", font=dict(color="#00c8ff", size=12)), height=280)
    return fig

def zone_bar(history_count):
    """Simple bar showing crowd intensity over time buckets"""
    if len(history_count) < 2:
        vals = [0]*10
    else:
        arr = list(history_count)
        chunk = max(1, len(arr)//10)
        vals = [int(np.mean(arr[i:i+chunk])) for i in range(0, len(arr), chunk)][:10]
        while len(vals) < 10:
            vals.append(0)
    colors = ["#00ff88" if v < 3 else "#ffaa00" if v < 6 else "#ff4466" for v in vals]
    fig = go.Figure(go.Bar(
        x=[f"T{i+1}" for i in range(len(vals))],
        y=vals,
        marker_color=colors,
        text=vals, textposition="outside",
        textfont=dict(size=10, family="JetBrains Mono", color="#aaccdd")
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Crowd Intensity Over Time", font=dict(color="#00c8ff", size=12)),
        xaxis=dict(showgrid=False, color="#2a4a5a"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,200,255,0.05)", color="#2a4a5a"),
        height=280
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.run:

    cap = cv2.VideoCapture(1)
    line_y = line_y_setting

    while cap.isOpened() and st.session_state.run:

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        st.session_state.total_frames += 1

        # ── YOLO tracking ────────────────────────────────────────────────────
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=conf_threshold,
            verbose=False
        )

        annotated_frame = results[0].plot()

        # ── Draw detection line ───────────────────────────────────────────────
        cv2.line(annotated_frame, (0, line_y), (640, line_y), (0, 200, 255), 2)
        cv2.putText(annotated_frame, "DETECTION LINE", (10, line_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,255), 1)

        people_count = 0

        # ── Process detections ────────────────────────────────────────────────
        for box in results[0].boxes:
            cls = int(box.cls[0])
            if cls != 0:
                continue

            people_count += 1
            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
            center_y = (by1 + by2) // 2

            if box.id is not None:
                track_id = int(box.id[0])

                # ID label on frame
                cv2.putText(annotated_frame, f"#{track_id}",
                            (bx1, by1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,220,0), 2)

                if track_id not in st.session_state.people_tracker:
                    st.session_state.people_tracker[track_id] = center_y

                prev_y = st.session_state.people_tracker[track_id]

                # ENTRY
                if (prev_y < line_y and center_y >= line_y and
                        track_id not in st.session_state.counted_ids):
                    st.session_state.entered += 1
                    st.session_state.counted_ids.add(track_id)
                    ts = datetime.now().strftime("%H:%M:%S")
                    st.session_state.alert_log.append(
                        f"🟢 {ts}  ID#{track_id} ENTERED")

                # EXIT
                elif (prev_y > line_y and center_y <= line_y and
                        track_id not in st.session_state.counted_ids):
                    st.session_state.exited += 1
                    st.session_state.counted_ids.add(track_id)
                    ts = datetime.now().strftime("%H:%M:%S")
                    st.session_state.alert_log.append(
                        f"🔴 {ts}  ID#{track_id} EXITED")

                st.session_state.people_tracker[track_id] = center_y

        # ── Peak count ────────────────────────────────────────────────────────
        if people_count > st.session_state.peak_count:
            st.session_state.peak_count = people_count

        # ── History update ────────────────────────────────────────────────────
        current_time = datetime.now().strftime("%H:%M:%S")
        st.session_state.history_time.append(current_time)
        st.session_state.history_count.append(people_count)
        st.session_state.history_entered.append(st.session_state.entered)
        st.session_state.history_exited.append(st.session_state.exited)

        # ── Crowd alert ───────────────────────────────────────────────────────
        crowd_alert = people_count >= crowd_threshold

        # ── Overlay info on frame ─────────────────────────────────────────────
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (0,0),(200,140),(0,0,0),-1)
        cv2.addWeighted(overlay, 0.45, annotated_frame, 0.55, 0, annotated_frame)

        cv2.putText(annotated_frame, f"PEOPLE: {people_count}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,200,255), 2)
        cv2.putText(annotated_frame, f"IN:  {st.session_state.entered}", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,  (0,255,130), 2)
        cv2.putText(annotated_frame, f"OUT: {st.session_state.exited}", (10,88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,  (0,80,255), 2)
        cv2.putText(annotated_frame, current_time, (460,22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

        if crowd_alert:
            cv2.putText(annotated_frame, "⚠ CROWD ALERT!", (180,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.rectangle(annotated_frame, (0,0),(640,480),(0,0,220), 3)

        # ── Display frame ─────────────────────────────────────────────────────
        rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb, channels="RGB", use_container_width=True)

        if crowd_alert:
            status_placeholder.error(
                f"🚨 CROWD ALERT — {people_count} people detected! "
                f"(Threshold: {crowd_threshold})")
        else:
            status_placeholder.success(
                f"✅ System Running — {people_count} people in frame")

        # ── Sidebar metrics ───────────────────────────────────────────────────
        m1.metric("👥 People Now",   people_count,
                  delta=people_count - (list(st.session_state.history_count)[-2]
                         if len(st.session_state.history_count) > 1 else 0))
        m2.metric("🟢 Entered",  st.session_state.entered)
        m3.metric("🔴 Exited",   st.session_state.exited)
        m4.metric("🏆 Peak",     st.session_state.peak_count)
        m5.metric("🎞️ Frames",   st.session_state.total_frames)

        # ── Alert log ─────────────────────────────────────────────────────────
        recent_alerts = st.session_state.alert_log[-8:]
        alert_text = "\n".join(recent_alerts) if recent_alerts else "No events yet"
        alert_box.code(alert_text, language=None)

        # ── Graphs ────────────────────────────────────────────────────────────
        with graph_col:
            graph1.plotly_chart(
                line_chart(
                    st.session_state.history_time,
                    st.session_state.history_count,
                    st.session_state.history_entered,
                    st.session_state.history_exited,
                ),
                use_container_width=True, config={"displayModeBar": False}, key=f"live_chart_1_{time.time()}"
            )
            graph2.plotly_chart(
                bar_chart(st.session_state.entered, st.session_state.exited),
                use_container_width=True, config={"displayModeBar": False}, key=f"bar_chart_live_2_{time.time()}"
            )
            graph3.plotly_chart(
                donut_chart(st.session_state.entered, st.session_state.exited),
                use_container_width=True, config={"displayModeBar": False}, key=f"donut_chart_live_3_{time.time()}"
            )

        # ── Bottom graphs ─────────────────────────────────────────────────────
        heatmap_ph.plotly_chart(
            zone_bar(st.session_state.history_count),
            use_container_width=True, config={"displayModeBar": False}, key=f"zone_bar_live_4_{time.time()}"
        )
        summary_ph.plotly_chart(
            summary_table(
                st.session_state.peak_count,
                st.session_state.entered,
                st.session_state.exited,
                st.session_state.session_start,
                st.session_state.total_frames,
            ),
            use_container_width=True, config={"displayModeBar": False}, key=f"summary_table_live_5_{time.time()}"
            )

        time.sleep(0.03)

    cap.release()
    status_placeholder.warning("⏹️ Video ended or system stopped.")

else:
    # ── IDLE STATE ────────────────────────────────────────────────────────────
    frame_placeholder.markdown("""
    <div style="
        background: rgba(0,200,255,0.04);
        border: 1px dashed rgba(0,200,255,0.2);
        border-radius: 16px;
        height: 300px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 12px;
    ">
        <div style="font-size:3.5rem">📹</div>
        <div style="font-family:'Rajdhani',sans-serif;font-size:1.3rem;
                    color:#00c8ff;letter-spacing:3px;">SYSTEM STANDBY</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;
                    color:#2a4a5a;">Press START in the sidebar to begin</div>
    </div>
    """, unsafe_allow_html=True)

    # Show empty charts in idle
    with graph_col:
        graph1.plotly_chart(line_chart([], [], [], []),
                            use_container_width=True, config={"displayModeBar":False}, key=f"line_chart_idle_1_{time.time()}")
        graph2.plotly_chart(bar_chart(0, 0),
                            use_container_width=True, config={"displayModeBar":False}, key=f"bar_chart_idle_2_{time.time()}")
        graph3.plotly_chart(donut_chart(0, 0),
                            use_container_width=True, config={"displayModeBar":False}, key=f"donut_chart_idle_3_{time.time()}")

    heatmap_ph.plotly_chart(zone_bar([]),
                            use_container_width=True, config={"displayModeBar":False}, key=f"zone_bar_idle_4_{time.time()}")
    summary_ph.plotly_chart(
        summary_table(0, 0, 0,
                      st.session_state.session_start,
                      st.session_state.total_frames),
        use_container_width=True, config={"displayModeBar":False}, key=f"summary_table_idle_5_{time.time()}"
    )
