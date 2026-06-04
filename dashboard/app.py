"""
StoreIQ Dashboard — Premium Streamlit dashboard for store intelligence.

Pages:
    📊 Overview          — KPI cards, event distribution, zone breakdown
    🔄 Conversion Funnel — North Star metric visualization
    🗺️ Zone Analytics    — Per-zone visits, dwell times, comparisons
    👤 Journey Explorer  — Individual customer journey replay
    📋 Event Log         — Filterable event stream
    ⚠️ Anomalies         — Detected anomalies with severity
    📈 Queue Monitor     — Billing queue health metrics
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))
from storage.database import Database
from config.settings import settings

# ═══════════════════════════════════════════════════════════
# Page Config & Theme
# ═══════════════════════════════════════════════════════════

st.set_page_config(
    page_title="StoreIQ — Store Intelligence",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium dark theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        background: linear-gradient(160deg, #0a0a1a 0%, #111127 40%, #16213e 100%);
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stSidebar"] {
        background: rgba(16, 18, 40, 0.97);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(124, 58, 237, 0.25);
    }

    /* Glassmorphism cards */
    .glass-card {
        background: rgba(25, 30, 56, 0.65);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(124, 58, 237, 0.18);
        border-radius: 16px;
        padding: 24px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-card:hover {
        border-color: rgba(124, 58, 237, 0.5);
        box-shadow: 0 8px 40px rgba(124, 58, 237, 0.12);
        transform: translateY(-2px);
    }

    /* KPI metric cards */
    .kpi-card {
        background: linear-gradient(135deg, rgba(25, 30, 56, 0.8), rgba(40, 45, 80, 0.5));
        backdrop-filter: blur(12px);
        border: 1px solid rgba(124, 58, 237, 0.15);
        border-radius: 16px;
        padding: 20px 24px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .kpi-card:hover {
        border-color: rgba(124, 58, 237, 0.45);
        box-shadow: 0 4px 24px rgba(124, 58, 237, 0.1);
    }

    .kpi-icon { font-size: 28px; margin-bottom: 4px; }
    .kpi-value {
        font-size: 34px;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
        margin: 6px 0;
    }
    .kpi-label {
        font-size: 12px;
        color: #7c8db5;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
    }

    /* Section titles */
    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #e2e8f0;
        margin: 28px 0 14px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(124, 58, 237, 0.25);
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Journey timeline */
    .journey-step {
        background: rgba(25, 30, 56, 0.6);
        border-radius: 12px;
        padding: 14px 18px;
        margin: 6px 0;
        border-left: 4px solid #7c3aed;
        transition: all 0.2s ease;
    }
    .journey-step:hover { background: rgba(35, 40, 70, 0.7); }

    .journey-arrow {
        text-align: center;
        color: #7c3aed;
        font-size: 18px;
        margin: 2px 0;
    }

    /* Alert cards */
    .alert-card {
        background: rgba(25, 30, 56, 0.6);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
        border-left: 4px solid;
        transition: all 0.2s ease;
    }
    .alert-card:hover { background: rgba(35, 40, 70, 0.7); }

    /* Brand header */
    .brand-title {
        font-size: 26px;
        font-weight: 800;
        background: linear-gradient(135deg, #7c3aed, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .brand-sub {
        font-size: 12px;
        color: #6b7db3;
        margin-top: -6px;
        letter-spacing: 0.5px;
    }

    /* Hide streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    h1, h2, h3 { color: #e2e8f0 !important; }
    .stDataFrame { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# Plotly Theme Helper
# ═══════════════════════════════════════════════════════════

def dark_layout(fig, height=350):
    """Apply consistent dark theme to Plotly figures."""
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d1e0", family="Inter"),
        margin=dict(l=20, r=20, t=30, b=40),
        height=height,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=11),
        ),
    )
    fig.update_xaxes(gridcolor="rgba(124,58,237,0.08)", zerolinecolor="rgba(124,58,237,0.15)")
    fig.update_yaxes(gridcolor="rgba(124,58,237,0.08)", zerolinecolor="rgba(124,58,237,0.15)")
    return fig


# ═══════════════════════════════════════════════════════════
# Database
# ═══════════════════════════════════════════════════════════

db = Database()


@st.cache_resource(show_spinner="🧠 Loading YOLOv8 model (first time only)...")
def _load_detector(model_path=None):
    """Cached YOLO detector — loads only once per session."""
    from core.detector import PersonDetector
    return PersonDetector(model_path=model_path)


# ═══════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="brand-title">🏪 StoreIQ</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">AI-Powered Store Intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "📊 Overview",
            "🔄 Conversion Funnel",
            "🗺️ Zone Analytics",
            "👤 Journey Explorer",
            "📋 Event Log",
            "📈 Queue Monitor",
            "⚠️ Anomalies",
            "🎬 Analyze Video",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    stats = db.get_stats()
    st.caption(f"📦 {stats['total_events']} events · {stats['total_journeys']} journeys")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()


# ═══════════════════════════════════════════════════════════
# 📊 OVERVIEW
# ═══════════════════════════════════════════════════════════

if page == "📊 Overview":
    st.markdown("## 📊 Store Intelligence Overview")

    funnel = db.get_conversion_funnel()
    event_counts = db.get_event_counts()
    zone_analytics = db.get_zone_analytics()

    # ── KPI Cards ──
    cols = st.columns(5)
    kpis = [
        ("👥", "Total Visitors", funnel.get("total_entries", 0)),
        ("📈", "Conversion", f"{funnel.get('conversion_rate', 0):.1%}"),
        ("🛒", "Reached Billing", funnel.get("reached_billing", 0)),
        ("💰", "Purchased", funnel.get("purchased", 0)),
        ("👀", "Browsed Zones", funnel.get("browsed_zones", 0)),
    ]

    for col, (icon, label, value) in zip(cols, kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # ── Charts ──
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-title">📊 Event Distribution</div>', unsafe_allow_html=True)
        if event_counts:
            sorted_events = dict(sorted(event_counts.items(), key=lambda x: x[1], reverse=True))
            fig = px.bar(
                x=list(sorted_events.keys()),
                y=list(sorted_events.values()),
                color=list(sorted_events.values()),
                color_continuous_scale=["#312e81", "#4c1d95", "#6d28d9", "#7c3aed", "#8b5cf6", "#a78bfa"],
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False, xaxis_title="", yaxis_title="Count")
            fig.update_xaxes(tickangle=45)
            dark_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No events yet. Process videos first!")

    with col_right:
        st.markdown('<div class="section-title">🗺️ Zone Visits</div>', unsafe_allow_html=True)
        if zone_analytics:
            fig = px.pie(
                values=[z["visit_count"] for z in zone_analytics],
                names=[z["zone"] for z in zone_analytics],
                hole=0.45,
                color_discrete_sequence=["#7c3aed", "#a855f7", "#c084fc", "#e9d5ff", "#6d28d9", "#4c1d95"],
            )
            fig.update_traces(textinfo="label+percent", textfont_size=12)
            dark_layout(fig, height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No zone data yet.")

    # ── Hourly Traffic ──
    st.markdown('<div class="section-title">⏰ Hourly Traffic Pattern</div>', unsafe_allow_html=True)
    hourly = db.get_hourly_traffic()
    if hourly:
        fig = px.area(
            x=[h["hour"] + ":00" for h in hourly],
            y=[h["entries"] for h in hourly],
        )
        fig.update_traces(
            fill="tonexty",
            line=dict(color="#7c3aed", width=3),
            fillcolor="rgba(124, 58, 237, 0.15)",
        )
        fig.update_layout(xaxis_title="Hour", yaxis_title="Entries")
        dark_layout(fig, height=280)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# 🔄 CONVERSION FUNNEL
# ═══════════════════════════════════════════════════════════

elif page == "🔄 Conversion Funnel":
    st.markdown("## 🔄 Conversion Funnel — The North Star")
    st.caption("Visitors who purchased ÷ Total visitors")

    funnel = db.get_conversion_funnel()

    if funnel.get("total_entries", 0) > 0:
        stages = [
            ("Entry", funnel["total_entries"]),
            ("Browsed Zones", funnel["browsed_zones"]),
            ("Reached Billing", funnel["reached_billing"]),
            ("Purchased", funnel["purchased"]),
        ]

        fig = go.Figure(go.Funnel(
            y=[s[0] for s in stages],
            x=[s[1] for s in stages],
            textinfo="value+percent initial",
            textfont=dict(size=14, color="white"),
            marker=dict(
                color=["#7c3aed", "#8b5cf6", "#a855f7", "#22c55e"],
                line=dict(width=0),
            ),
            connector=dict(line=dict(color="rgba(124,58,237,0.3)", width=2)),
        ))
        dark_layout(fig, height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Rate cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Browse Rate", f"{funnel.get('browse_rate', 0):.1%}")
        with col2:
            st.metric("Billing Reach Rate", f"{funnel.get('billing_reach_rate', 0):.1%}")
        with col3:
            st.metric("🎯 Conversion Rate", f"{funnel.get('conversion_rate', 0):.1%}")
        with col4:
            st.metric("Abandon Rate", f"{funnel.get('abandon_rate', 0):.1%}")
    else:
        st.info("No journey data available. Process videos first!")


# ═══════════════════════════════════════════════════════════
# 🗺️ ZONE ANALYTICS
# ═══════════════════════════════════════════════════════════

elif page == "🗺️ Zone Analytics":
    st.markdown("## 🗺️ Zone Analytics")

    zone_data = db.get_zone_analytics()

    if zone_data:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-title">📊 Visits by Zone</div>', unsafe_allow_html=True)
            fig = px.bar(
                x=[z["zone"] for z in zone_data],
                y=[z["visit_count"] for z in zone_data],
                color=[z["zone"] for z in zone_data],
                color_discrete_sequence=["#7c3aed", "#a855f7", "#c084fc", "#e9d5ff", "#6d28d9"],
            )
            fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Visit Count")
            dark_layout(fig)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="section-title">⏱️ Average Dwell Time</div>', unsafe_allow_html=True)
            fig = px.bar(
                x=[z["zone"] for z in zone_data],
                y=[z.get("avg_dwell") or 0 for z in zone_data],
                color=[z.get("avg_dwell") or 0 for z in zone_data],
                color_continuous_scale=["#1e1b4b", "#4c1d95", "#7c3aed", "#a855f7", "#c084fc"],
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False, xaxis_title="", yaxis_title="Seconds")
            dark_layout(fig)
            st.plotly_chart(fig, use_container_width=True)

        # Detailed table
        st.markdown('<div class="section-title">📋 Zone Details</div>', unsafe_allow_html=True)
        st.dataframe(
            zone_data,
            column_config={
                "zone": st.column_config.TextColumn("Zone"),
                "visit_count": st.column_config.NumberColumn("Visits", format="%d"),
                "unique_visitors": st.column_config.NumberColumn("Unique Visitors", format="%d"),
                "avg_dwell": st.column_config.NumberColumn("Avg Dwell (s)", format="%.1f"),
                "max_dwell": st.column_config.NumberColumn("Max Dwell (s)", format="%.1f"),
                "min_dwell": st.column_config.NumberColumn("Min Dwell (s)", format="%.1f"),
            },
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No zone data available. Process videos with zone configurations first.")


# ═══════════════════════════════════════════════════════════
# 👤 JOURNEY EXPLORER
# ═══════════════════════════════════════════════════════════

elif page == "👤 Journey Explorer":
    st.markdown("## 👤 Customer Journey Explorer")

    journeys = db.get_journeys()

    if journeys:
        # Summary stats
        total = len(journeys)
        purchased = sum(1 for j in journeys if j.get("purchased"))
        avg_dwell = sum(j.get("total_dwell_time", 0) for j in journeys) / total if total else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Journeys", total)
        c2.metric("Purchasers", purchased)
        c3.metric("Avg Dwell", f"{avg_dwell:.1f}s")

        st.markdown("---")

        # Journey selector
        person_ids = [j["person_id"] for j in journeys]
        selected = st.selectbox("Select a person to explore their journey", person_ids)

        journey = next((j for j in journeys if j["person_id"] == selected), None)

        if journey:
            # Journey header
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                entry = journey.get("entry_time", "N/A")
                st.metric("Entry Time", entry[:19] if entry and entry != "N/A" else "N/A")
            with col2:
                st.metric("Total Dwell", f"{journey.get('total_dwell_time', 0):.1f}s")
            with col3:
                st.metric("Reached Billing", "✅ Yes" if journey.get("reached_billing") else "❌ No")
            with col4:
                st.metric("Purchased", "✅ Yes" if journey.get("purchased") else "❌ No")

            # Zone timeline
            zones = journey.get("zones_visited", [])
            if zones:
                st.markdown('<div class="section-title">🚶 Journey Timeline</div>', unsafe_allow_html=True)

                # Entry marker
                st.markdown("""
                <div class="journey-step" style="border-left-color: #22c55e;">
                    🚪 <b>ENTRY</b> — Entered the store
                </div>
                """, unsafe_allow_html=True)
                st.markdown('<div class="journey-arrow">↓</div>', unsafe_allow_html=True)

                for i, z in enumerate(zones):
                    zone_name = z.get("zone", "Unknown")
                    dwell = z.get("dwell_seconds", 0)
                    zone_type = z.get("zone_type", "general")
                    enter_time = z.get("enter_time", "")

                    icons = {
                        "billing": "🛒", "skincare": "🧴", "makeup": "💄",
                        "fragrance": "🌸", "haircare": "💇", "entrance": "🚪",
                        "general": "📍",
                    }
                    icon = icons.get(zone_type, "📍")
                    color = "#22c55e" if zone_type == "billing" else "#7c3aed"

                    time_str = enter_time[:19] if enter_time else ""

                    st.markdown(f"""
                    <div class="journey-step" style="border-left-color: {color};">
                        {icon} <b>{zone_name}</b> — {dwell:.1f}s dwell
                        <br><span style="color: #6b7db3; font-size: 12px;">{time_str}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    if i < len(zones) - 1:
                        st.markdown('<div class="journey-arrow">↓</div>', unsafe_allow_html=True)

                # Exit marker
                st.markdown('<div class="journey-arrow">↓</div>', unsafe_allow_html=True)
                exit_color = "#22c55e" if journey.get("purchased") else "#ef4444"
                exit_text = "Purchased ✅" if journey.get("purchased") else "No purchase"
                st.markdown(f"""
                <div class="journey-step" style="border-left-color: {exit_color};">
                    🚪 <b>EXIT</b> — {exit_text}
                </div>
                """, unsafe_allow_html=True)

            # Events table
            events = db.get_events(person_id=selected, limit=100)
            if events:
                st.markdown('<div class="section-title">📋 Events</div>', unsafe_allow_html=True)
                st.dataframe(events, use_container_width=True, hide_index=True)
    else:
        st.info("No journey data available yet. Process videos first!")


# ═══════════════════════════════════════════════════════════
# 📋 EVENT LOG
# ═══════════════════════════════════════════════════════════

elif page == "📋 Event Log":
    st.markdown("## 📋 Event Log")

    col1, col2, col3 = st.columns(3)
    with col1:
        event_types = [
            "All", "ENTRY", "EXIT", "ZONE_ENTER", "ZONE_EXIT", "ZONE_DWELL",
            "BILLING_QUEUE_JOIN", "BILLING_QUEUE_ABANDON", "PURCHASE_INFERRED",
            "REENTRY", "ANOMALY_DETECTED",
        ]
        type_filter = st.selectbox("Event Type", event_types)
    with col2:
        person_filter = st.text_input("Person ID", placeholder="e.g., person_1")
    with col3:
        zone_filter = st.text_input("Zone", placeholder="e.g., skincare")

    limit = st.slider("Results limit", 10, 500, 100)

    events = db.get_events(
        event_type=type_filter if type_filter != "All" else None,
        person_id=person_filter or None,
        zone=zone_filter or None,
        limit=limit,
    )

    if events:
        st.dataframe(events, use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(events)} events")
    else:
        st.info("No events match the filter criteria.")

    # Event count summary
    counts = db.get_event_counts()
    if counts:
        st.markdown('<div class="section-title">📊 Event Counts by Type</div>', unsafe_allow_html=True)
        fig = px.bar(
            x=list(counts.keys()), y=list(counts.values()),
            color=list(counts.values()),
            color_continuous_scale=["#312e81", "#7c3aed", "#a855f7"],
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False, xaxis_title="", yaxis_title="")
        fig.update_xaxes(tickangle=45)
        dark_layout(fig, height=280)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# 📈 QUEUE MONITOR
# ═══════════════════════════════════════════════════════════

elif page == "📈 Queue Monitor":
    st.markdown("## 📈 Billing Queue Monitor")

    queue = db.get_queue_metrics()

    if queue.get("total_queue_joins", 0) > 0:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Queue Joins", queue["total_queue_joins"])
        with col2:
            st.metric("Purchases", queue["total_purchases"])
        with col3:
            st.metric("Abandonments", queue["total_queue_abandons"])
        with col4:
            st.metric("Abandon Rate", f"{queue.get('abandon_rate', 0):.1%}")

        st.markdown("")
        st.metric(
            "⏱️ Avg Billing Dwell Time",
            f"{queue.get('avg_billing_dwell_seconds', 0):.1f} seconds",
        )

        # Queue events timeline
        st.markdown('<div class="section-title">📋 Queue Events</div>', unsafe_allow_html=True)
        queue_events = db.get_events(event_type="BILLING_QUEUE_JOIN", limit=50)
        abandon_events = db.get_events(event_type="BILLING_QUEUE_ABANDON", limit=50)
        purchase_events = db.get_events(event_type="PURCHASE_INFERRED", limit=50)

        all_queue = sorted(
            queue_events + abandon_events + purchase_events,
            key=lambda e: e.get("timestamp", ""),
            reverse=True,
        )

        if all_queue:
            st.dataframe(all_queue[:50], use_container_width=True, hide_index=True)
    else:
        st.info("No queue data available. Process videos with a billing zone configured.")


# ═══════════════════════════════════════════════════════════
# ⚠️ ANOMALIES
# ═══════════════════════════════════════════════════════════

elif page == "⚠️ Anomalies":
    st.markdown("## ⚠️ Anomaly Detection")

    anomalies = db.get_anomalies(limit=50)

    if anomalies:
        # Severity summary
        high = sum(1 for a in anomalies if a.get("severity") == "high")
        med = sum(1 for a in anomalies if a.get("severity") == "medium")
        low = sum(1 for a in anomalies if a.get("severity") == "low")

        c1, c2, c3 = st.columns(3)
        c1.metric("🔴 High", high)
        c2.metric("🟡 Medium", med)
        c3.metric("🟢 Low", low)

        st.markdown("---")

        for anomaly in anomalies:
            severity = anomaly.get("severity", "medium")
            colors = {"low": "#22c55e", "medium": "#eab308", "high": "#ef4444"}
            icons = {"low": "🟢", "medium": "🟡", "high": "🔴"}
            color = colors.get(severity, "#eab308")
            icon = icons.get(severity, "🟡")

            st.markdown(f"""
            <div class="alert-card" style="border-left-color: {color};">
                {icon} <b style="color: {color};">{anomaly.get('anomaly_type', 'Unknown')}</b>
                <span style="color: #6b7db3; float: right; font-size: 12px;">
                    {anomaly.get('timestamp', '')[:19]}
                </span>
                <br>
                <span style="color: #c9d1e0;">{anomaly.get('message', '')}</span>
                <br>
                <span style="color: #6b7db3; font-size: 12px;">
                    Metric: {anomaly.get('metric_name', '')} |
                    Value: {anomaly.get('current_value', '')} |
                    Expected: {anomaly.get('expected_range', '')}
                </span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No anomalies detected. All metrics within normal ranges.")
        st.markdown("""
        The anomaly detection system monitors:
        - 📊 Zone occupancy spikes
        - ⏱️ Unusual dwell times
        - 🚶 Traffic pattern deviations
        - 🛒 Queue length anomalies
        - 📉 Conversion rate drops
        """)


# ═══════════════════════════════════════════════════════════
# 🎬 ANALYZE VIDEO
# ═══════════════════════════════════════════════════════════

elif page == "🎬 Analyze Video":
    st.markdown("## 🎬 Multi-Camera Live Analysis")
    st.caption("Process all store cameras simultaneously to get full store intelligence. Each camera maps to a specific zone.")

    # ── Multi-Camera Configuration ──
    cameras = [
        {"id": "cam_1", "path": "data/videos/CAM 1 - zone.mp4", "zones": "config/zones_cam1.json", "name": "Skincare", "offset": 1000},
        {"id": "cam_2", "path": "data/videos/CAM 2 - zone.mp4", "zones": "config/zones_cam2.json", "name": "Makeup", "offset": 2000},
        {"id": "cam_3", "path": "data/videos/CAM 3 - entry.mp4", "zones": "config/zones_cam3.json", "name": "Entrance", "offset": 3000},
        {"id": "cam_5", "path": "data/videos/CAM 5 - billing.mp4", "zones": "config/zones_cam5.json", "name": "Billing", "offset": 5000},
    ]

    missing_files = []
    for cam in cameras:
        if not Path(cam["path"]).exists():
            missing_files.append(cam["path"])
        if not Path(cam["zones"]).exists():
            missing_files.append(cam["zones"])

    if missing_files:
        st.error(f"Missing required files for multi-camera analysis: {missing_files}")
        st.stop()

    # ── Settings ──
    st.markdown('<div class="section-title">⚙️ Processing Settings</div>', unsafe_allow_html=True)
    sc1, sc2 = st.columns(2)
    store_id = sc1.text_input("Store ID", "store_01")
    run_anomalies = sc2.checkbox("Anomaly Detection", value=True)

    st.markdown("")

    # ── Start Analysis ──
    if st.button("🚀 Start Multi-Camera Analysis", type="primary", use_container_width=True):

        from core.tracker import PersonTracker
        from core.zone_manager import ZoneManager
        from core.event_engine import EventEngine
        from analytics.anomaly import AnomalyDetector
        import cv2

        with st.status("🔄 Running Multi-Camera AI Pipeline...", expanded=True) as status:
            st.write("🧠 Loading YOLOv8 person detection model...")
            detector = _load_detector()

            st.write("🔗 Initializing trackers and event engines...")
            cam_states = {}
            total_frames_approx = 0

            for cam in cameras:
                cap = cv2.VideoCapture(cam["path"])
                total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
                total_frames_approx = max(total_frames_approx, total_f)

                zm = ZoneManager(cam["zones"])
                cam_states[cam["id"]] = {
                    "cap": cap,
                    "fps": fps,
                    "frame_skip": max(1, fps // settings.PROCESS_FPS),
                    "tracker": PersonTracker(),
                    "zone_manager": zm,
                    "event_engine": EventEngine(zm),
                    "name": cam["name"],
                    "offset": cam["offset"],
                    "total_frames": total_f,
                    "active": True
                }

            proc_start = datetime.now()
            frame_number = 0
            processed_count = 0

            progress = st.progress(0, text="Starting frame processing...")
            
            # Live Metrics
            live_metrics = st.empty()
            
            # 2x2 Video Grid
            st.markdown("### 🔴 Live Feed")
            grid_row1 = st.columns(2)
            grid_row2 = st.columns(2)
            
            placeholders = {
                "cam_1": grid_row1[0].empty(),
                "cam_2": grid_row1[1].empty(),
                "cam_3": grid_row2[0].empty(),
                "cam_5": grid_row2[1].empty(),
            }

            active_cameras = len(cameras)

            while active_cameras > 0:
                frame_number += 1
                should_process_ui = False

                for cam_id, state in cam_states.items():
                    if not state["active"]:
                        continue

                    ret, frame = state["cap"].read()
                    if not ret:
                        state["active"] = False
                        active_cameras -= 1
                        continue

                    if frame_number % state["frame_skip"] != 0:
                        continue

                    should_process_ui = True
                    elapsed_seconds = frame_number / state["fps"]
                    timestamp = proc_start + timedelta(seconds=elapsed_seconds)

                    # 1. Detect
                    detections = detector.detect(frame)
                    
                    # 2. Track
                    detections = state["tracker"].update(detections)
                    
                    # Offset tracker ID so person IDs don't clash in DB across cameras
                    if detections.tracker_id is not None:
                        detections.tracker_id += state["offset"]

                    # 3. Events
                    events = state["event_engine"].process_frame(
                        detections, frame_number, timestamp, cam_id, store_id
                    )

                    # 4. Draw for live UI
                    annotated = state["zone_manager"].draw_zones(frame)
                    
                    # Draw boxes manually for live view
                    if detections.tracker_id is not None and len(detections) > 0:
                        for i in range(len(detections)):
                            tid = int(detections.tracker_id[i])
                            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (237, 58, 124), 2)
                            cv2.putText(annotated, f"#{tid}", (x1, max(0, y1 - 10)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # Resize for display performance
                    display_frame = cv2.resize(annotated, (640, 360))
                    placeholders[cam_id].image(
                        cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), 
                        caption=state["name"], 
                        use_container_width=True
                    )

                if should_process_ui:
                    processed_count += 1
                    pct = min(frame_number / total_frames_approx, 1.0)
                    progress.progress(pct, text=f"Processing Frame {frame_number:,} / {total_frames_approx:,}")

                    # Aggregate metrics every 20 processed cycles
                    if processed_count % 20 == 0:
                        total_entries = 0
                        total_billing = 0
                        in_store = 0
                        for state in cam_states.values():
                            m = state["event_engine"].get_store_metrics()
                            if state["name"] == "Entrance":
                                total_entries = m["total_entries"]
                                in_store = m["current_occupancy"]
                            elif state["name"] == "Billing":
                                total_billing = m["reached_billing"]
                        
                        conv_rate = (total_billing / total_entries) if total_entries > 0 else 0.0

                        with live_metrics.container():
                            lc1, lc2, lc3 = st.columns(3)
                            lc1.metric("🚪 Total Entries", total_entries)
                            lc2.metric("🛒 Reached Billing", total_billing)
                            lc3.metric("🎯 Conv. Rate", f"{conv_rate:.1%}")

            # Finalize
            all_events = []
            all_journeys = []
            total_entries = 0
            total_billing = 0

            for cam_id, state in cam_states.items():
                end_time = proc_start + timedelta(seconds=frame_number / state["fps"])
                state["event_engine"].finalize(frame_number, end_time)
                
                # Collect
                all_events.extend(state["event_engine"].all_events)
                all_journeys.extend(state["event_engine"].get_all_journeys())
                
                # Final metrics
                m = state["event_engine"].get_store_metrics()
                if state["name"] == "Entrance":
                    total_entries = m["total_entries"]
                elif state["name"] == "Billing":
                    total_billing = m["reached_billing"]
                
                state["cap"].release()

            progress.progress(1.0, text="✅ All frames processed!")
            st.write("💾 Saving events & journeys to database...")

            db.insert_events_batch(all_events)
            for journey in all_journeys:
                db.insert_journey(journey)

            overall_conv_rate = (total_billing / total_entries) if total_entries > 0 else 0.0

            run_results = {
                "video": "multi_camera",
                "camera_id": "all",
                "store_id": store_id,
                "total_frames": frame_number,
                "processed_frames": processed_count,
                "total_events": len(all_events),
                "metrics": {
                    "total_entries": total_entries,
                    "reached_billing": total_billing,
                    "conversion_rate": overall_conv_rate,
                    "purchased": total_billing # Assume reached billing = purchased
                },
            }
            db.record_processing_run(run_results)

            if run_anomalies:
                st.write("🔍 Running anomaly detection...")
                try:
                    anomaly_det = AnomalyDetector(db)
                    anomaly_det.compute_baselines()
                    anomalies_found = anomaly_det.run_detection(run_results["metrics"])
                    if anomalies_found:
                        st.write(f"⚠️ {len(anomalies_found)} anomalies detected")
                    else:
                        st.write("✅ No anomalies")
                except Exception as e:
                    st.write(f"⚠️ Anomaly detection skipped: {e}")

            status.update(label="✅ Analysis Complete!", state="complete", expanded=True)

        st.balloons()

        st.markdown("### 📊 Overall Store Analysis Results")
        st.markdown("")

        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("👥 Total Visitors", total_entries)
        rc2.metric("🎯 Conversion Rate", f"{overall_conv_rate:.1%}")
        rc3.metric("🛒 Reached Billing", total_billing)
        rc4.metric("📊 Total Events", len(all_events))

        st.markdown("")

        funnel_stages = ["Entered", "Reached Billing", "Purchased"]
        funnel_values = [total_entries, total_billing, total_billing]
        
        import plotly.graph_objects as go
        fig = go.Figure(go.Funnel(
            y=funnel_stages, x=funnel_values,
            textinfo="value+percent initial",
            textfont=dict(size=14, color="white"),
            marker=dict(
                color=["#7c3aed", "#8b5cf6", "#22c55e"],
                line=dict(width=0),
            ),
            connector=dict(line=dict(color="rgba(124,58,237,0.3)", width=2)),
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.success(
            f"✅ Multi-Camera processing done! {processed_count:,} frames per camera → {len(all_events)} combined events saved. "
            f"Use the sidebar to explore **Overview**, **Funnel**, **Zone Analytics**, and more!"
        )
