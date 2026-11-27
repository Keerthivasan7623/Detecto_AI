"""
Plotly chart builders for Detecto AI / SafetyVision AI.
Elevated with Taste Skill - Premium Enterprise Light Theme Design System.
Each returns None when there is no data to plot.
"""
from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# TASTE SKILL ENTERPRISE COLOR PALETTE
# ==========================================
NAVY = "#0F172A"         # Deep Navy / Slate
GREEN = "#16A34A"        # Safe / Compliant Green
AMBER = "#F59E0B"        # Warning / Moderate Amber
RED = "#EF4444"          # Critical / Violation Red
BLUE = "#2563EB"         # Primary / Analytics Blue
PURPLE = "#7C3AED"       # Violet / AI
VIOLET = "#7C3AED"       # Violet / AI
TEAL = "#0D9488"         # Crisp Teal
LIGHT_BLUE = "#EFF6FF"   # Light Blue Accent
INFO_BLUE = "#2563EB"    # Information Blue
SLATE = "#64748B"        # Muted Slate
TEXT_PRIMARY = "#0F172A" # Primary Text
TEXT_SECONDARY = "#64748B" # Secondary Text

STATUS_COLORS = {
    "pending": AMBER,
    "detected": AMBER,
    "confirmed": RED,
    "resolved": GREEN,
    "dismissed": SLATE
}
SEVERITY_COLORS = {
    "low": GREEN,
    "medium": AMBER,
    "high": RED,
    "critical": "#991B1B"
}
WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]


def _style(fig: go.Figure, height: int = 330) -> go.Figure:
    """Applies Taste Skill clean light layout, typography, and axes formatting."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=16, r=16, t=32, b=16),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.28,
            x=0,
            font=dict(color=TEXT_PRIMARY, size=11, family="Plus Jakarta Sans, Inter, sans-serif")
        ),
        font=dict(family="Plus Jakarta Sans, Inter, -apple-system, sans-serif", color=TEXT_PRIMARY),
        title=dict(font=dict(color=TEXT_PRIMARY, size=13, family="Plus Jakarta Sans, Inter, sans-serif")),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor="#E2E8F0",
            font=dict(family="Plus Jakarta Sans, Inter, sans-serif", color=TEXT_PRIMARY, size=12)
        )
    )
    fig.update_xaxes(
        showgrid=False,
        color=TEXT_PRIMARY,
        tickfont=dict(color=TEXT_SECONDARY, size=11),
        title_font=dict(color=TEXT_PRIMARY, size=12)
    )
    fig.update_yaxes(
        gridcolor="#F1F5F9",
        zerolinecolor="#E2E8F0",
        color=TEXT_PRIMARY,
        tickfont=dict(color=TEXT_SECONDARY, size=11),
        title_font=dict(color=TEXT_PRIMARY, size=12)
    )
    return fig


def category_bar(rows: List[Dict[str, Any]]) -> Optional[go.Figure]:
    """Horizontal/vertical bar chart showing infractions by violation category."""
    if not rows:
        return None
    df = pd.DataFrame(rows).rename(columns={"category": "Category", "count": "Incidents"})
    fig = px.bar(
        df,
        x="Category",
        y="Incidents",
        text="Incidents",
        color_discrete_sequence=[BLUE]
    )
    fig.update_traces(
        textposition="outside",
        marker=dict(line=dict(width=0))
    )
    return _style(fig)


def category_donut(rows: List[Dict[str, Any]]) -> Optional[go.Figure]:
    """Donut chart showing violation categories proportion."""
    if not rows:
        return None
    df = pd.DataFrame(rows)
    palette_cycle = [BLUE, RED, AMBER, PURPLE, TEAL, NAVY, SLATE]
    fig = go.Figure(go.Pie(
        labels=[str(r.get("category", "")).replace("_", " ").title() for r in rows],
        values=[r.get("count", 0) for r in rows],
        hole=0.62,
        marker=dict(colors=palette_cycle[:len(rows)], line=dict(color="#FFFFFF", width=2)),
        textinfo="percent",
        hovertemplate="<b>%{label}</b><br>Incidents: %{value}<br>Share: %{percent}<extra></extra>"
    ))
    return _style(fig, 300)


def status_donut(rows: List[Dict[str, Any]]) -> Optional[go.Figure]:
    """Donut chart showing review and resolution lifecycle status."""
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fig = go.Figure(go.Pie(
        labels=[s.title() for s in df["status"]],
        values=df["count"],
        hole=0.62,
        marker=dict(colors=[STATUS_COLORS.get(s.lower(), SLATE) for s in df["status"]], line=dict(color="#FFFFFF", width=2)),
        textinfo="percent",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Ratio: %{percent}<extra></extra>"
    ))
    return _style(fig, 260)


def compliance_gauge(rate: Optional[float]) -> Optional[go.Figure]:
    """Overall compliance rate gauge (0-100%)."""
    if rate is None:
        return None
    color = GREEN if rate >= 90 else AMBER if rate >= 70 else RED
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rate,
        number={"suffix": "%", "font": {"color": TEXT_PRIMARY, "size": 32, "family": "Plus Jakarta Sans"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": TEXT_SECONDARY, "tickfont": {"color": TEXT_SECONDARY}},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 70], "color": "rgba(239, 68, 68, 0.10)"},
                {"range": [70, 90], "color": "rgba(245, 158, 11, 0.10)"},
                {"range": [90, 100], "color": "rgba(22, 163, 74, 0.10)"}
            ]
        }
    ))
    return _style(fig, 240)


def class_counts_bar(counts: Dict[str, int]) -> Optional[go.Figure]:
    """Horizontal bar chart showing counts of detections by object class."""
    if not counts:
        return None
    df = pd.DataFrame({"Class": list(counts), "Detections": list(counts.values())})
    df = df.sort_values("Detections")
    fig = px.bar(
        df,
        x="Detections",
        y="Class",
        orientation="h",
        color_discrete_sequence=[BLUE]
    )
    fig.update_traces(marker=dict(line=dict(width=0)))
    return _style(fig)


def daily_trend(rows: List[Dict[str, Any]]) -> Optional[go.Figure]:
    """Smooth area trend chart showing daily infractions logged."""
    if not rows:
        return None
    df = pd.DataFrame(rows).rename(columns={"date": "Date", "count": "Violations"})
    fig = px.area(
        df,
        x="Date",
        y="Violations",
        markers=True,
        color_discrete_sequence=[RED]
    )
    fig.update_traces(
        line=dict(width=2.5, color=RED),
        fillcolor="rgba(239, 68, 68, 0.08)"
    )
    return _style(fig)


def daily_safety_trend_chart(rows: List[Dict[str, Any]]) -> Optional[go.Figure]:
    """Line trend chart showing daily safety performance."""
    return daily_trend(rows)


def hourly_bar(rows: List[Dict[str, Any]]) -> Optional[go.Figure]:
    """Hourly incident distribution bar chart."""
    df = pd.DataFrame(rows)
    if df.empty or df["count"].sum() == 0:
        return None
    fig = px.bar(
        df,
        x="hour",
        y="count",
        labels={"hour": "Hour of Day (24h)", "count": "Violations"},
        color_discrete_sequence=[BLUE]
    )
    return _style(fig)


def shift_bar(rows: List[Dict[str, Any]]) -> Optional[go.Figure]:
    """Incidents distributed across operational shifts."""
    df = pd.DataFrame(rows)
    if df.empty or df["count"].sum() == 0:
        return None
    fig = px.bar(
        df,
        x="shift",
        y="count",
        text="count",
        labels={"shift": "Operational Shift", "count": "Incidents Logged"},
        color_discrete_sequence=[AMBER]
    )
    fig.update_traces(textposition="outside")
    return _style(fig)


def category_trend(rows: List[Dict[str, Any]]) -> Optional[go.Figure]:
    """Stacked progression of violation types over date intervals."""
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fig = px.bar(
        df,
        x="date",
        y="count",
        color="violation_type",
        labels={"date": "Date", "count": "Violations", "violation_type": "Equipment Category"},
        color_discrete_sequence=[BLUE, RED, AMBER, PURPLE, TEAL]
    )
    return _style(fig)


def risk_severity_stacked_bar(rows: List[Dict[str, Any]]) -> Optional[go.Figure]:
    """Stacked bar chart representing distribution of risk severities."""
    if not rows:
        return None
    df = pd.DataFrame(rows)
    sev_col = "severity" if "severity" in df.columns else ("risk_level" if "risk_level" in df.columns else None)
    if not sev_col:
        return None
    fig = px.histogram(
        df,
        x="category" if "category" in df.columns else "date",
        color=sev_col,
        color_discrete_map=SEVERITY_COLORS,
        labels={sev_col: "Risk Severity"}
    )
    return _style(fig)


def weekday_hour_heatmap(rows: List[Dict[str, Any]]) -> Optional[go.Figure]:
    """7x24 hazard hotspot matrix mapping incident density across weekday and hour."""
    if not rows:
        return None
    df = pd.DataFrame(rows)
    grid = df.pivot_table(index="weekday", columns="hour", values="count", fill_value=0)
    grid = grid.reindex(index=range(7), columns=range(24), fill_value=0)
    
    # Custom elegant light heatmap colorscale: light blue to warning amber to danger red
    custom_colorscale = [
        [0.0, "#FFFFFF"],
        [0.2, "#EFF6FF"],
        [0.5, "#FEF3C7"],
        [0.8, "#FCA5A5"],
        [1.0, "#EF4444"]
    ]
    
    fig = go.Figure(go.Heatmap(
        z=grid.values,
        x=[f"{h:02d}:00" for h in range(24)],
        y=WEEKDAYS,
        colorscale=custom_colorscale,
        hovertemplate="%{y} %{x} &rarr; %{z} violations<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    return _style(fig, 300)


def video_timeline(rows: List[Dict[str, Any]]) -> Optional[go.Figure]:
    """rows: [{'second': int, 'violations': int}] for the processed video."""
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fig = px.bar(
        df,
        x="second",
        y="violations",
        labels={"second": "Video Timeline (Seconds)", "violations": "Infractions Detected"},
        color_discrete_sequence=[RED]
    )
    return _style(fig, 260)


def safety_index_gauge(score: float, rating: str = "") -> Optional[go.Figure]:
    """Overall Plant Safety Index gauge (0-100)."""
    color = GREEN if score >= 80 else (AMBER if score >= 65 else RED)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100", "font": {"size": 32, "color": TEXT_PRIMARY, "family": "Plus Jakarta Sans"}},
        title={"text": f"Plant Safety Health Index<br><span style='font-size:0.85em;color:{color};font-weight:700'>{rating}</span>", "font": {"size": 13, "color": TEXT_PRIMARY}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": TEXT_SECONDARY, "tickfont": {"color": TEXT_SECONDARY}},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 65], "color": "rgba(239, 68, 68, 0.10)"},
                {"range": [65, 80], "color": "rgba(245, 158, 11, 0.10)"},
                {"range": [80, 100], "color": "rgba(22, 163, 74, 0.10)"}
            ]
        }
    ))
    return _style(fig, 240)


def ppe_radar_chart(category_dict: Dict[str, float]) -> Optional[go.Figure]:
    """Radar chart comparing compliance rate across 5 protective equipment domains."""
    if not category_dict:
        return None
    categories = list(category_dict.keys())
    values = list(category_dict.values())
    # Close polygon
    categories.append(categories[0])
    values.append(values[0])

    fig = go.Figure()
    # Industrial target benchmark
    fig.add_trace(go.Scatterpolar(
        r=[95] * len(categories),
        theta=categories,
        fill="toself",
        name="Target Benchmark (95%)",
        line=dict(color="rgba(37, 99, 235, 0.6)", dash="dash"),
        fillcolor="rgba(37, 99, 235, 0.05)"
    ))
    # Plant Actual
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        name="Plant Actual Compliance",
        line=dict(color=GREEN, width=2.5),
        fillcolor="rgba(22, 163, 74, 0.16)"
    ))
    fig.update_layout(
        template="plotly_white",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color=TEXT_SECONDARY, gridcolor="#E2E8F0", tickfont=dict(color=TEXT_SECONDARY)),
            angularaxis=dict(color=TEXT_PRIMARY, gridcolor="#E2E8F0", tickfont=dict(color=TEXT_PRIMARY, size=11))
        ),
        showlegend=True,
        legend=dict(orientation="h", y=-0.22, x=0, font=dict(color=TEXT_PRIMARY, size=11)),
        font=dict(family="Plus Jakarta Sans, Inter, sans-serif", color=TEXT_PRIMARY),
        margin=dict(l=32, r=32, t=24, b=32),
        paper_bgcolor="rgba(0,0,0,0)",
        height=320
    )
    return fig


def zone_breakdown_bar(zone_rows: List[Dict[str, Any]]) -> Optional[go.Figure]:
    """Horizontal bar comparing safety incidents and resolutions across zones."""
    if not zone_rows:
        return None
    df = pd.DataFrame(zone_rows)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["zone_name"],
        x=df["incident_count"],
        name="Total Incidents",
        orientation="h",
        marker=dict(color=RED)
    ))
    fig.add_trace(go.Bar(
        y=df["zone_name"],
        x=df["resolved_count"],
        name="Resolved / Closed",
        orientation="h",
        marker=dict(color=GREEN)
    ))
    fig.update_layout(barmode="group", height=320)
    return _style(fig, 320)


def capa_status_donut(capa_stats: Dict[str, Any]) -> Optional[go.Figure]:
    """Donut chart for CAPA action workflow status."""
    labels = ["Open", "In Progress", "Verified", "Closed"]
    values = [capa_stats.get("open", 0), capa_stats.get("in_progress", 0),
              capa_stats.get("verified", 0), capa_stats.get("closed", 0)]
    if sum(values) == 0:
        return None
    colors = [RED, AMBER, BLUE, GREEN]
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.62,
        marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
        textinfo="percent",
        hovertemplate="<b>%{label} Actions</b><br>Count: %{value}<br>Ratio: %{percent}<extra></extra>"
    ))
    return _style(fig, 260)
