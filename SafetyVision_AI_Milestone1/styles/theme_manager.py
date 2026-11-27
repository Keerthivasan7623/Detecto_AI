"""
Detecto AI - Enterprise Theme Manager & Dynamic CSS Injection Engine
Powered by Taste Skill - Premium Light Theme Design System
Author: Detecto AI Platform Engineering & UX Architecture

Design System Characteristics:
- Pure Light Theme: #FFFFFF (primary background), #F7F8FA (soft grey), #F1F3F5 (secondary)
- Text Hierarchy: #171A21 (primary text), #5B6472 (secondary text), #87909D (muted text)
- Brand Accents: #172B4D (deep navy), #0F8B8D (professional teal), #7357D9 (AI purple), #EAF2FF (light blue)
- Status Colours: #16845B (success), #D99A22 (warning), #D94B4B (danger), #3276C8 (information)
- Typography: Inter / Plus Jakarta Sans / Manrope (high-contrast, professional numerical KPIs)
- Layout: 4/8-scale spacing, rounded cards (10-12px), subtle borders (#E2E8F0), soft shadows
"""

from typing import Dict, Any, Optional
import streamlit as st

# ==========================================
# 1. COLOR PALETTES DEFINITION (TASTE SKILL)
# ==========================================
LIGHT_PALETTE = {
    "name": "Light Mode",
    "bg_primary": "#FFFFFF",
    "bg_soft": "#F8FAFC",
    "bg_secondary": "#F1F5F9",
    "surface": "#FFFFFF",
    "surface_border": "#E2E8F0",
    "surface_border_hover": "#2563EB",
    "text_primary": "#0F172A",
    "text_secondary": "#475569",
    "text_muted": "#64748B",
    
    # Brand Accents
    "brand_navy": "#0F172A",
    "brand_teal": "#0D9488",
    "brand_purple": "#7C3AED",
    "brand_light_blue": "#EFF6FF",
    
    # Status Colors
    "status_success": "#16A34A",
    "status_warning": "#F59E0B",
    "status_danger": "#EF4444",
    "status_info": "#2563EB",
    
    # Legacy Color Mappings for backward-compatibility
    "accent_cyan": "#0D9488",
    "accent_blue": "#2563EB",
    "accent_dark_blue": "#0F172A",
    "accent_emerald": "#16A34A",
    "accent_red": "#EF4444",
    "accent_amber": "#F59E0B",
    
    # Containers & Cards
    "hud_bg": "linear-gradient(135deg, #FFFFFF 0%, #F7F8FA 50%, #F1F3F5 100%)",
    "hud_border": "#E2E8F0",
    "user_card_bg": "#FFFFFF",
    "user_card_border": "#E2E8F0",
    "empty_state_bg": "#F7F8FA",
    "empty_state_border": "#E2E8F0",
    "alert_bg": "#FDF1F1",
    "alert_border": "#F7BDBD",
    "alert_text": "#D94B4B",
    "input_bg": "#FFFFFF",
    "input_border": "#CBD5E1",
    "input_text": "#171A21",
    
    # Plotly Chart Integration
    "plotly_template": "plotly_white",
    "chart_grid": "#F1F5F9",
    "chart_radar_benchmark_fill": "rgba(37, 99, 235, 0.05)",
    "chart_radar_benchmark_line": "rgba(37, 99, 235, 0.50)",
    "chart_radar_actual_fill": "rgba(22, 163, 74, 0.16)",
    "chart_radar_actual_line": "#16A34A",
}

# DARK_PALETTE maps to light palette values to guarantee dark styling is completely eliminated
DARK_PALETTE = LIGHT_PALETTE


# ==========================================
# 2. THEME STATE HELPERS
# ==========================================
def get_theme_mode() -> str:
    """Retrieves current theme mode from session state (enforced 'Light Mode')."""
    return "Light Mode"


def set_theme_mode(theme_mode: str) -> None:
    """Persists theme mode in session state."""
    st.session_state["theme_mode"] = "Light Mode"


def render_theme_switcher(location: str = "sidebar") -> str:
    """Theme switcher locked to Taste Skill Enterprise Light Theme."""
    st.session_state["theme_mode"] = "Light Mode"
    return "Light Mode"


# ==========================================
# 3. DYNAMIC CSS INJECTION ENGINE
# ==========================================
def apply_theme_css(theme_mode: Optional[str] = None) -> None:
    """
    Injects custom CSS dynamically adapting Streamlit root containers,
    sidebar, metric cards, HUD headers, forms, inputs, tables, and typography
    to the Taste Skill enterprise light design system.
    """
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-primary: #FFFFFF;
        --bg-soft: #F8FAFC;
        --bg-secondary: #F1F5F9;
        --surface: #FFFFFF;
        --surface-border: #E2E8F0;
        --surface-border-hover: #2563EB;
        --text-primary: #0F172A;
        --text-secondary: #475569;
        --text-muted: #64748B;
        
        --brand-navy: #0F172A;
        --brand-teal: #0D9488;
        --brand-purple: #7C3AED;
        --brand-light-blue: #EFF6FF;
        
        --status-success: #16A34A;
        --status-warning: #F59E0B;
        --status-danger: #EF4444;
        --status-info: #2563EB;

        /* Legacy Aliases */
        --accent-blue: #2563EB;
        --accent-dark-blue: #0F172A;
        --accent-emerald: #16A34A;
        --accent-amber: #F59E0B;
        --accent-red: #EF4444;
        
        --hud-bg: linear-gradient(135deg, #FFFFFF 0%, #F7F8FA 50%, #F1F3F5 100%);
        --hud-border: #E2E8F0;
        --user-card-bg: #FFFFFF;
        --user-card-border: #E2E8F0;
        --empty-state-bg: #F7F8FA;
        --empty-state-border: #E2E8F0;
        --alert-bg: #FDF1F1;
        --alert-border: #F7BDBD;
        --alert-text: #D94B4B;
        --input-bg: #FFFFFF;
        --input-border: #CBD5E1;
        --input-text: #171A21;
        --chart-grid: rgba(23, 26, 33, 0.06);
    }

    /* Global Typography */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--text-primary) !important;
        background-color: var(--bg-soft) !important;
        -webkit-font-smoothing: antialiased;
    }
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--surface-border) !important;
        border-radius: 6px !important;
    }

    /* Root App Background */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: var(--bg-soft) !important;
        color: var(--text-primary) !important;
    }
    .main .block-container {
        padding-top: 1.25rem;
        padding-bottom: 3.5rem;
        max-width: 1480px;
        background-color: transparent !important;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: var(--text-primary) !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em;
    }
    .stMarkdown p, .stMarkdown span, .stMarkdown div {
        color: var(--text-secondary);
    }

    /* ===================================================
       SIDEBAR & NAVIGATION STYLING (TASTE SKILL LIGHT)
       =================================================== */
    [data-testid="stSidebar"] {
        background-color: var(--bg-primary) !important;
        border-right: 1px solid var(--surface-border) !important;
        box-shadow: 1px 0 6px rgba(0, 0, 0, 0.02) !important;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: var(--text-muted) !important;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: var(--text-secondary) !important;
    }

    /* Brand Header Card */
    .sidebar-brand-card {
        background: #FFFFFF !important;
        border: 1px solid var(--surface-border);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        transition: border-color 0.2s ease;
    }
    .sidebar-brand-card:hover {
        border-color: #2563EB;
    }
    .brand-icon-box {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        background: linear-gradient(135deg, #0F172A 0%, #2563EB 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        color: #FFFFFF !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25);
        flex-shrink: 0;
    }
    .brand-title {
        font-size: 1.05rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        color: #0F172A !important;
        line-height: 1.15;
        margin: 0;
    }
    .brand-title-accent {
        color: #2563EB !important;
    }
    .brand-subtitle {
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-muted) !important;
        margin-top: 0.15rem;
    }
    .brand-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.18rem 0.5rem;
        border-radius: 9999px;
        font-size: 0.65rem;
        font-weight: 700;
        background: #F0FDF4;
        color: #16A34A !important;
        border: 1px solid #BBF7D0;
        flex-shrink: 0;
    }
    .pulse-dot-emerald {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #16A34A;
        display: inline-block;
    }

    /* Executive User Profile Card */
    .sidebar-user-card {
        background: #FFFFFF !important;
        border: 1px solid var(--user-card-border) !important;
        border-radius: 12px;
        padding: 0.8rem 0.95rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.03);
        display: flex;
        align-items: center;
        gap: 0.75rem;
        transition: all 0.2s ease;
    }
    .sidebar-user-card:hover {
        border-color: #2563EB;
        transform: translateY(-1px);
    }
    .user-avatar-badge {
        width: 36px;
        height: 36px;
        border-radius: 8px;
        background: linear-gradient(135deg, #0F172A 0%, #2563EB 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #ffffff !important;
        font-weight: 800;
        font-size: 0.85rem;
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.2);
        flex-shrink: 0;
    }
    .user-card-name {
        font-weight: 700;
        font-size: 0.88rem;
        color: var(--text-primary) !important;
        line-height: 1.2;
    }
    .user-card-email {
        font-size: 0.72rem;
        color: var(--text-muted) !important;
        margin-bottom: 0.25rem;
    }
    .user-card-location {
        font-size: 0.7rem;
        color: var(--text-secondary) !important;
    }

    /* Active Operational Category Banner */
    .sidebar-category-card {
        background: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-left: 3.5px solid #2563EB !important;
        border-radius: 8px;
        padding: 0.45rem 0.75rem;
        margin-top: 0.35rem;
        margin-bottom: 0.65rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    }
    .sidebar-category-dot {
        width: 5px;
        height: 5px;
        border-radius: 50%;
        background-color: #2563EB;
    }
    .sidebar-category-label {
        font-size: 0.64rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-muted) !important;
    }
    .sidebar-category-title {
        font-size: 0.74rem;
        font-weight: 800;
        color: #0F172A !important;
        margin-top: 0.12rem;
    }

    /* Modern Light Navigation Buttons */
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] {
        display: flex !important;
        flex-direction: column !important;
        gap: 4px !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label {
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
        padding: 0.6rem 0.85rem !important;
        border-radius: 8px !important;
        margin: 0 !important;
        background: #FFFFFF !important;
        border: 1px solid var(--surface-border) !important;
        border-left: 3px solid transparent !important;
        cursor: pointer !important;
        transition: all 0.18s cubic-bezier(.2, .8, .2, 1) !important;
    }
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label div[data-testid="stMarkdownContainer"] p {
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
        line-height: 1.35 !important;
        margin: 0 !important;
    }
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        background: #F8FAFC !important;
        border-color: #CBD5E1 !important;
        transform: translateX(3px) !important;
    }
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:hover div[data-testid="stMarkdownContainer"] p {
        color: #2563EB !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked),
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"],
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input[aria-checked="true"]),
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input[type="radio"]:checked),
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label[aria-checked="true"] {
        background: linear-gradient(90deg, rgba(37, 99, 235, 0.08) 0%, #FFFFFF 100%) !important;
        border: 1px solid rgba(37, 99, 235, 0.35) !important;
        border-left: 4px solid #2563EB !important;
        box-shadow: 0 1px 4px rgba(37, 99, 235, 0.08) !important;
        transform: translateX(2px) !important;
    }
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) div[data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] div[data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input[aria-checked="true"]) div[data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input[type="radio"]:checked) div[data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label[aria-checked="true"] div[data-testid="stMarkdownContainer"] p {
        color: #0F172A !important;
        font-weight: 700 !important;
    }

    /* Sidebar Telemetry Footer */
    .sidebar-telemetry-box {
        background: var(--bg-soft) !important;
        border: 1px solid var(--surface-border) !important;
        border-radius: 8px;
        padding: 0.65rem 0.8rem;
        margin: 0.75rem 0;
    }
    .telemetry-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 0.73rem;
        margin-bottom: 0.35rem;
    }
    .telemetry-label {
        color: var(--text-muted) !important;
        font-weight: 500;
    }
    .telemetry-val {
        font-weight: 700;
        font-size: 0.7rem;
        color: var(--text-primary) !important;
    }

    /* ===================================================
       METRIC CARDS & COMMERCIAL HUD HEADER
       =================================================== */
    .sv-hud {
        background: #FFFFFF !important;
        border: 1px solid var(--hud-border) !important;
        border-radius: 12px;
        padding: 1.15rem 1.6rem;
        margin-bottom: 1.35rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03), 0 4px 12px rgba(0, 0, 0, 0.02);
    }
    .sv-hud-title {
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        color: var(--brand-navy) !important;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .sv-hud-subtitle {
        font-size: 0.88rem;
        color: var(--text-secondary) !important;
        margin: 0.25rem 0 0 0;
        line-height: 1.45;
    }
    .sv-hud-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 0.8rem;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF !important;
        border: 1px solid var(--surface-border) !important;
        border-radius: 10px;
        padding: 0.85rem 1.1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        transition: transform 0.18s cubic-bezier(.2, .8, .2, 1), border-color 0.18s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: var(--surface-border-hover) !important;
        transform: translateY(-2px);
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
        color: var(--text-muted) !important;
        font-size: 0.78rem !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-weight: 800 !important;
        font-size: 1.7rem !important;
        letter-spacing: -0.02em;
    }

    /* ===================================================
       STATUS PILLS & BADGES (TASTE SKILL PALETTE)
       =================================================== */
    .sv-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.2rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .pill-teal {
        background: #F0FDFA !important;
        color: #0D9488 !important;
        border: 1px solid #99F6E4 !important;
    }
    .pill-navy {
        background: #F1F5F9 !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
    }
    .pill-purple {
        background: #FAF5FF !important;
        color: #7C3AED !important;
        border: 1px solid #E9D5FF !important;
    }
    .pill-emerald, .pill-success {
        background: #F0FDF4 !important;
        color: #16A34A !important;
        border: 1px solid #BBF7D0 !important;
    }
    .pill-amber, .pill-warning {
        background: #FFFBEB !important;
        color: #D97706 !important;
        border: 1px solid #FDE68A !important;
    }
    .pill-red, .pill-danger {
        background: #FEF2F2 !important;
        color: #EF4444 !important;
        border: 1px solid #FECACA !important;
    }
    .pill-blue, .pill-info {
        background: #EFF6FF !important;
        color: #2563EB !important;
        border: 1px solid #BFDBFE !important;
    }
    .pill-slate {
        background: #F8FAFC !important;
        color: #64748B !important;
        border: 1px solid #E2E8F0 !important;
    }

    /* ===================================================
       STREAMLIT TABS & NAVIGATION
       =================================================== */
    .stTabs [data-baseweb="tab-list"] {
        background-color: var(--bg-secondary) !important;
        border-radius: 8px;
        padding: 0.25rem 0.35rem;
        gap: 0.2rem;
        border: 1px solid var(--surface-border) !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px !important;
        color: var(--text-secondary) !important;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 0.45rem 1rem !important;
        background-color: transparent !important;
        border: none !important;
        transition: all 0.18s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--brand-teal) !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: var(--brand-teal) !important;
        font-weight: 700 !important;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.05);
    }

    /* ===================================================
       INPUTS, FORMS, SELECTBOXES, TEXT AREAS
       =================================================== */
    input, textarea, .stTextInput input, .stTextArea textarea,
    [data-baseweb="input"] input, [data-baseweb="select"] {
        background-color: var(--input-bg) !important;
        color: var(--input-text) !important;
        border: 1px solid var(--input-border) !important;
        border-radius: 8px !important;
        font-size: 0.88rem !important;
    }
    input:focus, textarea:focus {
        border-color: var(--brand-teal) !important;
        box-shadow: 0 0 0 2px rgba(15, 139, 141, 0.15) !important;
    }
    [data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {
        background-color: #FFFFFF !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--surface-border) !important;
        border-radius: 8px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    }
    [role="option"] {
        color: var(--text-primary) !important;
        background-color: #FFFFFF !important;
        font-size: 0.85rem !important;
    }
    [role="option"]:hover, [aria-selected="true"] {
        background-color: var(--brand-light-blue) !important;
        color: var(--brand-teal) !important;
    }

    /* ===================================================
       BUTTONS
       =================================================== */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.85rem;
        transition: all 0.18s cubic-bezier(.2, .8, .2, 1);
        padding: 0.45rem 1rem;
    }
    .stButton > button[kind="primary"] {
        background: #2563EB !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 1px 4px rgba(37, 99, 235, 0.25);
    }
    .stButton > button[kind="primary"]:hover {
        background: #1D4ED8 !important;
        color: #FFFFFF !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    .stButton > button[kind="secondary"], .stButton > button:not([kind="primary"]) {
        background: #FFFFFF !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--input-border) !important;
    }
    .stButton > button:not([kind="primary"]):hover {
        border-color: #2563EB !important;
        color: #2563EB !important;
        background: #F8FAFC !important;
        transform: translateY(-1px);
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.05);
    }

    /* ===================================================
       EXPANDERS & CARDS
       =================================================== */
    details[data-testid="stExpander"] {
        background: #FFFFFF !important;
        border: 1px solid var(--surface-border) !important;
        border-radius: 10px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    }
    details[data-testid="stExpander"] summary {
        color: var(--text-primary) !important;
        font-weight: 600;
    }

    /* ===================================================
       TABLES & DATAFRAMES (LIGHT HIGH CONTRAST)
       =================================================== */
    div[data-testid="stDataFrame"], .stTable {
        background-color: #FFFFFF !important;
        border: 1px solid var(--surface-border) !important;
        border-radius: 10px;
        overflow: hidden;
    }
    div[data-testid="stDataFrame"] div, div[data-testid="stDataFrame"] span,
    div[data-testid="stDataFrame"] td, div[data-testid="stDataFrame"] th {
        color: var(--text-primary) !important;
    }

    /* ===================================================
       CHAT MESSAGES & HIGHLIGHT BOXES
       =================================================== */
    [data-testid="stChatMessage"] {
        background-color: #FFFFFF !important;
        border: 1px solid var(--surface-border) !important;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    }
    .rag-highlight-box {
        background: #FFFFFF !important;
        border: 1px solid rgba(15, 139, 141, 0.3) !important;
        border-left: 4px solid var(--brand-teal) !important;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin: 0.7rem 0;
    }
    .rag-reader-panel {
        background: #FFFFFF !important;
        border: 1px solid var(--surface-border) !important;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.03);
    }

    /* Alert Banners */
    .alert-banner {
        background: #FEF2F2 !important;
        border-left: 4px solid var(--status-danger) !important;
        border: 1px solid #FECACA;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.8rem 0;
        color: #B91C1C !important;
        font-size: 0.88rem;
        font-weight: 500;
    }

    /* Streamlit Alert Callouts */
    div[data-testid="stAlert"] {
        border-radius: 8px !important;
        border-width: 1px !important;
        border-style: solid !important;
        padding: 0.75rem 1rem !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.02) !important;
    }
    div[data-testid="stAlert"]:has([data-testid="stNotificationContentError"]) {
        background-color: #FEF2F2 !important;
        border-color: #FECACA !important;
        color: #B91C1C !important;
    }
    div[data-testid="stAlert"]:has([data-testid="stNotificationContentSuccess"]) {
        background-color: #F0FDF4 !important;
        border-color: #BBF7D0 !important;
        color: #15803D !important;
    }
    div[data-testid="stAlert"]:has([data-testid="stNotificationContentInfo"]) {
        background-color: #EFF6FF !important;
        border-color: #BFDBFE !important;
        color: #1D4ED8 !important;
    }
    div[data-testid="stAlert"]:has([data-testid="stNotificationContentWarning"]) {
        background-color: #FFFBEB !important;
        border-color: #FDE68A !important;
        color: #B45309 !important;
    }

    /* Responsive Layout & Overflow Protection */
    .main .block-container {
        max-width: 100% !important;
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        overflow-x: hidden !important;
    }
    img, video, canvas, svg {
        max-width: 100% !important;
        height: auto !important;
    }
    div[data-testid="column"] {
        min-width: 0 !important;
    }
    .stTable, div[data-testid="stDataFrame"] {
        max-width: 100% !important;
        overflow-x: auto !important;
    }

    /* Empty State */
    .empty-state {
        background: #FFFFFF;
        border: 1px dashed var(--surface-border);
        border-radius: 12px;
        padding: 2.75rem 1.5rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.02);
    }

    /* Divider */
    hr {
        border-color: var(--surface-border) !important;
        margin: 1.25rem 0 !important;
    }

    /* Universal light overrides for legacy dark inline styles */
    div[style*="background:#111827"],
    div[style*="background: #111827"],
    div[style*="background:#0b1329"],
    div[style*="background: #0b1329"],
    div[style*="background:#0f172a"],
    div[style*="background: #0f172a"],
    div[style*="background:#1e293b"],
    div[style*="background: #1e293b"] {
        background: #FFFFFF !important;
        border-color: #E2E8F0 !important;
        color: #171A21 !important;
    }
    div[style*="color:#f8fafc"],
    div[style*="color: #f8fafc"],
    span[style*="color:#f8fafc"],
    span[style*="color: #f8fafc"],
    div[style*="color:#e2e8f0"],
    div[style*="color: #e2e8f0"] {
        color: #171A21 !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ==========================================
# 4. PLOTLY CHART THEME INTEGRATION
# ==========================================
def get_chart_theme(theme_mode: Optional[str] = None) -> Dict[str, Any]:
    """Returns Plotly layout colorway and background configuration matching Taste Skill light theme."""
    pal = LIGHT_PALETTE

    return {
        "theme_mode": "Light Mode",
        "template": "plotly_white",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font_family": "Plus Jakarta Sans, Inter, sans-serif",
        "font_color": "#171A21",
        "muted_color": "#87909D",
        "grid_color": "rgba(23, 26, 33, 0.06)",
        "accent_emerald": pal["status_success"],
        "accent_blue": pal["brand_teal"],
        "accent_red": pal["status_danger"],
        "accent_amber": pal["status_warning"],
        "radar_benchmark_fill": pal["chart_radar_benchmark_fill"],
        "radar_benchmark_line": pal["chart_radar_benchmark_line"],
        "radar_actual_fill": pal["chart_radar_actual_fill"],
        "radar_actual_line": pal["chart_radar_actual_line"],
    }


def apply_chart_theme(fig, theme_mode: Optional[str] = None):
    """
    Modifies and returns a Plotly figure with high-contrast font, dark axes,
    light grid lines, and transparent background adhering to the Taste Skill light theme.
    """
    if fig is None:
        return None

    ct = get_chart_theme(theme_mode)

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=ct["paper_bgcolor"],
        plot_bgcolor=ct["plot_bgcolor"],
        font=dict(
            family=ct["font_family"],
            color="#171A21"
        ),
        title=dict(font=dict(color="#0F172A", size=14)),
        legend=dict(font=dict(color="#0F172A"))
    )

    # Standard Cartesian axes update
    fig.update_xaxes(
        gridcolor="#F1F5F9",
        zerolinecolor="#E2E8F0",
        color="#0F172A",
        tickfont=dict(color="#64748B", size=11),
        title_font=dict(color="#0F172A", size=12)
    )
    fig.update_yaxes(
        gridcolor="#F1F5F9",
        zerolinecolor="#E2E8F0",
        color="#0F172A",
        tickfont=dict(color="#64748B", size=11),
        title_font=dict(color="#0F172A", size=12)
    )

    # Radar / Polar charts update
    if hasattr(fig, "layout") and fig.layout.polar is not None:
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    color="#64748B",
                    gridcolor="#E2E8F0",
                    tickfont=dict(color="#64748B")
                ),
                angularaxis=dict(
                    color="#0F172A",
                    gridcolor="#E2E8F0",
                    tickfont=dict(color="#0F172A")
                )
            )
        )

    return fig
