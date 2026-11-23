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
    "bg_soft": "#F7F8FA",
    "bg_secondary": "#F1F3F5",
    "surface": "#FFFFFF",
    "surface_border": "#E2E8F0",
    "surface_border_hover": "#0F8B8D",
    "text_primary": "#171A21",
    "text_secondary": "#5B6472",
    "text_muted": "#87909D",
    
    # Brand Accents
    "brand_navy": "#172B4D",
    "brand_teal": "#0F8B8D",
    "brand_purple": "#7357D9",
    "brand_light_blue": "#EAF2FF",
    
    # Status Colors
    "status_success": "#16845B",
    "status_warning": "#D99A22",
    "status_danger": "#D94B4B",
    "status_info": "#3276C8",
    
    # Legacy Color Mappings for backward-compatibility
    "accent_cyan": "#0F8B8D",
    "accent_blue": "#0F8B8D",
    "accent_dark_blue": "#172B4D",
    "accent_emerald": "#16845B",
    "accent_red": "#D94B4B",
    "accent_amber": "#D99A22",
    
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
    "chart_grid": "rgba(23, 26, 33, 0.06)",
    "chart_radar_benchmark_fill": "rgba(15, 139, 141, 0.08)",
    "chart_radar_benchmark_line": "rgba(15, 139, 141, 0.45)",
    "chart_radar_actual_fill": "rgba(22, 132, 91, 0.20)",
    "chart_radar_actual_line": "#16845B",
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
        --bg-soft: #F7F8FA;
        --bg-secondary: #F1F3F5;
        --surface: #FFFFFF;
        --surface-border: #E2E8F0;
        --surface-border-hover: #0F8B8D;
        --text-primary: #171A21;
        --text-secondary: #5B6472;
        --text-muted: #87909D;
        
        --brand-navy: #172B4D;
        --brand-teal: #0F8B8D;
        --brand-purple: #7357D9;
        --brand-light-blue: #EAF2FF;
        
        --status-success: #16845B;
        --status-warning: #D99A22;
        --status-danger: #D94B4B;
        --status-info: #3276C8;

        /* Legacy Aliases */
        --accent-blue: #0F8B8D;
        --accent-dark-blue: #172B4D;
        --accent-emerald: #16845B;
        --accent-amber: #D99A22;
        --accent-red: #D94B4B;
        
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
        border-color: var(--brand-teal);
    }
    .brand-icon-box {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        background: linear-gradient(135deg, #172B4D 0%, #0F8B8D 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        color: #FFFFFF !important;
        box-shadow: 0 2px 8px rgba(15, 139, 141, 0.25);
        flex-shrink: 0;
    }
    .brand-title {
        font-size: 1.05rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        color: var(--brand-navy) !important;
        line-height: 1.15;
        margin: 0;
    }
    .brand-title-accent {
        color: var(--brand-teal) !important;
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
        background: #E8F6F0;
        color: #16845B !important;
        border: 1px solid #A4DFC5;
        flex-shrink: 0;
    }
    .pulse-dot-emerald {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: var(--status-success);
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
        border-color: var(--surface-border-hover);
        transform: translateY(-1px);
    }
    .user-avatar-badge {
        width: 36px;
        height: 36px;
        border-radius: 8px;
        background: linear-gradient(135deg, #172B4D 0%, #0F8B8D 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #ffffff !important;
        font-weight: 800;
        font-size: 0.85rem;
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
        background: var(--bg-secondary) !important;
        border: 1px solid rgba(15, 139, 141, 0.2) !important;
        border-left: 3.5px solid var(--brand-teal) !important;
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
        background-color: var(--brand-teal);
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
        color: var(--brand-navy) !important;
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
        background: var(--bg-soft) !important;
        border-color: #CBD5E1 !important;
        transform: translateX(3px) !important;
    }
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:hover div[data-testid="stMarkdownContainer"] p {
        color: var(--brand-teal) !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked),
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"],
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input[aria-checked="true"]),
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input[type="radio"]:checked),
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label[aria-checked="true"] {
        background: linear-gradient(90deg, rgba(15, 139, 141, 0.08) 0%, #FFFFFF 100%) !important;
        border: 1px solid rgba(15, 139, 141, 0.35) !important;
        border-left: 4px solid var(--brand-teal) !important;
        box-shadow: 0 1px 4px rgba(15, 139, 141, 0.08) !important;
        transform: translateX(2px) !important;
    }
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) div[data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] div[data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input[aria-checked="true"]) div[data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input[type="radio"]:checked) div[data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label[aria-checked="true"] div[data-testid="stMarkdownContainer"] p {
        color: var(--text-primary) !important;
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
        background: #E6F7F7 !important;
        color: #0F8B8D !important;
        border: 1px solid #A2E5E6 !important;
    }
    .pill-navy {
        background: #EAEFF5 !important;
        color: #172B4D !important;
        border: 1px solid #B4C3D8 !important;
    }
    .pill-purple {
        background: #F2EFFC !important;
        color: #7357D9 !important;
        border: 1px solid #D3C6F7 !important;
    }
    .pill-emerald, .pill-success {
        background: #E8F6F0 !important;
        color: #16845B !important;
        border: 1px solid #A4DFC5 !important;
    }
    .pill-amber, .pill-warning {
        background: #FEF7EA !important;
        color: #D99A22 !important;
        border: 1px solid #F8D89B !important;
    }
    .pill-red, .pill-danger {
        background: #FDF1F1 !important;
        color: #D94B4B !important;
        border: 1px solid #F7BDBD !important;
    }
    .pill-blue, .pill-info {
        background: #EDF4FC !important;
        color: #3276C8 !important;
        border: 1px solid #B4D4F3 !important;
    }
    .pill-slate {
        background: #F1F3F5 !important;
        color: #5B6472 !important;
        border: 1px solid #D6DCE2 !important;
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
        background: #0F8B8D !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 1px 4px rgba(15, 139, 141, 0.25);
    }
    .stButton > button[kind="primary"]:hover {
        background: #0c7274 !important;
        color: #FFFFFF !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(15, 139, 141, 0.3);
    }
    .stButton > button[kind="secondary"], .stButton > button:not([kind="primary"]) {
        background: #FFFFFF !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--input-border) !important;
    }
    .stButton > button:not([kind="primary"]):hover {
        border-color: var(--brand-teal) !important;
        color: var(--brand-teal) !important;
        background: var(--bg-soft) !important;
        transform: translateY(-1px);
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
        background: #FDF1F1 !important;
        border-left: 4px solid var(--status-danger) !important;
        border: 1px solid #F7BDBD;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.8rem 0;
        color: #D94B4B !important;
        font-size: 0.88rem;
        font-weight: 500;
    }

    /* Empty State */
    .empty-state {
        background: #FFFFFF;
        border: 1px dashed var(--surface-border);
        border-radius: 12px;
        padding: 3rem 1.5rem;
        text-align: center;
        margin: 1rem 0;
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
        title=dict(font=dict(color="#171A21", size=14)),
        legend=dict(font=dict(color="#171A21"))
    )

    # Standard Cartesian axes update
    fig.update_xaxes(
        gridcolor="rgba(23, 26, 33, 0.06)",
        zerolinecolor="rgba(23, 26, 33, 0.06)",
        color="#171A21",
        tickfont=dict(color="#5B6472", size=11),
        title_font=dict(color="#171A21", size=12)
    )
    fig.update_yaxes(
        gridcolor="rgba(23, 26, 33, 0.06)",
        zerolinecolor="rgba(23, 26, 33, 0.06)",
        color="#171A21",
        tickfont=dict(color="#5B6472", size=11),
        title_font=dict(color="#171A21", size=12)
    )

    # Radar / Polar charts update
    if hasattr(fig, "layout") and fig.layout.polar is not None:
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    color="#5B6472",
                    gridcolor="rgba(23, 26, 33, 0.08)",
                    tickfont=dict(color="#5B6472")
                ),
                angularaxis=dict(
                    color="#171A21",
                    gridcolor="rgba(23, 26, 33, 0.08)",
                    tickfont=dict(color="#171A21")
                )
            )
        )

    return fig
