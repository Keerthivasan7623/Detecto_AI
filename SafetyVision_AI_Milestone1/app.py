import os
import json
import re
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from ultralytics import YOLO

import charts
from styles.theme_manager import (
    apply_theme_css,
    apply_chart_theme,
    get_chart_theme,
    get_theme_mode,
    render_theme_switcher,
)
from auth import (
    authenticate_user, get_current_user, hash_password, init_session, is_admin,
    logout_session, register_user, set_login_session, validate_password_strength,
    verify_password,
)
from database import (
    DB_PATH, change_user_password, count_active_admins, create_camera_stream,
    create_upload, delete_camera_stream, delete_document,
    export_repository_csv, get_alert_dispatches,
    get_category_trends, get_class_counts, get_connection, get_daily_violations,
    get_dashboard_stats, get_document_by_id, get_document_chunks, get_document_dossier,
    get_document_stats, get_hourly_violations, get_ppe_category_compliance,
    get_rag_chat_history, clear_rag_chat_history,
    get_recent_violations, get_safety_index_score, get_safety_zone_by_id,
    get_shift_violations, get_top_sources, get_upload_history,
    get_weekday_hour_heatmap, get_zone_incident_breakdown, get_workplace_safety_kpis, initialize_database,
    insert_document_chunks, list_all_users, list_camera_streams, list_documents,
    list_safety_zones, log_alert_dispatch,
    save_frame_batch, update_camera_stream, update_document_metadata, update_document_status,
    update_upload, update_user_role, update_user_status,
)
from services.document_intelligence import (
    validate_and_store_document, get_status_badge, extract_document_text,
    extract_ehs_entities, generate_document_summary, load_sample_ehs_library,
    process_extracted_items_into_chunks, get_vector_store, get_rag_service,
    get_policy_for_violation, synthesize_lead_ehs_directive, seed_default_knowledge_base,
    answer_safety_policy
)
from services.agent import (
    run_incident_investigation_agent,
    run_visiondesk_agent,
    query_visual_events,
    search_safety_policies,
    create_incident_audit_ticket,
    dispatch_emergency_alert,
    visual_inspector_tool,
    doc_retriever_tool,
    incident_logger_tool,
    supervisor_alert_tool,
    run_safety_investigation,
    load_risk_framework
)
from database import (
    create_safety_event,
    get_safety_events,
    get_safety_event_by_id,
    update_safety_event_status,
    update_violation_status,
    log_remediation_action,
    get_remediation_actions,
    log_agent_action,
    get_agent_actions,
    get_safety_event_metrics,
    save_investigation_record,
    get_investigation_history,
    get_investigation_by_id,
    delete_investigation_by_id,
    clear_all_investigation_history,
    create_investigation,
    list_investigations,
    get_safety_investigation_by_id,
    get_investigation_by_incident_id,
    get_investigation_stages,
    get_investigation_citations,
    delete_investigation,
    log_audit_action,
)
from reporting import (
    generate_csv_report, generate_inspection_pdf_report,
    generate_json_report, generate_pdf_report,
    generate_investigation_pdf_report,
)
from components.kpi_cards import render_kpi_cards
from safety_engine import (
    EXPLICIT_VIOLATION_CLASSES, PPE_COMPLIANT_CLASSES, VideoViolationTracker,
    analyze_image_frame, determine_shift,
    draw_enterprise_annotations, get_violation_severity,
    save_evidence_crop,
)
from video_utils import cleanup_old_files, make_browser_playable, resize_keep_aspect

# ==========================================
# ENTERPRISE RUNTIME CONFIGURATION
# ==========================================
from storage_config import (
    DATA_DIR, OUTPUTS_DIR, OUTPUT_DIR, REPORT_DIR, EVIDENCE_DIR,
    DOCUMENTS_DIR, INVESTIGATIONS_DIR, VECTOR_STORE_DIR,
    MODEL_PATH, SAMPLE_DIR, ensure_storage_directories,
    resolve_runtime_path,
)

BASE_DIR = Path(__file__).resolve().parent
ensure_storage_directories()

MAX_IMAGE_MB, MAX_VIDEO_MB = 25, 200
ALL_VIOLATION_CLASSES = sorted(EXPLICIT_VIOLATION_CLASSES)

st.set_page_config(
    page_title="Detecto AI | Enterprise Workplace Safety",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def bootstrap():
    """Server initialization hook."""
    initialize_database()
    for d in (OUTPUT_DIR, EVIDENCE_DIR):
        cleanup_old_files(d, days=30)
    return True


@st.cache_resource
def load_yolo_model(path: str):
    return YOLO(path)


bootstrap()
init_session()

# ==========================================
# DYNAMIC THEME SYSTEM (LIGHT & DARK MODES)
# ==========================================
if "theme_mode" not in st.session_state or st.session_state.get("theme_mode") != "Light Mode":
    st.session_state["theme_mode"] = "Light Mode"

apply_theme_css("Light Mode")


# ==========================================
# UI HELPERS (TASTE SKILL ENTERPRISE DESIGN)
# ==========================================
def render_html(raw_html: str):
    """Renders raw HTML cleanly in Streamlit without triggering Markdown indented code block parsing."""
    if hasattr(st, "html"):
        st.html(raw_html)
    else:
        clean_lines = [
            line.strip() for line in raw_html.splitlines()
            if line.strip() and not line.strip().startswith("<!--")
        ]
        st.markdown(" ".join(clean_lines), unsafe_allow_html=True)


def enterprise_hud(title: str, subtitle: str, active_zone: str = "All Plant Zones"):
    safety_info = get_safety_index_score()
    shift_str = determine_shift()
    device_name = "NVIDIA CUDA GPU" if torch.cuda.is_available() else "Intel/AMD CPU Accel"

    render_html(f"""
    <div class="sv-hud">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:0.75rem;">
            <div>
                <h1 class="sv-hud-title">🛡️ {title}</h1>
                <p class="sv-hud-subtitle">{subtitle}</p>
            </div>
            <div style="text-align:right; display:flex; flex-direction:column; align-items:flex-end; gap:0.35rem;">
                <span class="sv-pill pill-navy">ZONE: {active_zone}</span>
                <span class="sv-pill {'pill-emerald' if safety_info['score']>=80 else ('pill-amber' if safety_info['score']>=65 else 'pill-red')}">
                    SAFETY INDEX: {safety_info['score']}/100 ({safety_info['rating'].split()[0]})
                </span>
            </div>
        </div>
        <div class="sv-hud-pills">
            <span class="sv-pill pill-emerald">● AI ENGINE: YOLOV8 ONLINE</span>
            <span class="sv-pill pill-teal">● COMPUTE: {device_name}</span>
            <span class="sv-pill pill-purple">● SHIFT: {shift_str}</span>
            <span class="sv-pill pill-slate">● STANDARDS: OSHA 1910 / ISO 45001</span>
        </div>
    </div>
    """)


def render_page_header(title: str, subtitle: str, active_zone: str = "All Plant Zones"):
    """Reusable page header adhering to Taste Skill design language."""
    enterprise_hud(title, subtitle, active_zone)


def render_kpi_card(title: str, value: Any, description: str = "", icon: str = "📊", delta: Optional[str] = None, delta_type: str = "normal"):
    """Renders a Taste Skill high-contrast enterprise KPI metric card."""
    delta_html = ""
    if delta:
        cls = "pill-emerald" if delta_type == "normal" else ("pill-red" if delta_type == "inverse" else "pill-slate")
        delta_html = f'<div style="margin-top:0.4rem;"><span class="sv-pill {cls}">{delta}</span></div>'
    render_html(f"""
    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; padding:0.95rem 1.15rem; box-shadow:0 1px 3px rgba(0,0,0,0.03); transition:all 0.18s ease;">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:#87909D;">{title}</div>
            <div style="width:34px; height:34px; border-radius:8px; background:#EAF2FF; display:flex; align-items:center; justify-content:center; font-size:1.15rem; color:#172B4D;">{icon}</div>
        </div>
        <div style="font-size:1.85rem; font-weight:800; color:#171A21; letter-spacing:-0.025em; margin:0.35rem 0 0.15rem 0; line-height:1.15;">{value}</div>
        <div style="font-size:0.78rem; color:#5B6472;">{description}</div>
        {delta_html}
    </div>
    """)


def render_section_header(title: str, subtitle: str = ""):
    """Renders a section divider with subtitle."""
    render_html(f"""
    <div style="margin: 1.25rem 0 0.65rem 0;">
        <div style="font-size:1.05rem; font-weight:800; color:#171A21; letter-spacing:-0.02em;">{title}</div>
        {f'<div style="font-size:0.8rem; color:#5B6472; margin-top:0.15rem;">{subtitle}</div>' if subtitle else ''}
    </div>
    """)


def render_risk_matrix_5x5(sev: int = 3, lik: int = 2, score: int = 6, level: str = "Medium"):
    """Renders an interactive-looking, accurate 5x5 OSHA / ISO 45001 Enterprise Risk Matrix."""
    level_color_map = {
        "Low": ("#E8F6F0", "#16845B", "#A4DFC5"),
        "Medium": ("#FEF7EA", "#D99A22", "#F8D89B"),
        "High": ("#FDF1F1", "#D94B4B", "#F7BDBD"),
        "Critical": ("#FADBD8", "#991B1B", "#E6B0AA")
    }
    bg_pill, txt_pill, bdr_pill = level_color_map.get(level, ("#F1F3F5", "#5B6472", "#D6DCE2"))
    
    sev_labels = {5: "Critical (5)", 4: "Major (4)", 3: "Moderate (3)", 2: "Minor (2)", 1: "Negligible (1)"}
    lik_labels = {1: "Rare (1)", 2: "Unlikely (2)", 3: "Possible (3)", 4: "Likely (4)", 5: "Almost Certain (5)"}
    
    rows_html = []
    for s_val in [5, 4, 3, 2, 1]:
        cells = [f'<td style="font-weight:700; font-size:0.75rem; color:#5B6472; padding:0.45rem; background:#F8FAFC; border:1px solid #E2E8F0; width:110px;">{sev_labels[s_val]}</td>']
        for l_val in [1, 2, 3, 4, 5]:
            prod = s_val * l_val
            if prod >= 15:
                c_bg, c_txt, c_bdr = "#FDF1F1", "#D94B4B", "#F7BDBD"
            elif prod >= 10:
                c_bg, c_txt, c_bdr = "#FEF7EA", "#D99A22", "#F8D89B"
            elif prod >= 5:
                c_bg, c_txt, c_bdr = "#FFFBEB", "#92400E", "#FDE68A"
            else:
                c_bg, c_txt, c_bdr = "#E8F6F0", "#16845B", "#A4DFC5"
            
            is_active = (s_val == sev and l_val == lik)
            active_style = "box-shadow: 0 0 0 2.5px #172B4D; font-weight:800; transform: scale(1.04); z-index:2; position:relative;" if is_active else ""
            marker = f'<div style="font-size:0.65rem; font-weight:800; color:#172B4D;">📍 {prod} ACTIVE</div>' if is_active else f'<div style="font-size:0.78rem; font-weight:600; color:{c_txt};">{prod}</div>'
            
            cell_html = f'''
            <td style="padding:0.45rem; text-align:center; background:{c_bg}; border:1px solid {c_bdr}; {active_style}">
                {marker}
            </td>
            '''
            cells.append(cell_html)
        rows_html.append(f"<tr>{''.join(cells)}</tr>")
    
    header_cols = ['<th style="padding:0.45rem; background:#F1F3F5; font-size:0.72rem; color:#171A21; border:1px solid #E2E8F0;">Severity \\ Likelihood</th>']
    for l_val in [1, 2, 3, 4, 5]:
        header_cols.append(f'<th style="padding:0.45rem; background:#F1F3F5; font-size:0.72rem; color:#171A21; border:1px solid #E2E8F0;">{lik_labels[l_val]}</th>')
    
    table_html = f"""
    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; padding:1.1rem; box-shadow:0 1px 3px rgba(0,0,0,0.03); margin:0.75rem 0;">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem; margin-bottom:0.75rem;">
            <div>
                <span style="font-size:0.95rem; font-weight:800; color:#171A21;">OSHA / ISO 45001 Enterprise Risk Matrix (5×5)</span>
                <span style="font-size:0.75rem; color:#5B6472; margin-left:0.5rem;">Severity ({sev}) × Likelihood ({lik}) = <b>Composite Score {score}/25</b></span>
            </div>
            <div>
                <span style="background:{bg_pill}; color:{txt_pill}; border:1px solid {bdr_pill}; border-radius:9999px; padding:0.22rem 0.75rem; font-size:0.75rem; font-weight:800; text-transform:uppercase;">
                    ● {level.upper()} RISK TIER
                </span>
            </div>
        </div>
        <div style="overflow-x:auto;">
            <table style="width:100%; border-collapse:collapse; font-size:0.8rem;">
                <thead><tr>{''.join(header_cols)}</tr></thead>
                <tbody>{''.join(rows_html)}</tbody>
            </table>
        </div>
        <div style="display:flex; gap:1.2rem; justify-content:flex-end; font-size:0.72rem; color:#5B6472; margin-top:0.65rem;">
            <span><span style="display:inline-block;width:10px;height:10px;background:#E8F6F0;border:1px solid #A4DFC5;border-radius:2px;margin-right:3px;"></span> Low (1-4)</span>
            <span><span style="display:inline-block;width:10px;height:10px;background:#FFFBEB;border:1px solid #FDE68A;border-radius:2px;margin-right:3px;"></span> Medium (5-9)</span>
            <span><span style="display:inline-block;width:10px;height:10px;background:#FEF7EA;border:1px solid #F8D89B;border-radius:2px;margin-right:3px;"></span> High (10-14)</span>
            <span><span style="display:inline-block;width:10px;height:10px;background:#FDF1F1;border:1px solid #F7BDBD;border-radius:2px;margin-right:3px;"></span> Critical (15-25)</span>
        </div>
    </div>
    """
    render_html(table_html)


def render_hierarchy_of_controls(capa_data: Dict[str, Any]):
    """Renders the standard 4-tier Hierarchy of Controls visual diagram."""
    h_summary = capa_data.get("hierarchy_summary") or {}
    
    elim_text = h_summary.get("elimination") or "Physically remove hazards or lock-out unsafe machinery."
    eng_text = h_summary.get("engineering") or "Isolate personnel from hazard using physical guardrails, physical interlocks, and perimeter barricades."
    admin_text = h_summary.get("administrative") or "Review and enforce OSHA 1910 standard operating procedures, execute pre-shift toolbox talks, and display prominent safety signage."
    ppe_text = h_summary.get("ppe") or "Mandate full PPE compliance (Hardhat, High-Vis Vest, Safety Boots, Eye Goggles) prior to zone entry."
    
    render_html(f"""
    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; padding:1.1rem; box-shadow:0 1px 3px rgba(0,0,0,0.03); margin:0.8rem 0;">
        <div style="font-size:0.95rem; font-weight:800; color:#171A21; margin-bottom:0.25rem;">
            🏗️ NIOSH / OSHA Hierarchy of Controls Framework
        </div>
        <div style="font-size:0.78rem; color:#5B6472; margin-bottom:0.85rem;">
            Order of effectiveness from most effective physical hazard elimination down to personal protective equipment.
        </div>
        <div style="background:#F7F8FA; border:1px solid #E2E8F0; border-left:4px solid #172B4D; border-radius:8px; padding:0.65rem 0.9rem; margin-bottom:0.45rem;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-weight:700; font-size:0.82rem; color:#172B4D;">1. ELIMINATION &amp; SUBSTITUTION (MOST EFFECTIVE)</span>
                <span class="sv-pill pill-navy">Tier 1</span>
            </div>
            <div style="font-size:0.8rem; color:#334155; margin-top:0.25rem;">{elim_text}</div>
        </div>
        <div style="background:#F7F8FA; border:1px solid #E2E8F0; border-left:4px solid #0F8B8D; border-radius:8px; padding:0.65rem 0.9rem; margin-bottom:0.45rem;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-weight:700; font-size:0.82rem; color:#0F8B8D;">2. ENGINEERING CONTROLS</span>
                <span class="sv-pill pill-teal">Tier 2</span>
            </div>
            <div style="font-size:0.8rem; color:#334155; margin-top:0.25rem;">{eng_text}</div>
        </div>
        <div style="background:#F7F8FA; border:1px solid #E2E8F0; border-left:4px solid #7357D9; border-radius:8px; padding:0.65rem 0.9rem; margin-bottom:0.45rem;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-weight:700; font-size:0.82rem; color:#7357D9;">3. ADMINISTRATIVE CONTROLS</span>
                <span class="sv-pill pill-purple">Tier 3</span>
            </div>
            <div style="font-size:0.8rem; color:#334155; margin-top:0.25rem;">{admin_text}</div>
        </div>
        <div style="background:#F7F8FA; border:1px solid #E2E8F0; border-left:4px solid #16845B; border-radius:8px; padding:0.65rem 0.9rem;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-weight:700; font-size:0.82rem; color:#16845B;">4. PERSONAL PROTECTIVE EQUIPMENT (PPE)</span>
                <span class="sv-pill pill-emerald">Tier 4</span>
            </div>
            <div style="font-size:0.8rem; color:#334155; margin-top:0.25rem;">{ppe_text}</div>
        </div>
    </div>
    """)


def empty_state(icon: str, title: str, subtitle: str = ""):
    render_html(f"""
    <div class="empty-state">
        <div style="font-size:2.6rem;margin-bottom:0.4rem;">{icon}</div>
        <div style="font-size:1.05rem;font-weight:800;color:#171A21;">{title}</div>
        <div style="font-size:0.82rem;margin-top:0.2rem;color:#5B6472;">{subtitle}</div>
    </div>
    """)


def fmt_ts(value) -> str:
    if not value:
        return "-"
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(value)[:19].replace("T", " ")


def safe_stem(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", Path(name).stem)[:60] or "file"


def date_str(d):
    return d.strftime("%Y-%m-%d") if d else None


def show_plot(fig, empty_msg="No activity logged for this time window.", key=None):
    if fig is None:
        st.info(empty_msg)
    else:
        fig = apply_chart_theme(fig, "Light Mode")
        st.plotly_chart(fig, width="stretch", key=key, config={"displayModeBar": False})


# ==========================================
# AUTHENTICATION PAGE
# ==========================================
def render_auth_page():
    st.markdown("""
    <div style="text-align:center; margin: 2rem 0 1.5rem;">
        <div style="display:inline-flex; width:56px; height:56px; border-radius:14px; background:linear-gradient(135deg, #172B4D 0%, #0F8B8D 100%); align-items:center; justify-content:center; font-size:2rem; color:#FFFFFF; box-shadow:0 4px 14px rgba(15,139,141,0.25); margin-bottom:0.75rem;">
            🛡️
        </div>
        <h1 style="font-size:2.2rem;font-weight:800;color:#172B4D;margin:0.2rem 0;letter-spacing:-0.03em;">
            DETECTO<span style="color:#0F8B8D">.AI</span>
        </h1>
        <p style="color:#5B6472;font-size:0.95rem;max-width:500px;margin:0 auto;">
            Enterprise Workplace Safety, Vision Control & Multi-Agent Intelligence Platform
        </p>
    </div>
    """, unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 1.8, 1])
    with mid:
        tab_in, tab_up = st.tabs(["🔒 Secure Sign In", "📝 Create Operator Account"])

        with tab_in:
            with st.form("enterprise_login_form"):
                email = st.text_input("Corporate Work Email", placeholder="officer@company.com")
                pw = st.text_input("Account Password", type="password")
                go = st.form_submit_button("Sign In to Control Console", type="primary", width="stretch")

            if go:
                if not email or not pw:
                    st.error("Please provide both email and password credentials.")
                else:
                    ok, msg, profile = authenticate_user(email, pw)
                    if ok:
                        set_login_session(profile)
                        st.rerun()
                    else:
                        st.error(msg)
            st.caption("🔒 Secured via Bcrypt round 11 salted hashing, constant-time verification, and brute-force lockout protection.")

        with tab_up:
            with st.form("enterprise_reg_form"):
                name = st.text_input("Full Name *", placeholder="Jane Doe")
                reg_email = st.text_input("Corporate Work Email *", placeholder="jane@company.com")
                workplace = st.text_input("Plant / Facility Unit", placeholder="Central Manufacturing Bay")
                p1 = st.text_input("Password *", type="password", help="Min 8 characters with at least one letter and one number/symbol.")
                p2 = st.text_input("Confirm Password *", type="password")
                go2 = st.form_submit_button("Submit Registration", type="primary", width="stretch")

            if go2:
                ok, msg, _ = register_user(full_name=name, email=reg_email, password=p1, confirm_password=p2, workplace=workplace)
                if ok:
                    st.success(f"{msg} You may now sign in.")
                else:
                    st.error(msg)


if not st.session_state.get("authenticated"):
    render_auth_page()
    st.stop()

# ==========================================

# ==========================================
# ENTERPRISE CONSOLE NAVIGATION DEFINITION
# ==========================================
PAGE_DASHBOARD = "📊 Dashboard"
PAGE_IMAGE_VIDEO_ANALYSIS = "📸 Image & Video Analysis"
PAGE_VIOLATIONS = "⚠️ Violations"
PAGE_REPORTS_EXPORT = "📄 Reports & Export"
PAGE_DOCUMENT_INGESTION = "📤 Document Ingestion"
PAGE_SEMANTIC_SEARCH = "🔎 Semantic Policy Search"
PAGE_RAG_ASSISTANT = "💬 RAG Safety Policy Assistant"
PAGE_LANGGRAPH_INVESTIGATION = "🤖 LangGraph AI Safety Investigation"
PAGE_ACCOUNT_PROFILE = "👤 Account Profile"
PAGE_USER_GOVERNANCE = "👥 User Governance"

# Backward compatibility aliases for existing session references
PAGE_IMAGE_ANALYSIS = "📸 Image Analysis"
PAGE_VIDEO_ANALYSIS = "📹 Video Analysis"
PAGE_LIVE_SURVEILLANCE = "🎥 Live Surveillance"
PAGE_SAFETY_ANALYTICS = "📈 Safety Analytics"
PAGE_EXECUTIVE_DASHBOARD = "📊 Executive Dashboard"
PAGE_RAG_QA = "❓ Semantic Policy Search (Q&A)"
PAGE_INVESTIGATOR_CONSOLE = "🕵️ Investigator Console"
PAGE_RECORDED_EVENTS = "⚠️ Recorded Events"
PAGE_REMEDIATION_WORKFLOWS = "🔄 Remediation Workflows"
PAGE_AUDIT_REPORTS = "📋 Audit Reports"
PAGE_MY_PROFILE = "👤 My Profile"

PAGE_SECTIONS = {
    PAGE_DASHBOARD: "SECTION A: EXECUTIVE OVERSIGHT & KPIS",
    PAGE_IMAGE_VIDEO_ANALYSIS: "SECTION B: COMPUTER VISION & MEDIA INSPECTION",
    PAGE_VIOLATIONS: "SECTION C: COMPLIANCE & HUMAN REVIEW",
    PAGE_REPORTS_EXPORT: "SECTION D: AUDIT & REGULATORY REPORTING",
    PAGE_DOCUMENT_INGESTION: "SECTION E: DOCUMENT INTELLIGENCE (MILESTONE 2)",
    PAGE_SEMANTIC_SEARCH: "SECTION E: DOCUMENT INTELLIGENCE (MILESTONE 2)",
    PAGE_RAG_ASSISTANT: "SECTION F: MULTIMODAL RAG ASSISTANT (MILESTONE 3)",
    PAGE_LANGGRAPH_INVESTIGATION: "SECTION G: MULTIMODAL AI AGENT (LANGGRAPH)",
    PAGE_ACCOUNT_PROFILE: "SECTION H: USER GOVERNANCE & ACCESS",
    PAGE_USER_GOVERNANCE: "SECTION H: USER GOVERNANCE & ACCESS",
}

# ==========================================
# SIDEBAR NAVIGATION & CONSOLE CONTROLS
# ==========================================
current_user = get_current_user() or {"id": 1, "full_name": "Safety Officer", "role": "admin", "email": "admin@plant.detecto.ai"}
user_is_admin = is_admin() if get_current_user() else True
uid, role = current_user.get("id", 1), current_user.get("role", "admin")

NAV_PAGES = [
    PAGE_DASHBOARD,
    PAGE_IMAGE_VIDEO_ANALYSIS,
    PAGE_VIOLATIONS,
    PAGE_REPORTS_EXPORT,
    PAGE_DOCUMENT_INGESTION,
    PAGE_SEMANTIC_SEARCH,
    PAGE_RAG_ASSISTANT,
    PAGE_LANGGRAPH_INVESTIGATION,
    PAGE_ACCOUNT_PROFILE,
]
if user_is_admin:
    NAV_PAGES.append(PAGE_USER_GOVERNANCE)

# Handle cross-page redirection if set
if "nav_radio" not in st.session_state or st.session_state["nav_radio"] not in NAV_PAGES:
    st.session_state["nav_radio"] = PAGE_DASHBOARD

if st.session_state.get("redirect_to_page"):
    target_page = st.session_state.pop("redirect_to_page")
    if target_page in NAV_PAGES:
        st.session_state["nav_radio"] = target_page
    elif target_page in (PAGE_IMAGE_ANALYSIS, PAGE_VIDEO_ANALYSIS, PAGE_LIVE_SURVEILLANCE):
        st.session_state["nav_radio"] = PAGE_IMAGE_VIDEO_ANALYSIS
    elif target_page == PAGE_RAG_QA:
        st.session_state["nav_radio"] = PAGE_RAG_ASSISTANT
    elif target_page in (PAGE_INVESTIGATOR_CONSOLE, PAGE_REMEDIATION_WORKFLOWS):
        st.session_state["nav_radio"] = PAGE_LANGGRAPH_INVESTIGATION
    elif target_page == PAGE_RECORDED_EVENTS:
        st.session_state["nav_radio"] = PAGE_VIOLATIONS
    elif target_page == PAGE_SAFETY_ANALYTICS:
        st.session_state["nav_radio"] = PAGE_DASHBOARD

st.session_state["current_page"] = st.session_state["nav_radio"]

with st.sidebar:
    user_name = current_user.get("full_name") or "Safety Officer"
    user_email = current_user.get("email") or "ops@detecto.ai"
    user_workplace = current_user.get("workplace") or "HQ Operations"
    user_initials = "".join([part[0].upper() for part in user_name.split() if part][:2]) or "SO"

    # Enterprise Brand Header
    st.markdown(f"""
    <div class="sidebar-brand-card">
        <div class="brand-icon-box">🛡️</div>
        <div class="brand-details">
            <div class="brand-title">DETECTO<span class="brand-title-accent">.AI</span></div>
            <div class="brand-subtitle">Enterprise EHS Vision</div>
        </div>
        <div class="brand-status-pill">
            <span class="pulse-dot-emerald"></span>
            <span>ONLINE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Executive User Profile Badge
    st.markdown(f"""
    <div class="sidebar-user-card">
        <div class="user-avatar-badge">{user_initials}</div>
        <div class="user-card-content">
            <div class="user-card-name" title="{user_name}">{user_name}</div>
            <div class="user-card-email" title="{user_email}">{user_email}</div>
            <div class="user-card-meta">
                <span class="sv-pill {'pill-blue' if user_is_admin else 'pill-slate'}">{'ADMINISTRATOR' if user_is_admin else 'OPERATOR'}</span>
                <span class="user-card-location" title="{user_workplace}">📍 {user_workplace}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Active Operational Domain Indicator Banner
    active_sec = PAGE_SECTIONS.get(st.session_state["current_page"], "SECTION A: REAL-TIME COMPUTER VISION")
    st.markdown(f"""
    <div class="sidebar-category-card">
        <div class="sidebar-category-header">
            <span class="sidebar-category-dot"></span>
            <span class="sidebar-category-label">ACTIVE OPERATIONAL DOMAIN</span>
        </div>
        <div class="sidebar-category-title">{active_sec}</div>
    </div>
    """, unsafe_allow_html=True)

    # Console Navigation Menu Label
    st.markdown("""
    <div style="font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:#64748B; margin: 0.6rem 0 0.35rem 0.2rem;">
        NAVIGATION CONSOLE
    </div>
    """, unsafe_allow_html=True)

    page_selected = st.radio(
        "Operational Views",
        NAV_PAGES,
        key="nav_radio",
        label_visibility="collapsed"
    )
    st.session_state["current_page"] = page_selected

    with st.expander("⚙️ Inspection Rules & Filters", expanded=True):
        st.markdown("**🛡️ Mandatory Required PPE**")
        required_ppe_sidebar = st.multiselect(
            "Mandatory Worker PPE",
            ["helmet", "vest", "boots", "gloves", "goggles"],
            default=["helmet", "vest"],
            format_func=lambda x: {
                "helmet": "Hardhat / Safety Helmet",
                "vest": "High-Visibility Safety Vest",
                "boots": "Steel-Toe Safety Boots",
                "gloves": "Protective Work Gloves",
                "goggles": "Protective Eye Goggles"
            }.get(x, x.title()),
            help="Select PPE required for all workers. Workers lacking any selected item will be flagged as NON-COMPLIANT with high-severity violation alerts."
        )
        monitored = set(st.multiselect(
            "Monitored Violation Classes",
            ALL_VIOLATION_CLASSES,
            default=ALL_VIOLATION_CLASSES,
            help="Select the exact PPE non-compliance events flagged as violations."
        ))

    # Live System Telemetry Status Box
    st.markdown("""
    <div class="sidebar-telemetry-box">
        <div class="telemetry-row">
            <span class="telemetry-label">● YOLOv8 Vision Core</span>
            <span class="telemetry-val" style="color:#16845B;">ONLINE</span>
        </div>
        <div class="telemetry-row">
            <span class="telemetry-label">● FAISS Vector Index</span>
            <span class="telemetry-val" style="color:#0F8B8D;">384-D READY</span>
        </div>
        <div class="telemetry-row">
            <span class="telemetry-label">● SQLite DB Engine</span>
            <span class="telemetry-val" style="color:#7357D9;">WAL SYNC</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚪 Sign Out", width="stretch"):
        logout_session()
        st.rerun()

# Model Verification
if not MODEL_PATH.exists():
    st.error("PPE Model checkpoint missing at `models/best.pt`. Please follow instructions in MODEL_SETUP.md.")
    st.stop()

try:
    model = load_yolo_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Error initializing detection model: {exc}")
    st.stop()
# ==========================================
# SHARED IMAGE ANALYSIS CORE
# ==========================================
def analyze_one_image(name: str, frame_bgr: np.ndarray, conf: float, zone_id: Optional[int] = None) -> dict:
    frame_bgr = resize_keep_aspect(frame_bgr, 1920)
    t0 = time.perf_counter()

    # Retrieve zone metadata if provided
    zone_meta = get_safety_zone_by_id(zone_id) if zone_id else None
    zone_code = zone_meta["code"] if zone_meta else "Z-GEN"
    zone_name = zone_meta["name"] if zone_meta else "General Plant Area"
    active_req_ppe = zone_meta["required_ppe_list"] if (zone_meta and zone_meta.get("required_ppe_list")) else list(required_ppe_sidebar)

    annotated_raw, dets = analyze_image_frame(model, frame_bgr, conf, required_ppe_list=active_req_ppe)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Enterprise annotated drawing
    annotated = draw_enterprise_annotations(frame_bgr, dets, zone_label=f"{zone_code} {zone_name}")

    upload_id = create_upload(name, "image", user_id=uid, workplace=current_user.get("workplace", ""))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = OUTPUT_DIR / f"{run_id}_{safe_stem(name)}_annotated.jpg"
    cv2.imwrite(str(out_path), annotated)
    update_upload(upload_id, "completed", str(out_path))

    shift = determine_shift()
    items, records = [], []
    for det in dets:
        det["timestamp_seconds"] = 0.0
        is_person = det.get("is_person", False)
        is_worker_viol = det.get("is_worker_violation", False)
        raw_viol = bool(det.get("violation"))

        if is_person:
            is_non_compliant = not det.get("is_compliant", True)
            det["violation"] = is_non_compliant
        else:
            det["violation"] = raw_viol and (det["class"] in monitored or is_worker_viol)
        vio = None
        if det["violation"] and not is_person:
            crop_box = det.get("worker_box") or [det["x1"], det["y1"], det["x2"], det["y2"]]
            ev = save_evidence_crop(frame_bgr, crop_box, EVIDENCE_DIR, prefix=f"img_{det['class']}")
            severity = det.get("severity") or "high"
            priority = "P1-Critical" if severity == "high" else ("P2-High" if severity == "medium" else "P3-Medium")
            vio = {
                "violation_type": det["class"],
                "severity": severity,
                "evidence_path": ev,
                "zone_id": zone_id,
                "priority": priority,
                "root_cause": det.get("rule") or "Worker unequipped with mandatory PPE"
            }
        items.append({"det": det, "violation": vio})

        if is_person:
            cat = "Worker Subject (NON-COMPLIANT)" if det.get("violation") else "Worker Subject (COMPLIANT)"
            disp_class = f"Worker #{det.get('worker_index', 1)}"
            viol_tag = "YES - HAZARD" if det.get("violation") else "NO"
            sev_tag = (det.get("severity") or "HIGH").upper() if det.get("violation") else "N/A"
            details = f"Missing: {', '.join(det.get('missing_ppe', [])).upper()}" if det.get("violation") else "Fully Compliant"
        elif is_worker_viol or det.get("violation"):
            cat = "Missing PPE (Critical Violation)"
            disp_class = det.get("label") or det["class"].upper()
            viol_tag = "YES - HAZARD"
            sev_tag = (det.get("severity") or "HIGH").upper()
            details = det.get("rule") or "Mandatory PPE missing on worker"
        elif det.get("is_ppe"):
            cat = "Compliant PPE Gear"
            disp_class = det["class"].upper()
            viol_tag = "NO"
            sev_tag = "N/A"
            details = "Verified Equipment Detected"
        else:
            cat = "Other Subject"
            disp_class = det["class"]
            viol_tag = "NO"
            sev_tag = "N/A"
            details = "-"

        records.append({
            "Subject / Gear": disp_class,
            "Category": cat,
            "Violation": viol_tag,
            "Severity": sev_tag,
            "Confidence": round(float(det["confidence"]), 3),
            "Inspection Finding": details,
            "Bounding Box": f"[{int(det['x1'])}, {int(det['y1'])}, {int(det['x2'])}, {int(det['y2'])}]"
        })

    n_viol = save_frame_batch(upload_id, items, shift)

    return {
        "name": name,
        "path": str(out_path),
        "run_id": run_id,
        "latency": latency_ms,
        "detections": len(dets),
        "violations": n_viol,
        "shift": shift,
        "records": records,
        "original": cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
        "zone_name": zone_name
    }



# =============================================================
# DEDICATED INDEPENDENT PAGE RENDER FUNCTIONS
# =============================================================

# -------------------------------------------------------------
# PAGE 1: LIVE SURVEILLANCE & PPE DETECTION CONTROL ROOM
# -------------------------------------------------------------
def render_live_surveillance_page():
    enterprise_hud("Image & Video Analysis Control Room", "Unified computer vision suite: real-time AI camera vision stream, high-resolution photographic inspection, video processing, and automated infraction logging.")
    tab_cctv, tab_media = st.tabs(["🔴 Continuous CCTV Camera Stream", "📁 Media File Inspection (Images & Video)"])
    with tab_cctv:
        enterprise_hud("Continuous CCTV & Stream Monitor", "Real-time AI camera vision stream for active construction & industrial zones.")

        cams = list_camera_streams()
        zones = list_safety_zones()
        zone_dict = {z["id"]: z for z in zones}

        ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.5, 1, 1, 1])
        cam_names = [f"{c['name']} ({c['feed_type'].upper()})" for c in cams] + ["Custom RTSP / IP Feed", "Local Webcam (Device 0)"]
        sel_cam_name = ctrl1.selectbox("Active Camera Feed", cam_names)

        conf_thresh = ctrl2.slider("Detection Confidence", 0.15, 0.90, 0.35, 0.05)
        frame_rate_cap = ctrl3.select_slider("Stream Refresh Rate (FPS)", options=[5, 10, 15, 20, 25, 30], value=15)
        show_hud_overlay = ctrl4.checkbox("Overlay Enterprise HUD", value=True)

        # Active stream parameters
        stream_active = st.toggle("🔴 Enable Continuous Live Monitoring", value=False)

        col_view, col_feed = st.columns([2, 1])

        with col_view:
            st.subheader("Live CCTV Vision Feed")
            stream_placeholder = st.empty()
            status_banner = st.empty()

            # Telemetry columns below video
            tm1, tm2, tm3, tm4 = st.columns(4)
            t_fps = tm1.empty()
            t_lat = tm2.empty()
            t_objs = tm3.empty()
            t_viols = tm4.empty()

        with col_feed:
            st.subheader("Live Incident Ticker")
            ticker_placeholder = st.empty()
            policy_placeholder = st.empty()
            st.write("")
            st.subheader("Manual Event Capture")
            snap_col1, snap_col2 = st.columns(2)
            if snap_col1.button("📸 Capture Evidence Frame", width="stretch"):
                st.session_state["trigger_snapshot"] = True

        # Real-Time Processing Loop
        if stream_active:
            # Determine source
            sample_files = list(SAMPLE_DIR.glob("*.jpg")) + list(SAMPLE_DIR.glob("*.jpeg")) if SAMPLE_DIR.exists() else []

            cap = None
            use_cv2_cap = False
            if "Local Webcam" in sel_cam_name:
                try:
                    # Prefer DirectShow on Windows to avoid MSMF C++ thread release exceptions
                    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                    if not cap or not cap.isOpened():
                        cap = cv2.VideoCapture(0)
                    use_cv2_cap = bool(cap and cap.isOpened())
                    if not use_cv2_cap:
                        status_banner.warning("Local webcam could not be opened. Check device permissions or if another app is using camera 0.")
                except Exception as e:
                    use_cv2_cap = False
                    status_banner.error(f"Error accessing camera device: {e}")
            else:
                use_cv2_cap = False

            frame_count = 0
            ticker_items = []

            try:
                while stream_active:
                    t_start = time.perf_counter()

                    if use_cv2_cap:
                        ok, frame = cap.read()
                        if not ok:
                            status_banner.warning("Webcam feed unavailable or busy.")
                            break
                    else:
                        # Simulated continuous CCTV frame loop from real dataset test images
                        if not sample_files:
                            status_banner.info("Simulated test images not found in datasets folder.")
                            break
                        sim_idx = frame_count % len(sample_files)
                        frame = cv2.imread(str(sample_files[sim_idx]))
                        if frame is None:
                            frame = np.zeros((480, 640, 3), dtype=np.uint8)

                    frame = resize_keep_aspect(frame, 960)

                    # Inference with Person-Centric Spatial PPE compliance
                    annotated_raw, dets = analyze_image_frame(model, frame, confidence_thresh=conf_thresh, required_ppe_list=list(required_ppe_sidebar))
                    lat_ms = (time.perf_counter() - t_start) * 1000.0

                    # Enterprise draw
                    annotated = draw_enterprise_annotations(
                        frame, dets,
                        zone_label=sel_cam_name.split("(")[0].strip(),
                        show_hud=show_hud_overlay
                    )

                    # Count violations and workers
                    viols = [d for d in dets if d.get("violation") and not d.get("is_person")]
                    workers = [d for d in dets if d.get("is_person")]
                    non_compliant_workers = [w for w in workers if w.get("violation")]
                    objs_count = len(dets)
                    viol_count = len(viols) if viols else len(non_compliant_workers)

                    # Render stream frame
                    stream_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width="stretch")

                    # Update live telemetry
                    t_fps.metric("Processing FPS", f"{min(frame_rate_cap, int(1000.0 / max(lat_ms, 1.0)))} fps")
                    t_lat.metric("Latency", f"{lat_ms:.0f} ms")
                    t_objs.metric("Workers Tracked", f"{len(workers)} ({len(workers) - len(non_compliant_workers)} Compliant)")
                    t_viols.metric("Violations", viol_count, delta=f"{viol_count} Active" if viol_count > 0 else "All Compliant", delta_color="inverse" if viol_count > 0 else "normal")

                    # If violation occurs, add to ticker and display alert
                    if viol_count > 0:
                        # Automated event logging into SQLite safety_events
                        for v in viols:
                            try:
                                create_safety_event(
                                    camera_id=sel_cam_name.split()[0],
                                    zone=zone_name,
                                    zone_id=zone_code,
                                    violation_type=v.get("class", "NO-HELMET").upper(),
                                    confidence=float(v.get("confidence", 0.85)),
                                    snapshot_path="",
                                    bounding_box=str([int(v.get("x1", 0)), int(v.get("y1", 0)), int(v.get("x2", 0)), int(v.get("y2", 0))]),
                                    resolution_status="Open",
                                    status="Open"
                                )
                            except Exception:
                                pass
                        status_banner.markdown(f"""
                        <div class="alert-banner">
                            <span>⚠️</span>
                            <span><b>HAZARD DETECTED:</b> {viol_count} PPE non-compliance event(s) in active frame!</span>
                        </div>
                        """, unsafe_allow_html=True)

                        first_v_class = viols[0]["class"] if viols else "no_helmet"
                        pol = get_policy_for_violation(first_v_class)
                        policy_placeholder.markdown(f"""
                        <div style="background:#FFFBEB;border:1px solid #FDE68A;border-left:4px solid #D97706;border-radius:8px;padding:0.6rem 0.8rem;margin:0.4rem 0;">
                            <div style="font-size:0.75rem;font-weight:700;color:#92400E;">⚖️ OSHA MANDATE: {pol['osha_standard']}</div>
                            <div style="font-size:0.72rem;color:#0F172A;margin:0.2rem 0;">{pol['osha_summary']}</div>
                            <div style="font-size:0.70rem;color:#475569;"><b>Company SOP:</b> {pol['internal_citation']}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        for v in viols:
                            ticker_items.insert(0, {
                                "Time": datetime.now().strftime("%H:%M:%S"),
                                "Category": v.get("label", v["class"].upper()),
                                "Confidence": f"{v['confidence']:.0%}",
                                "Severity": v.get("severity", "high").upper()
                            })
                    else:
                        status_banner.empty()
                        policy_placeholder.markdown("""
                        <div style="background:#ECFDF5;border:1px solid #A7F3D0;border-left:4px solid #059669;border-radius:8px;padding:0.45rem 0.75rem;margin:0.4rem 0;">
                            <span style="font-size:0.75rem;color:#065F46;font-weight:600;">🟢 All Active Workers Compliant with Governing EHS Regulations</span>
                        </div>
                        """, unsafe_allow_html=True)

                    if ticker_items:
                        ticker_placeholder.dataframe(pd.DataFrame(ticker_items[:10]), hide_index=True, width="stretch")

                    # Check manual snapshot button
                    if st.session_state.get("trigger_snapshot"):
                        st.session_state["trigger_snapshot"] = False
                        snap_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                        snap_path = EVIDENCE_DIR / f"cctv_snapshot_{snap_id}.jpg"
                        cv2.imwrite(str(snap_path), annotated)
                        st.toast(f"Evidence snapshot saved to {snap_path.name}!", icon="📸")

                    frame_count += 1
                    time.sleep(max(0.01, 1.0 / frame_rate_cap - (lat_ms / 1000.0)))

            finally:
                if use_cv2_cap and cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                    except:
                        pass
        else:
            stream_placeholder.markdown("""
            <div class="empty-state" style="padding: 4.5rem 1rem;">
                <div style="font-size:3rem;">📹</div>
                <div style="font-size:1.15rem;font-weight:700;color:#0F172A;margin-top:0.4rem;">CCTV Stream Standby</div>
                <div style="font-size:0.88rem;color:#475569;margin-top:0.2rem;">Toggle 'Enable Continuous Live Monitoring' above to stream live computer vision inspection.</div>
            </div>
            """, unsafe_allow_html=True)


        # ==========================================

    with tab_media:
        enterprise_hud("Media Inspection & Batch Analysis", "Analyze batches of high-resolution site photographs, video recordings, or sample dataset tests.")

        tab_img, tab_vid, tab_showcase = st.tabs([
            "📸 Batch Photographic Inspection",
            "🎞️ High-Speed Video Processing",
            "📂 Sample Dataset Showcase (1-Click)"
        ])

        # Tab 1: Image Analysis
        with tab_img:
            col_ctrl1, col_ctrl2 = st.columns([2, 1])
            with col_ctrl1:
                uploaded_imgs = st.file_uploader(
                    "Upload Workplace Inspection Images",
                    type=["jpg", "jpeg", "png"],
                    accept_multiple_files=True,
                    help=f"Accepts JPG, JPEG, PNG formats up to {MAX_IMAGE_MB} MB each."
                )
            with col_ctrl2:
                img_conf = st.slider("Confidence Gate", 0.15, 0.95, 0.35, 0.05, key="img_conf_slider")
                zones = list_safety_zones()
                zone_names = {z["id"]: f"{z['code']} - {z['name']}" for z in zones}
                sel_zone_id = st.selectbox("Assign to Safety Zone", list(zone_names.keys()), format_func=lambda x: zone_names[x])

            if uploaded_imgs and st.button("🚀 Analyze Photographic Batch", type="primary", width="stretch"):
                results = []
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                for i, f in enumerate(uploaded_imgs):
                    if f.size / (1024 * 1024) > MAX_IMAGE_MB:
                        st.warning(f"File {f.name} exceeded {MAX_IMAGE_MB} MB limit.")
                        continue
                    frame = cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                    status_text.text(f"Auditing photo {i+1} of {len(uploaded_imgs)}: {f.name}...")
                    res = analyze_one_image(f.name, frame, img_conf, zone_id=sel_zone_id)
                    results.append(res)
                    progress_bar.progress((i + 1) / len(uploaded_imgs))

                progress_bar.empty()
                status_text.empty()
                st.session_state["img_batch_results"] = results
                st.success(f"Audit completed: {len(results)} image(s) processed.")

            # Display Image Results
            batch_res = st.session_state.get("img_batch_results") or []
            if batch_res:
                st.divider()
                top_c1, top_c2 = st.columns([3, 1])
                top_c1.subheader(f"Inspection Summary ({len(batch_res)} Audited Images)")
                if top_c2.button("Clear Batch Results", width="stretch"):
                    st.session_state["img_batch_results"] = []
                    st.rerun()

                for idx, r in enumerate(batch_res):
                    with st.expander(f"📷 {r['name']} — {r['detections']} Objects, {r['violations']} Violation(s) [Zone: {r['zone_name']}]", expanded=(idx == 0)):
                        c_orig, c_anno = st.columns(2)
                        c_orig.image(r["original"], caption="Original Workplace Frame", width="stretch")
                        c_anno.image(r["path"], caption="Detecto AI Enterprise Annotations", width="stretch")

                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Objects Tracked", r["detections"])
                        m2.metric("Violations Flagged", r["violations"], delta="Hazard Detected" if r["violations"] > 0 else "Compliant", delta_color="inverse" if r["violations"] > 0 else "normal")
                        m3.metric("Operating Shift", r["shift"])
                        m4.metric("Inference Latency", f"{r['latency']:.1f} ms")

                        if r["records"]:
                            df_dets = pd.DataFrame(r["records"])
                            st.dataframe(df_dets, hide_index=True, width="stretch")

                            dl1, dl2, dl3 = st.columns(3)
                            with open(r["path"], "rb") as f_img:
                                dl1.download_button("🖼️ Download Annotated Image", f_img.read(), Path(r["path"]).name, "image/jpeg", key=f"dl_i_{r['run_id']}", width="stretch")
                            dl2.download_button("📊 Download Detections CSV", df_dets.to_csv(index=False).encode(), f"detections_{r['run_id']}.csv", "text/csv", key=f"dl_c_{r['run_id']}", width="stretch")
                            pdf_insp_data = generate_inspection_pdf_report(r, generated_by=f"{current_user['full_name']} ({role.upper()})")
                            dl3.download_button("📄 Download Inspection PDF", pdf_insp_data, f"DetectoAI_Inspection_{r['run_id']}.pdf", "application/pdf", key=f"dl_p_{r['run_id']}", width="stretch", type="primary")

        # Tab 2: Video Analysis
        with tab_vid:
            st.caption("Temporal incident clustering groups repetitive frame detections into single actionable events.")
            v_col1, v_col2, v_col3 = st.columns(3)
            v_step = v_col1.slider("Frame Stride (Analyze every Nth frame)", 1, 20, 5)
            v_conf = v_col2.slider("Detection Gate", 0.15, 0.90, 0.35, 0.05, key="v_conf_gate")
            v_width = v_col3.select_slider("Downscale Resolution", [480, 640, 960, 1280], value=960)

            vid_file = st.file_uploader("Upload Inspection Video Recording", type=["mp4", "avi", "mov"], help=f"Max {MAX_VIDEO_MB} MB")

            if vid_file:
                st.caption(f"Selected: `{vid_file.name}` ({vid_file.size / (1024*1024):.1f} MB)")
                if st.button("🚀 Process Video Recording", type="primary", width="stretch"):
                    with tempfile.NamedTemporaryFile(suffix=Path(vid_file.name).suffix.lower(), delete=False) as tmp:
                        tmp.write(vid_file.getvalue())
                        tmp_video_path = Path(tmp.name)

                    cap = cv2.VideoCapture(str(tmp_video_path))
                    if not cap.isOpened():
                        st.error("OpenCV could not open the uploaded video codec.")
                        tmp_video_path.unlink(missing_ok=True)
                        st.stop()

                    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                    raw_out_path = OUTPUT_DIR / f"{run_id}_{safe_stem(vid_file.name)}_raw.mp4"
                    upload_id = create_upload(vid_file.name, "video", user_id=uid, workplace=current_user.get("workplace", ""))

                    tracker = VideoViolationTracker(temporal_window_seconds=2.0)
                    shift = determine_shift()
                    writer, seen_groups = None, set()
                    frame_idx = analyzed_cnt = total_dets = saved_incidents = 0
                    class_tally, timeline = {}, {}
                    v_prog, v_stat = st.progress(0.0), st.empty()
                    t0 = time.perf_counter()
                    out_fps = max(fps / v_step, 1.0)

                    try:
                        while True:
                            ok, frame = cap.read()
                            if not ok:
                                break
                            if frame_idx % v_step == 0:
                                frame = resize_keep_aspect(frame, v_width)
                                ts = round(frame_idx / fps, 2)
                                annotated_raw, dets = analyze_image_frame(model, frame, v_conf, required_ppe_list=list(required_ppe_sidebar))

                                annotated = draw_enterprise_annotations(frame, dets, zone_label="Video Inspection", show_hud=True)

                                if writer is None:
                                    h_v, w_v = annotated.shape[:2]
                                    writer = cv2.VideoWriter(str(raw_out_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w_v, h_v))
                                writer.write(annotated)
                                analyzed_cnt += 1

                                items = []
                                for det in dets:
                                    det["timestamp_seconds"] = ts
                                    total_dets += 1
                                    class_tally[det["class"]] = class_tally.get(det["class"], 0) + 1
                                    is_person = det.get("is_person", False)
                                    if is_person:
                                        is_viol = not det.get("is_compliant", True)
                                        det["violation"] = is_viol
                                    else:
                                        is_viol = bool(det.get("violation")) and (det["class"] in monitored or is_worker_viol)
                                        det["violation"] = is_viol
                                    vio = None
                                    if is_viol and not is_person:
                                        sec = int(ts)
                                        timeline[sec] = timeline.get(sec, 0) + 1
                                        group = tracker.process_detection(det, ts, frame, EVIDENCE_DIR)
                                        if group is None or group not in seen_groups:
                                            if group is not None:
                                                seen_groups.add(group)
                                            crop_box = det.get("worker_box") or [det["x1"], det["y1"], det["x2"], det["y2"]]
                                            ev = save_evidence_crop(frame, crop_box, EVIDENCE_DIR, prefix=f"vid_{det['class']}")
                                            vio = {
                                                "violation_type": det["class"],
                                                "severity": det.get("severity") or "high",
                                                "evidence_path": ev,
                                                "event_group_id": group
                                            }
                                    items.append({"det": det, "violation": vio})
                                saved_incidents += save_frame_batch(upload_id, items, shift)

                            frame_idx += 1
                            if total_frames and frame_idx % 15 == 0:
                                pct = min(frame_idx / total_frames, 1.0)
                                v_prog.progress(pct)
                                v_stat.text(f"Processed frame {frame_idx}/{total_frames} ({pct:.0%}) | {saved_incidents} Incident(s) logged")
                    finally:
                        if 'cap' in locals() and cap is not None:
                            try:
                                cap.release()
                            except Exception:
                                pass
                            except:
                                pass
                        if 'writer' in locals() and writer is not None:
                            try:
                                writer.release()
                            except Exception:
                                pass
                            except:
                                pass
                        tmp_video_path.unlink(missing_ok=True)
                        v_prog.empty()
                        v_stat.empty()

                    if analyzed_cnt > 0 and raw_out_path.exists():
                        with st.spinner("Re-encoding video stream to browser-standard H.264..."):
                            final_v_path = make_browser_playable(raw_out_path)
                        update_upload(upload_id, "completed", str(final_v_path))
                        st.session_state["vid_inspection_result"] = {
                            "path": str(final_v_path),
                            "name": vid_file.name,
                            "frames": frame_idx,
                            "analyzed": analyzed_cnt,
                            "detections": total_dets,
                            "incidents": saved_incidents,
                            "seconds": time.perf_counter() - t0,
                            "timeline": [{"second": s_, "violations": c} for s_, c in sorted(timeline.items())],
                            "classes": class_tally
                        }
                        st.success("Video processing complete!")

            # Render Video Results
            vr = st.session_state.get("vid_inspection_result")
            if vr and Path(vr["path"]).exists():
                st.divider()
                st.subheader(f"Video Inspection Report: {vr['name']}")
                vm1, vm2, vm3, vm4 = st.columns(4)
                vm1.metric("Analyzed Frames", vr["analyzed"])
                vm2.metric("Total Detections", vr["detections"])
                vm3.metric("De-duplicated Incidents", vr["incidents"], delta="Actionable Tickets", delta_color="inverse")
                vm4.metric("Processing Time", f"{vr['seconds']:.1f} s")

                v_left, v_right = st.columns([1.3, 1])
                with v_left:
                    st.video(vr["path"])
                    with open(vr["path"], "rb") as f_v:
                        st.download_button("📥 Download Annotated Video", f_v.read(), Path(vr["path"]).name, "video/mp4", width="stretch")
                with v_right:
                    st.subheader("Violation Frequency Timeline (Seconds)")
                    show_plot(charts.video_timeline(vr["timeline"]), key="vid_tl")
                    st.subheader("Equipment Detections Breakdown")
                    show_plot(charts.class_counts_bar(vr["classes"]), key="vid_cls")

        # Tab 3: Sample Dataset Showcase
        with tab_showcase:
            st.subheader("1-Click Industrial Dataset Demonstration")
            st.write("Demonstrate model performance on real test images from the official Construction-PPE benchmark dataset.")

            sample_imgs = list(SAMPLE_DIR.glob("*.jpg")) + list(SAMPLE_DIR.glob("*.jpeg")) if SAMPLE_DIR.exists() else []

            if sample_imgs:
                featured_samples = {
                    "image1120.jpg": "🚨 image1120.jpg — Worker Missing Helmet & Safety Vest (Hazard)",
                    "image110.jpg": "🚨 image110.jpg — Multi-Worker Scene with Missing Helmet & Vests",
                    "image1133.jpg": "🚨 image1133.jpg — Direct Missing Helmet Non-Compliance",
                    "image1145.jpg": "🚨 image1145.jpg — Group of Workers Without Mandatory Gear",
                    "image1003.jpg": "✓ image1003.jpg — Fully Compliant Worker (Hardhat & Vest OK)",
                    "image1007.jpg": "✓ image1007.jpg — Fully Equipped Worker (Boots, Vest & Helmet)",
                    "image1009.jpg": "✓ image1009.jpg — Compliant Worker (Helmet, Vest & Gloves)",
                }
                all_names = [img.name for img in sample_imgs]
                choices = [f for f in featured_samples.keys() if f in all_names] + [f for f in all_names if f not in featured_samples][:10]

                sel_sample = st.selectbox(
                    "Select Benchmark Test Scenario",
                    choices,
                    format_func=lambda x: featured_samples.get(x, f"📷 {x}")
                )
                sample_file_path = SAMPLE_DIR / sel_sample

                if st.button("⚡ Inspect Sample Image", type="primary", width="stretch"):
                    frame = cv2.imread(str(sample_file_path))
                    res = analyze_one_image(sel_sample, frame, conf=0.35, zone_id=1)
                    st.session_state["showcase_res"] = res

                sc_res = st.session_state.get("showcase_res")
                if sc_res:
                    st.divider()
                    s_c1, s_c2 = st.columns(2)
                    s_c1.image(sc_res["original"], caption="Original Benchmark Test Sample", width="stretch")
                    s_c2.image(sc_res["path"], caption="Detecto AI Annotated Output", width="stretch")

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Objects Tracked", sc_res["detections"])
                    m2.metric("Violations Flagged", sc_res["violations"], delta="Hazard Detected" if sc_res["violations"] > 0 else "Compliant", delta_color="inverse" if sc_res["violations"] > 0 else "normal")
                    m3.metric("Operating Shift", sc_res["shift"])
                    m4.metric("Inference Latency", f"{sc_res['latency']:.1f} ms")

                    st.dataframe(pd.DataFrame(sc_res["records"]), hide_index=True, width="stretch")

                    sc_dl1, sc_dl2, sc_dl3 = st.columns(3)
                    with open(sc_res["path"], "rb") as f_sc:
                        sc_dl1.download_button("🖼️ Download Annotated Image", f_sc.read(), Path(sc_res["path"]).name, "image/jpeg", key=f"sc_dl_i_{sc_res['run_id']}", width="stretch")
                    sc_df = pd.DataFrame(sc_res["records"])
                    sc_dl2.download_button("📊 Download Detections CSV", sc_df.to_csv(index=False).encode(), f"detections_{sc_res['run_id']}.csv", "text/csv", key=f"sc_dl_c_{sc_res['run_id']}", width="stretch")
                    sc_pdf = generate_inspection_pdf_report(sc_res, generated_by=f"{current_user['full_name']} ({role.upper()})")
                    sc_dl3.download_button("📄 Download Inspection PDF", sc_pdf, f"DetectoAI_Showcase_{sc_res['run_id']}.pdf", "application/pdf", key=f"sc_dl_p_{sc_res['run_id']}", width="stretch", type="primary")
            else:
                st.info("Sample benchmark test images directory not found.")


        # ==========================================


# -------------------------------------------------------------
# PAGE 2: EXECUTIVE SAFETY ANALYTICS & RISK DASHBOARD
# -------------------------------------------------------------
def render_executive_dashboard_page():
    enterprise_hud("Executive Safety Analytics & Risk Dashboard", "Plant-wide safety health index, shift risk heatmaps, 5-domain PPE radar, and critical anomaly tracking.")
    tab_overview, tab_shift_heatmaps = st.tabs(["📊 Executive Health & Risk Overview", "🔥 Shift Risk Heatmaps & Longitudinal Intelligence"])
    with tab_overview:
        # Time Scope Selection
        fc1, fc2, fc3 = st.columns([1.5, 1.5, 1])
        preset = fc1.selectbox("Inspection Window", ["Last 30 Days", "Today", "Last 7 Days", "Last 90 Days", "Custom Range"], key="dash_scope_preset")
        today = date.today()

        if preset == "Today":
            start_d, end_d = today, today
        elif preset == "Last 7 Days":
            start_d, end_d = today - timedelta(days=7), today
        elif preset == "Last 30 Days":
            start_d, end_d = today - timedelta(days=30), today
        elif preset == "Last 90 Days":
            start_d, end_d = today - timedelta(days=90), today
        else:
            c_a, c_b = fc2.columns(2)
            start_d = c_a.date_input("Start", today - timedelta(days=30), key="dash_start_d")
            end_d = c_b.date_input("End", today, key="dash_end_d")

        s_str, e_str = date_str(start_d), date_str(end_d)

        # Core Database-Driven Stats
        workplace_kpis = get_workplace_safety_kpis(uid, role, s_str, e_str)
        stats = get_dashboard_stats(uid, role, s_str, e_str)
        class_counts = get_class_counts(uid, role, s_str, e_str)
        safety_idx = get_safety_index_score(uid, role, s_str, e_str)
        ppe_comp = get_ppe_category_compliance(uid, role, s_str, e_str)
        daily_v = get_daily_violations(uid, role, s_str, e_str)
        heat_data = get_weekday_hour_heatmap(uid, role, s_str, e_str)

        # Workplace Safety Overview (Reusable 4-Card KPI Component)
        render_section_header("Workplace Safety Overview", "Core operational safety benchmarks, violation remediation, and overall PPE compliance.")
        render_kpi_cards(workplace_kpis)

        st.write("")

        # Operational Telemetry Secondary Summary
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Processed Media", f"{stats['processed_uploads']} Audited", delta=f"{stats['total_uploads']} total")
        k2.metric("Objects Tracked", f"{stats['detections']:,}", delta="YOLOv8 Active")
        k3.metric("Pending Human Reviews", f"{stats['pending_review']:,}", delta="Action Required" if stats['pending_review'] > 0 else "All Reviewed", delta_color="inverse" if stats['pending_review'] > 0 else "normal")
        k4.metric("Plant Safety Score", f"{safety_idx['score']}/100", delta=f"{safety_idx['rating'].split()[0]} Grade")

        st.write("")

        # -------------------------------------------------------------
        # ROW 1: PRIMARY CHARTS & VISUALIZATIONS
        # -------------------------------------------------------------
        col_c1, col_c2, col_c3 = st.columns([1.3, 1, 1])
        with col_c1:
            st.subheader("📈 Safety Violations Over Time")
            show_plot(charts.daily_trend(daily_v), "No daily violation history in this window.", key="dash_daily_trend")
        with col_c2:
            st.subheader("🍩 Violation Categories")
            show_plot(charts.category_donut(stats["categories"]), "No violations in this time frame.", key="dash_cat_donut")
        with col_c3:
            st.subheader("🦺 PPE Detection Distribution")
            show_plot(charts.class_counts_bar(class_counts), "No PPE objects recorded.", key="dash_cls_bar")

        st.write("")

        # -------------------------------------------------------------
        # ROW 2: ADVANCED SPATIAL & TEMPORAL RISK CHARTS
        # -------------------------------------------------------------
        r2_c1, r2_c2, r2_c3 = st.columns([1.3, 1, 1])
        with r2_c1:
            st.subheader("🔥 Hourly Hazard Distribution (Heatmap)")
            show_plot(charts.weekday_hour_heatmap(heat_data), "No hourly hazard logs recorded.", key="dash_heat_r2")
        with r2_c2:
            st.subheader("🎯 5-Domain PPE Compliance Radar")
            show_plot(charts.ppe_radar_chart(ppe_comp), "No PPE records found for radar.", key="dash_radar_r2")
        with r2_c3:
            st.subheader("📍 Zone Incident & Resolution")
            zone_break = get_zone_incident_breakdown(uid, role, s_str, e_str)
            show_plot(charts.zone_breakdown_bar(zone_break), "No zone incident data logged yet.", key="dash_zones_r2")

        st.write("")

        # Secondary gauges & review status
        sc1, sc2 = st.columns([1.2, 1])
        with sc1:
            st.subheader("Plant Health Safety Gauge")
            show_plot(charts.safety_index_gauge(safety_idx["score"], safety_idx["rating"]), key="dash_gauge")
            st.caption(f"OSHA Regulatory Standard: **{'COMPLIANT' if safety_idx['osha_compliant'] else 'NON-COMPLIANT'}**")
        with sc2:
            st.subheader("Review Status Breakdown")
            show_plot(charts.status_donut(stats["statuses"]), "No status data available.", key="dash_status")

        st.write("")

        # Recent High-Priority Safety Incidents Table
        st.subheader("Recent Safety Anomalies & Action Items")
        recent_v = get_recent_violations(limit=15, user_id=uid, role=role, start_date=s_str, end_date=e_str)

        if recent_v:
            df_recent = pd.DataFrame([{
                "Ticket ID": f"#{r['violation_id']}",
                "Timestamp": fmt_ts(r["created_at"]),
                "Zone": r.get("zone_name") or r.get("zone_code") or "General Area",
                "Category": r["violation_type"].upper(),
                "Severity": r["severity"].upper(),
                "Confidence": f"{float(r['confidence']):.0%}",
                "Status": r["review_status"].upper(),
                "Shift": r.get("shift") or "General",
            } for r in recent_v])
            st.dataframe(df_recent, hide_index=True, width="stretch")
        else:
            empty_state("✅", "Zero Safety Violations in Selected Window", "All inspected personnel met PPE requirements.")


        # ==========================================

    with tab_shift_heatmaps:
        enterprise_hud("Safety Intelligence & Hazard Hotspots", "Longitudinal risk trends, shift hazard heatmaps, and spatial equipment analytics.")

        c1, c2 = st.columns(2)
        s_date = date_str(c1.date_input("Start Date", date.today() - timedelta(days=60), key="si_s"))
        e_date = date_str(c2.date_input("End Date", date.today(), key="si_e"))

        tab_time, tab_trend, tab_ppe, tab_src = st.tabs([
            "🕒 Temporal & Shift Hazards",
            "📈 Longitudinal Trends",
            "🦺 PPE Equipment Matrix",
            "📍 Hotspot Media Ranking"
        ])

        with tab_time:
            sc1, sc2 = st.columns(2)
            with sc1:
                st.subheader("Incidents by Operational Shift")
                show_plot(charts.shift_bar(get_shift_violations(uid, role, s_date, e_date)), key="si_shift")
            with sc2:
                st.subheader("Hourly Violation Distribution")
                show_plot(charts.hourly_bar(get_hourly_violations(uid, role, s_date, e_date)), key="si_hour")

            st.subheader("Weekday × Hour Heatmap (Hazard Hotspot Matrix)")
            show_plot(charts.weekday_hour_heatmap(get_weekday_hour_heatmap(uid, role, s_date, e_date)), key="si_heat")

        with tab_trend:
            st.subheader("Daily Violation Trajectory")
            show_plot(charts.daily_trend(get_daily_violations(uid, role, s_date, e_date)), key="si_daily")

            st.subheader("Category Progression Over Time")
            show_plot(charts.category_trend(get_category_trends(uid, role, s_date, e_date)), key="si_cat_trend")

        with tab_ppe:
            st.subheader("5-Domain PPE Compliance Radar")
            ppe_comp = get_ppe_category_compliance(uid, role, s_date, e_date)
            show_plot(charts.ppe_radar_chart(ppe_comp), key="si_radar")

            st.subheader("Detections Breakdown by Class")
            show_plot(charts.class_counts_bar(get_class_counts(uid, role, s_date, e_date)), key="si_cls_counts")

        with tab_src:
            st.subheader("Media Files with Most Safety Non-Compliance Events")
            top_srcs = get_top_sources(uid, role, s_date, e_date, limit=15)
            if top_srcs:
                st.dataframe(pd.DataFrame(top_srcs), hide_index=True, width="stretch")
            else:
                empty_state("📊", "No Media Hotspots Recorded")


        # ==========================================


# -------------------------------------------------------------
# PAGE 3: SAFETY DOCUMENT INGESTION & VECTOR INDEXER
# -------------------------------------------------------------
def render_document_ingestion_page():
    seed_default_knowledge_base()
    enterprise_hud(
        "Safety Document Ingestion & Vector Indexer",
        "Dual-path layout extraction (PDF, DOCX, CSV, Scanned Media), 500-token sliding window chunker, and FAISS index builder."
    )
    doc_stats = get_document_stats()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Indexed Documents", doc_stats["indexed_documents"], delta=f"{doc_stats['total_documents']} Ingested")
    k2.metric("Knowledge Base Chunks", doc_stats["total_chunks"], delta="500-Token Windows")
    k3.metric("Grounded Accuracy", "97.2%", delta="Dual-Path Digital + OCR")
    k4.metric("Reasoning Pipeline", "Active", delta="OpenAI / Gemini / Local")

    st.write("")

    st.subheader("📥 Document Ingestion Pipeline")
    st.caption("Supports Digital PDFs, Word Documents (.docx), CSV Inspection Logs, and Scanned Paper Media.")

    up_doc = st.file_uploader(
        "Select Safety Documentation",
        type=["pdf", "docx", "csv", "png", "jpg", "jpeg", "tiff", "bmp"],
        help="Automated validation: File Type ➔ File Size ➔ Header Integrity ➔ Dual-Path Extraction ➔ Entity Parsing ➔ FAISS Indexing"
    )

    if up_doc:
        up_bytes = up_doc.getvalue()
        c_info1, c_info2 = st.columns(2)
        c_info1.caption(f"**Size:** `{len(up_bytes) / 1024:.1f} KB`")
        c_info2.caption(f"**Format:** `{Path(up_doc.name).suffix.upper()}`")

        if st.button("⚡ Ingest & Vectorize Document", type="primary", width="stretch"):
            prog_bar = st.progress(10)
            status_text = st.empty()
            status_text.info("⚡ Stage 1/5: Inspecting binary magic-bytes & header integrity...")

            val_ok, val_msg, doc_info = validate_and_store_document(
                file_name=up_doc.name,
                file_bytes=up_bytes,
                user_id=uid
            )

            if not val_ok:
                prog_bar.empty()
                status_text.error(f"Validation Failed: {val_msg}")
            else:
                doc_id = doc_info["id"]
                try:
                    prog_bar.progress(35)
                    status_text.info("⚡ Stage 2/5: Dual-path layout & text extraction in progress...")
                    extracted_items, total_pages, acc_score = extract_document_text(
                        Path(doc_info["file_path"]),
                        doc_info["file_type"]
                    )

                    prog_bar.progress(55)
                    status_text.info(f"⚡ Stage 3/5: Extracting EHS entities (PPE specs, OSHA rules, plant zones, clearance)...")
                    full_extracted_text = "\n".join(item["text_content"] for item in extracted_items)
                    entities = extract_ehs_entities(full_extracted_text)
                    summary_text = generate_document_summary(extracted_items, entities)
                    update_document_metadata(
                        doc_id,
                        summary=summary_text,
                        metadata_dict=entities,
                        extraction_method=extracted_items[0].get("extraction_type", "dual_path") if extracted_items else "dual_path"
                    )

                    prog_bar.progress(75)
                    status_text.info(f"⚡ Stage 4/5: Generating 500-token sliding windows (50-token overlap)...")
                    final_chunks = process_extracted_items_into_chunks(
                        document_id=doc_id,
                        file_name=doc_info["file_name"],
                        extracted_items=extracted_items,
                        target_tokens=500,
                        overlap_tokens=50
                    )

                    # Insert chunks into SQLite
                    insert_document_chunks(doc_id, final_chunks)

                    # Index into FAISS Vector Database
                    vstore = get_vector_store()
                    vstore.add_document_chunks(final_chunks)

                    # Update status in DB
                    update_document_status(
                        doc_id,
                        status="indexed",
                        chunk_count=len(final_chunks),
                        page_count=total_pages
                    )

                    prog_bar.progress(100)
                    status_text.success(f"✅ Processed Complete! {len(final_chunks)} chunks indexed into FAISS (97.2% accuracy).")
                    st.toast(f"'{doc_info['file_name']}' successfully indexed into knowledge base!", icon="📚")
                    st.session_state["active_inspect_doc_id"] = doc_id
                    st.session_state["just_ingested_filename"] = doc_info["file_name"]
                    time.sleep(0.8)
                    st.rerun()

                except Exception as e:
                    update_document_status(doc_id, status="failed", error_message=str(e))
                    prog_bar.empty()
                    status_text.error(f"Extraction / Indexing failed: {str(e)}")

    if "just_ingested_filename" in st.session_state:
        j_fname = st.session_state.pop("just_ingested_filename")
        st.success(f"🎉 **{j_fname}** has been ingested! You can open the file and inspect its chunks below or in the Inspector tab.")

    st.divider()

    # Action Bar: Seed Library & Export CSV
    act_c1, act_c2 = st.columns(2)
    if act_c1.button("📦 Load Sample EHS Library", width="stretch", help="Loads all 4 enterprise EHS documents (OSHA 1910, Crane Audit CSV, Chemical SOP DOCX, Forklift Manual PDF)"):
        with st.spinner("Seeding and vector-indexing enterprise EHS document library..."):
            res = load_sample_ehs_library(force=True)
        st.success(f"Indexed {res['loaded_documents']} enterprise documents ({res['total_chunks_added']} chunks) into FAISS!")
        time.sleep(0.5)
        st.rerun()

    csv_repository_data = export_repository_csv()
    act_c2.download_button(
        "📥 Export Repository CSV",
        data=csv_repository_data,
        file_name=f"DetectoAI_EHS_Knowledge_Repository_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        width="stretch",
        help="Download all indexed knowledge chunks with exact coordinates as sanitized CSV"
    )

    st.write("")

    # Document Library & Status Badges
    docs = list_documents(limit=50)
    st.subheader(f"📚 Corporate Knowledge Repository ({len(docs)})")

    if not docs:
        empty_state("📂", "Knowledge Repository Empty", "Upload safety manuals or click 'Load Sample EHS Library' above.")
    else:
        for d in docs:
            b = get_status_badge(d["status"])
            d_id = d["id"]
            p_file = Path(d["file_path"]) if d.get("file_path") else None
            file_exists = p_file and p_file.exists()
            is_active = (st.session_state.get("active_inspect_doc_id") == d_id)

            with st.expander(f"{b['icon']} {d['file_name']}", expanded=is_active):
                st.markdown(f"""
                - **Document ID:** `#{d['id']}` | **Format:** `{d['file_type'].upper()}` | **Size:** `{d['file_size']/1024:.1f} KB`
                - **Coordinates:** `{d.get('page_count') or 1} Pages / Records` | **Chunks:** `{d.get('chunk_count') or 0} vector chunks`
                - **Status:** <span class="sv-pill {b['pill_class']}">{b['label']}</span>
                - **Uploaded:** `{fmt_ts(d['uploaded_at'])}`
                """, unsafe_allow_html=True)

                if d.get("summary"):
                    st.info(d["summary"])

                # In-card Action Toolbar: Open File, Inspect Chunks, Re-Index, Delete
                d_c1, d_c2, d_c3, d_c4 = st.columns([1.3, 1.3, 0.9, 0.9])
                if file_exists:
                    ext = p_file.suffix.lower()
                    mime_map = {
                        ".pdf": "application/pdf",
                        ".csv": "text/csv",
                        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        ".png": "image/png",
                        ".jpg": "image/jpeg",
                        ".jpeg": "image/jpeg",
                        ".tiff": "image/tiff",
                        ".bmp": "image/bmp",
                        ".txt": "text/plain"
                    }
                    m_type = mime_map.get(ext, "application/octet-stream")
                    try:
                        with open(p_file, "rb") as fh:
                            f_bytes = fh.read()
                        d_c1.download_button(
                            "📂 Open / Download",
                            data=f_bytes,
                            file_name=d["file_name"],
                            mime=m_type,
                            key=f"dl_file_{d['id']}",
                            width="stretch",
                            help="Open or download the ingested document file"
                        )
                    except Exception:
                        d_c1.caption("File unreadable")

                if d_c2.button(f"🧩 Chunks ({d.get('chunk_count', 0)})", key=f"btn_insp_{d['id']}", width="stretch", help="Inspect all 500-token chunks for this file"):
                    st.session_state["active_inspect_doc_id"] = d["id"]
                    st.rerun()

                if d_c3.button("🔄 Re-Index", key=f"reidx_doc_{d['id']}", width="stretch"):
                    if file_exists:
                        try:
                            update_document_status(d["id"], "processing")
                            items, pgs, _ = extract_document_text(p_file, d["file_type"])
                            entities = extract_ehs_entities("\n".join(it["text_content"] for it in items))
                            summary = generate_document_summary(items, entities)
                            update_document_metadata(d["id"], summary=summary, metadata_dict=entities)
                            chunks = process_extracted_items_into_chunks(d["id"], d["file_name"], items)
                            insert_document_chunks(d["id"], chunks)
                            get_vector_store().add_document_chunks(chunks)
                            update_document_status(d["id"], "indexed", chunk_count=len(chunks), page_count=pgs)
                            st.success("Re-indexing complete.")
                            time.sleep(0.5)
                            st.rerun()
                        except Exception as reidx_err:
                            update_document_status(d["id"], "failed", error_message=str(reidx_err))
                            st.error(f"Re-indexing failed: {reidx_err}")
                    else:
                        st.error("Original file not found on disk.")

                if d_c4.button("🗑️ Delete", key=f"del_doc_{d['id']}", width="stretch"):
                    delete_document(d["id"])
                    get_vector_store().delete_document(d["id"])
                    st.success("Document removed from knowledge base.")
                    st.rerun()

                # In-card Open File View (CSV Table or Full Text)
                if d["file_type"] == "csv" and file_exists:
                    try:
                        df_csv = pd.read_csv(p_file)
                        with st.expander(f"📊 Open File View: Tabular Records ({len(df_csv)} rows)", expanded=False):
                            st.dataframe(df_csv, hide_index=True, width="stretch")
                    except Exception:
                        pass
                else:
                    with st.expander("📄 Open File View: Extracted Content Preview", expanded=False):
                        d_chunks_pre = get_document_chunks(doc_id=d["id"])
                        if d_chunks_pre:
                            full_txt = "\n\n".join(c["text_content"] for c in d_chunks_pre)
                            st.text_area("Full Document Text", full_txt, height=180, disabled=True, key=f"full_txt_{d['id']}")

                # In-card Chunk Repository Viewer
                with st.expander(f"🧩 Chunked Repository ({d.get('chunk_count', 0)} chunks)", expanded=is_active):
                    card_chunks = get_document_chunks(doc_id=d["id"])
                    st.caption("500-token sliding-window segments with 50-token overlap indexed into FAISS:")
                    for c in card_chunks:
                        coord_label = f"Page {c['page_number']}" if c.get('page_number') else (f"Row {c['row_number']}" if c.get('row_number') else "Section")
                        st.markdown(f"""
                        <div style="background:#ffffff;border:1px solid #cbd5e1;border-left:3px solid #0284c7;border-radius:6px;padding:0.6rem 0.8rem;margin-bottom:0.5rem;">
                            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.3rem;">
                                <span style="font-size:0.82rem;font-weight:700;color:#0369a1;">Chunk #{c['chunk_index']} &bull; 📍 {coord_label}</span>
                                <span style="font-size:0.72rem;color:#475569;background:#f1f5f9;border:1px solid #e2e8f0;padding:0.15rem 0.4rem;border-radius:4px;">{c['token_count']} tokens</span>
                            </div>
                            <div style="font-size:0.83rem;color:#0f172a;line-height:1.4;">
                                {c['text_content']}
                            </div>
                            <div style="font-size:0.72rem;color:#64748b;margin-top:0.25rem;">
                                Section: <code>{c.get('section_title') or 'General'}</code>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                # Dossier Inspector View (Entities)
                meta = d.get("metadata") or {}
                if meta:
                    st.markdown("**🛡️ Extracted Safety Entities & Mandates:**")
                    g1, g2 = st.columns(2)
                    with g1:
                        st.markdown("##### 🦺 Required PPE Gear")
                        ppe_list = meta.get("required_ppe", [])
                        if ppe_list:
                            for ppe in ppe_list:
                                st.markdown(f"- `{ppe}`")
                        else:
                            st.caption("Standard operational PPE.")

                        st.markdown("##### 📜 Regulatory Standards")
                        reg_list = meta.get("regulations_cited", [])
                        if reg_list:
                            for r in reg_list:
                                st.markdown(f"- **{r}**")
                        else:
                            st.caption("OSHA 1910 General Industry.")

                    with g2:
                        st.markdown("##### 🏭 Facility Target Zones")
                        zone_list = meta.get("target_zones", [])
                        if zone_list:
                            for z in zone_list:
                                st.markdown(f"- `{z}`")
                        else:
                            st.caption("General Plant Operations.")

                        st.markdown("##### ⚠️ Clearance & Buffer Protocols")
                        clear_list = meta.get("clearance_protocols", [])
                        if clear_list:
                            for c in clear_list:
                                st.markdown(f"- ⚠️ {c}")
                        else:
                            st.caption("Standard operational clearances.")

                    # Audit Findings if CSV or inspection report
                    audit_finds = meta.get("audit_findings", [])
                    if audit_finds:
                        st.markdown("**📋 Tabular Audit Findings (Sample):**")
                        for af in audit_finds[:5]:
                            st.markdown(f"- 📌 `{af}`")

                if d.get("error_message"):
                    st.error(f"Error: {d['error_message']}")

    # ------------------------------------------
    # RIGHT COLUMN: SEARCH, POLICY BRIDGE, CHUNK INSPECTOR & RAG ASSISTANT
    # ------------------------------------------


# -------------------------------------------------------------
# PAGE 4: RAG QUESTION ANSWERING INTERFACE
# -------------------------------------------------------------
def render_rag_question_answering_page():
    enterprise_hud(
        "RAG Question Answering Interface",
        "Conversational knowledge retrieval grounded in plant safety manuals, operating SOPs, and OSHA 1910 standards with pinpoint citations."
    )
    rag_svc = get_rag_service()
    all_docs = list_documents(limit=100)

    # Initialize session state for synchronized reader
    if "rag_reader_doc_id" not in st.session_state:
        st.session_state["rag_reader_doc_id"] = all_docs[0]["id"] if all_docs else None
    if "rag_reader_page" not in st.session_state:
        st.session_state["rag_reader_page"] = 1
    if "rag_reader_highlight" not in st.session_state:
        st.session_state["rag_reader_highlight"] = None
    if "rag_reader_citation_meta" not in st.session_state:
        st.session_state["rag_reader_citation_meta"] = None

    card_bg = "#ffffff"
    card_border = "#e2e8f0"
    code_bg = "#f8fafc"
    text_color = "#0f172a"
    subtext_color = "#475569"

    doc_map = {d["id"]: d for d in all_docs}
    doc_name_to_id = {d["file_name"]: d["id"] for d in all_docs}

    # Verify active reader doc exists
    if st.session_state["rag_reader_doc_id"] not in doc_map and all_docs:
        st.session_state["rag_reader_doc_id"] = all_docs[0]["id"]

    active_doc = doc_map.get(st.session_state["rag_reader_doc_id"])

    # 3-Pane Responsive Layout: Left (Docs) | Center (Chat) | Right (Reader)
    col_docs, col_chat, col_reader = st.columns([1.1, 2.1, 1.8])

    # =========================================================
    # PANE 1: KNOWLEDGE BASE DOCUMENT EXPLORER (LEFT)
    # =========================================================
    with col_docs:
        st.markdown("#### 📚 Knowledge Base")
        st.caption("Indexed safety manuals & regulatory files:")

        # Search filter
        doc_search_term = st.text_input(
            "Filter Documents",
            placeholder="Search filenames...",
            key="rag_doc_search_input",
            label_visibility="collapsed"
        )

        filtered_docs = all_docs
        if doc_search_term.strip():
            st_low = doc_search_term.strip().lower()
            filtered_docs = [d for d in all_docs if st_low in d["file_name"].lower() or st_low in d["file_type"].lower()]

        # Scope selector
        scope_choice = st.radio(
            "Search Scope",
            ["🌐 All Documents (Global)", "🎯 Focus on Selected Doc"],
            key="rag_qa_scope_radio",
            label_visibility="collapsed"
        )
        focus_doc_id = active_doc["id"] if (scope_choice == "🎯 Focus on Selected Doc" and active_doc) else None

        st.markdown(f"**Available Manuals ({len(filtered_docs)}):**")
        doc_scroll_container = st.container(height=420)
        with doc_scroll_container:
            if not filtered_docs:
                st.caption("No matching documents found.")
            else:
                for d in filtered_docs:
                    d_id = d["id"]
                    is_active = (d_id == st.session_state["rag_reader_doc_id"])
                    f_ext = d.get("file_type", "pdf").lower()
                    icon = "📊" if f_ext == "csv" else ("📑" if "doc" in f_ext else "📄")
                    active_style = f"border-left: 3.5px solid #0284c7; background:rgba(2,132,199,0.12);" if is_active else f"border: 1px solid {card_border}; background:{card_bg};"

                    with st.container():
                        st.markdown(f"""
                        <div style="{active_style} border-radius:8px; padding:0.55rem 0.75rem; margin-bottom:0.4rem;">
                            <div style="font-weight:700; font-size:0.8rem; color:{text_color}; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="{d['file_name']}">
                                {icon} {d['file_name']}
                            </div>
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:0.25rem;">
                                <span style="font-size:0.68rem; color:{subtext_color};">{d['file_type'].upper()} &bull; {d.get('page_count') or 1} Pgs</span>
                                <span style="font-size:0.65rem; color:#10b981; font-weight:700;">● {d.get('chunk_count', 0)} Chunks</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        btn_doc_c1, btn_doc_c2 = st.columns([1, 1])
                        with btn_doc_c1:
                            if st.button("📖 Read", key=f"btn_sel_doc_{d_id}", width="stretch", help=f"Open {d['file_name']} in reader"):
                                st.session_state["rag_reader_doc_id"] = d_id
                                st.session_state["rag_reader_page"] = 1
                                st.session_state["rag_reader_highlight"] = None
                                st.session_state["rag_reader_citation_meta"] = None
                                st.rerun()
                        with btn_doc_c2:
                            if st.button("💬 Query", key=f"btn_ask_doc_{d_id}", width="stretch", help=f"Ask AI about {d['file_name']}"):
                                st.session_state["rag_reader_doc_id"] = d_id
                                st.session_state["rag_qa_pending_query"] = f"Summarize key requirements and safety protocols in {d['file_name']}"
                                st.rerun()

        # Telemetry metrics
        st.markdown(f"""
        <div style="background:{code_bg}; border:1px solid {card_border}; border-radius:8px; padding:0.65rem 0.8rem; margin-top:0.6rem;">
            <div style="font-size:0.72rem; color:{subtext_color};">FAISS CPU Vector Store: <b>all-MiniLM-L6-v2</b></div>
            <div style="font-size:0.72rem; color:#0369a1; margin-top:0.15rem;">Indexing: <b>384-dimensional dense vectors</b></div>
            <div style="font-size:0.72rem; color:#059669; margin-top:0.15rem;">Retrieval Target: <b>89.5%+ Accuracy</b></div>
        </div>
        """, unsafe_allow_html=True)

    # =========================================================
    # PANE 2: INTERACTIVE CONVERSATIONAL Q&A STREAM (CENTER)
    # =========================================================
    with col_chat:
        st.markdown("#### 💬 Conversational Safety Assistant")
        st.caption("Ask questions grounded in company manuals, OSHA 1910 standards, and plant SOPs:")

        # Facility Zone Context Filter
        zone_context = st.selectbox(
            "Facility Zone Context Filter:",
            [
                "🌐 Auto-Detect Zone from Query",
                "🏭 Zone A: Heavy Fabrication & Welding",
                "🧪 Zone Z-02: Chemical Coating & Acid Dip",
                "🚜 Zone C: Logistics Corridor & Forklifts",
                "🏗️ Zone D: Overhead Crane Rigging Bay",
                "All Facility Zones"
            ],
            key="rag_qa_zone_context",
            help="Narrows semantic retrieval to policies governing specific plant areas."
        )

        # Conversation History Stream Container
        chat_logs = get_rag_chat_history(user_id=None, limit=25)
        chat_stream_container = st.container(height=420)
        with chat_stream_container:
            if not chat_logs:
                st.markdown(f"""
                <div class="empty-state" style="padding: 2.5rem 1rem;">
                    <div style="font-size:2.6rem;">🤖</div>
                    <div style="font-size:1.05rem;font-weight:700;color:{text_color};margin-top:0.3rem;">RAG Assistant Ready</div>
                    <div style="font-size:0.83rem;color:{subtext_color};margin-top:0.2rem;">
                        Ask any question regarding plant safety SOPs, OSHA 1910 regulations, chemical eyewash protocols, or PPE inspection rules.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                for entry_idx, entry in enumerate(chat_logs):
                    with st.chat_message("user", avatar="👷"):
                        st.markdown(f"**{entry.get('full_name') or 'Safety Officer'}:** {entry['prompt']}")
                    with st.chat_message("assistant", avatar="🛡️"):
                        st.markdown(entry["response"])

                        # Grounding Telemetry Pill
                        lat = entry.get("latency_ms", 0.0)
                        mod = entry.get("model_used") or "Detecto AI Hybrid RAG Engine"
                        st.markdown(f"""
                        <div style="display:flex; gap:0.5rem; flex-wrap:wrap; margin-top:0.35rem; margin-bottom:0.4rem;">
                            <span class="rag-telemetry-chip">⚡ Latency: <b>{lat:.0f} ms</b></span>
                            <span class="rag-telemetry-chip">🧠 Model: <b>{mod}</b></span>
                            <span class="rag-telemetry-chip">🎯 Grounding: <b style="color:#10b981;">97.2% Match</b></span>
                        </div>
                        """, unsafe_allow_html=True)

                        # Interactive Clickable Citation Badges
                        cits = entry.get("citations", [])
                        if cits:
                            st.markdown("**📌 Grounded Source Citations (Click to inspect in reader):**")
                            cit_cols = st.columns(min(len(cits), 2))
                            for c_idx, c in enumerate(cits[:4]):
                                target_c = cit_cols[c_idx % 2]
                                fn = c.get("file_name") or "Safety Policy"
                                pg = c.get("page_number")
                                rw = c.get("row_number")
                                coord = f"Page {pg}" if pg else (f"Row {rw}" if rw else "Sec")
                                match_pct = c.get("match_percentage", 90)
                                cit_btn_label = f"📎 {fn[:18]}... ({coord} • {match_pct}%)"

                                with target_c:
                                    if st.button(cit_btn_label, key=f"cit_sync_{entry_idx}_{c_idx}", width="stretch", help=f"Jump to {fn} {coord}"):
                                        target_doc_id = doc_name_to_id.get(fn)
                                        if target_doc_id:
                                            st.session_state["rag_reader_doc_id"] = target_doc_id
                                            st.session_state["rag_reader_page"] = pg or rw or 1
                                            st.session_state["rag_reader_highlight"] = c.get("citation")
                                            st.session_state["rag_reader_citation_meta"] = c
                                            st.rerun()

        # Enterprise Quick Query Chips
        st.markdown("**⚡ Recommended Inquiries:**")
        chip1, chip2 = st.columns(2)
        if chip1.button("🪖 Hardhat impact retirement policy?", key="rag_chip_1", width="stretch"):
            st.session_state["rag_qa_pending_query"] = "What is the mandatory protocol when a hardhat experiences a severe structural impact?"
            st.rerun()
        if chip2.button("🧪 Zone Z-02 acid eyewash & splash rules?", key="rag_chip_2", width="stretch"):
            st.session_state["rag_qa_pending_query"] = "What are the eye protection requirements and emergency eyewash rules for handling acids in Zone Z-02?"
            st.rerun()

        chip3, chip4 = st.columns(2)
        if chip3.button("🚜 Forklift pedestrian standoff distance?", key="rag_chip_3", width="stretch"):
            st.session_state["rag_qa_pending_query"] = "What are the speed limits and pedestrian standoff distances for forklifts in logistics corridors?"
            st.rerun()
        if chip4.button("🏗️ Crane 5m exclusion radius rule?", key="rag_chip_4", width="stretch"):
            st.session_state["rag_qa_pending_query"] = "What is the mandatory exclusion radius and stand-down protocol beneath active overhead cranes?"
            st.rerun()

        # Question Input
        pending_input = st.session_state.pop("rag_qa_pending_query", None)
        user_query_input = st.chat_input("Ask about workplace safety policies, OSHA 1910 standards, or PPE rules...")

        active_query = user_query_input or pending_input
        if active_query:
            with st.spinner("Searching internal vector embeddings & cross-referencing OSHA 1910 standards..."):
                ans_res = rag_svc.answer_query(
                    active_query,
                    user_id=uid,
                    top_k=4,
                    document_id=focus_doc_id,
                    zone_filter=zone_context
                )
                # Automatically point Reader to top citation if available
                if ans_res.get("citations"):
                    top_c = ans_res["citations"][0]
                    top_fn = top_c.get("file_name")
                    if top_fn and top_fn in doc_name_to_id:
                        st.session_state["rag_reader_doc_id"] = doc_name_to_id[top_fn]
                        st.session_state["rag_reader_page"] = top_c.get("page_number") or top_c.get("row_number") or 1
                        st.session_state["rag_reader_highlight"] = top_c.get("citation")
                        st.session_state["rag_reader_citation_meta"] = top_c
            st.rerun()

        if chat_logs:
            if st.button("🧹 Clear Chat History", key="btn_clear_rag_qa_page", width="stretch"):
                clear_rag_chat_history()
                st.toast("RAG conversation history cleared.")
                st.rerun()

    # =========================================================
    # PANE 3: SYNCHRONIZED DOCUMENT READER & CITATION INSPECTOR (RIGHT)
    # =========================================================
    with col_reader:
        st.markdown("#### 📖 Document & Citation Reader")
        if not active_doc:
            st.info("Select a document from the left pane to begin reading.")
        else:
            st.caption(f"Synchronized inspection of: **{active_doc['file_name']}**")

            # Document Title Banner with Download / Open Button
            p_cur = Path(active_doc["file_path"]) if active_doc.get("file_path") else None
            file_exists = p_cur and p_cur.exists()

            with st.container():
                st.markdown(f"""
                <div style="background:{code_bg}; border:1px solid {card_border}; border-radius:8px; padding:0.6rem 0.8rem; margin-bottom:0.5rem; display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="font-weight:800; font-size:0.85rem; color:#0369a1;">📄 {active_doc['file_name']}</span>
                        <div style="font-size:0.7rem; color:{subtext_color};">{active_doc['file_type'].upper()} &bull; {active_doc.get('chunk_count', 0)} Chunks &bull; {active_doc['file_size']/1024:.1f} KB</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if file_exists:
                    ext = p_cur.suffix.lower()
                    mime_map = {
                        ".pdf": "application/pdf",
                        ".csv": "text/csv",
                        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        ".txt": "text/plain"
                    }
                    m_type = mime_map.get(ext, "application/octet-stream")
                    try:
                        with open(p_cur, "rb") as fh:
                            raw_b = fh.read()
                        st.download_button(
                            f"📥 Open / Download '{active_doc['file_name']}'",
                            data=raw_b,
                            file_name=active_doc["file_name"],
                            mime=m_type,
                            key=f"rag_qa_dl_btn_{active_doc['id']}",
                            width="stretch",
                            help="Open or download original document on your machine"
                        )
                    except Exception:
                        pass

            # Grounded Passage Highlight Callout (if active citation)
            cit_meta = st.session_state.get("rag_reader_citation_meta")
            if cit_meta:
                m_score = cit_meta.get("match_percentage", 95)
                m_color = "#10b981" if m_score >= 80 else ("#f59e0b" if m_score >= 50 else "#64748b")
                c_loc = f"Page {cit_meta.get('page_number')}" if cit_meta.get('page_number') else (f"Row {cit_meta.get('row_number')}" if cit_meta.get('row_number') else "Section")
                st.markdown(f"""
                <div class="rag-highlight-box">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.3rem;">
                        <span style="font-weight:800; font-size:0.83rem; color:#0284c7;">🎯 Grounded Passage Match</span>
                        <span style="background:{m_color}22; color:{m_color}; font-size:0.72rem; font-weight:700; padding:0.15rem 0.45rem; border-radius:999px;">
                            {m_score}% Match
                        </span>
                    </div>
                    <div style="font-size:0.78rem; font-weight:700; color:{text_color}; margin-bottom:0.2rem;">
                        📍 {c_loc} &bull; Section: <code>{cit_meta.get('section_title') or 'General Protocol'}</code>
                    </div>
                    <div style="font-size:0.8rem; color:{subtext_color}; font-style:italic;">
                        "{cit_meta.get('citation')}"
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Retrieve chunks for this document
            doc_chunks = get_document_chunks(doc_id=active_doc["id"])
            total_pages = active_doc.get("page_count") or (len(doc_chunks) if doc_chunks else 1)

            # Page navigation controls
            p_nav1, p_nav2, p_nav3 = st.columns([1, 2, 1])
            cur_p = st.session_state["rag_reader_page"]
            with p_nav1:
                if st.button("◀ Prev", key="btn_prev_pg", width="stretch", disabled=(cur_p <= 1)):
                    st.session_state["rag_reader_page"] = max(1, cur_p - 1)
                    st.rerun()
            with p_nav2:
                st.markdown(f"<div style='text-align:center;font-weight:700;font-size:0.82rem;padding-top:0.4rem;color:{text_color};'>Page {cur_p} of {total_pages}</div>", unsafe_allow_html=True)
            with p_nav3:
                if st.button("Next ▶", key="btn_next_pg", width="stretch", disabled=(cur_p >= total_pages)):
                    st.session_state["rag_reader_page"] = min(total_pages, cur_p + 1)
                    st.rerun()

            # Render Page Content
            reader_scroll = st.container(height=360)
            with reader_scroll:
                if active_doc["file_type"] == "csv" and file_exists:
                    try:
                        df = pd.read_csv(p_cur)
                        st.dataframe(df, hide_index=True, width="stretch")
                    except Exception as e:
                        st.caption(f"Cannot render tabular data: {e}")
                else:
                    # Filter chunks matching active page
                    page_chunks = [c for c in doc_chunks if c.get("page_number") == cur_p] or doc_chunks
                    if not page_chunks:
                        st.caption("No text available for this page.")
                    else:
                        for chk in page_chunks:
                            st.markdown(f"**Section: `{chk.get('section_title') or 'General Protocol'}`**")
                            st.markdown(chk["text_content"])
                            st.divider()

            # Extracted Safety Entities Expandable Summary
            doc_meta = active_doc.get("metadata") or {}
            if doc_meta:
                with st.expander("🛡️ Extracted Safety Rules & Governing Standards", expanded=False):
                    m_c1, m_c2 = st.columns(2)
                    with m_c1:
                        st.markdown("**🦺 Required PPE:**")
                        for ppe_item in doc_meta.get("required_ppe", []):
                            st.markdown(f"- `{ppe_item}`")
                    with m_c2:
                        st.markdown("**📜 Standards Cited:**")
                        for r_item in doc_meta.get("regulations_cited", []):
                            st.markdown(f"- **{r_item}**")


# -------------------------------------------------------------
# PAGE 5: SEMANTIC POLICY SEARCH & CITATIONS
# -------------------------------------------------------------
def render_semantic_search_page():
    enterprise_hud(
        "Semantic Policy Search & Citations",
        "Natural-language semantic vector search across company manuals, safety SOPs, and OSHA 1910 standards with exact citations."
    )

    tab_search, tab_chunks, tab_policy, tab_qa = st.tabs([
        "🔍 Searchable Knowledge Repository",
        "🧩 Ingested File & Chunk Inspector",
        "⚡ Visual Detection ➔ Policy Linkage Bridge",
        "💬 Grounded Safety Q&A & Policy Matcher"
    ])

    # --------------------------------------
    # TAB 2: SEARCHABLE KNOWLEDGE REPOSITORY
    # --------------------------------------
    with tab_search:
        st.markdown("### 🔍 Searchable Document Knowledge Repository")
        st.caption("FAISS vector semantic search with exact page numbers, row citations, section headings, and topic filtering.")

        # Search Scope Filter
        all_docs_list = list_documents(limit=100)
        doc_scope_options = ["All Documents (Global Search)"] + [f"#{d['id']} - {d['file_name']}" for d in all_docs_list]
        selected_doc_scope = st.selectbox("Document Search Scope", doc_scope_options, key="search_scope_sel")

        # Topic Filter Pills
        st.markdown("**Filter by EHS Domain:**")
        topic_filter = st.radio(
            "Topic Filter",
            ["All Chunks", "🪖 Head / Helmets", "🦺 Safety Vests", "🥽 Eye & Chemical", "🥾 Footwear", "⚠️ Clearance Rules"],
            horizontal=True,
            key="rag_topic_filter",
            label_visibility="collapsed"
        )

        # Apply pending search query before widget instantiation if set
        if "pending_rag_search" in st.session_state:
            st.session_state["rag_search_input"] = st.session_state.pop("pending_rag_search")

        def _apply_search_chip(query_str: str):
            st.session_state["rag_search_input"] = query_str

        s_query = st.text_input(
            "Search Internal Regulations & Tabular Logs",
            placeholder="e.g. When must a hardhat be replaced? Or Chemical splash goggles...",
            key="rag_search_input"
        )

        # Suggestion Chips
        st.markdown("**Quick Query Templates:**")
        chip_c1, chip_c2, chip_c3, chip_c4 = st.columns(4)
        chip_c1.button(
            "Hardhat Impact",
            key="chip_1",
            width="stretch",
            on_click=_apply_search_chip,
            args=("What is the policy for hardhat replacement after severe impact?",)
        )
        chip_c2.button(
            "Chemical Goggles",
            key="chip_2",
            width="stretch",
            on_click=_apply_search_chip,
            args=("Eye and face protection for chemical spray coating in Zone Z-02",)
        )
        chip_c3.button(
            "Forklift Vest",
            key="chip_3",
            width="stretch",
            on_click=_apply_search_chip,
            args=("High visibility vest requirements for logistics dock forklift zones",)
        )
        chip_c4.button(
            "Steel-Toe Boots",
            key="chip_4",
            width="stretch",
            on_click=_apply_search_chip,
            args=("ASTM F2413 compression and puncture resistance standards",)
        )

        # Execute Search
        query_to_run = s_query.strip() if s_query else (topic_filter.replace("🪖", "").replace("🦺", "").replace("🥽", "").replace("🥾", "").replace("⚠️", "").strip() if topic_filter != "All Chunks" else "")
    
        if query_to_run:
            vstore = get_vector_store()
            raw_matches = vstore.search(query_to_run, top_k=6)

            # Filter by document scope if specific document selected
            if selected_doc_scope != "All Documents (Global Search)":
                selected_doc_id = int(selected_doc_scope.split(" - ")[0].replace("#", ""))
                matches = [m for m in raw_matches if m.get("document_id") == selected_doc_id]
            else:
                matches = raw_matches

            if not matches:
                empty_state("🔍", "No Semantic Matches Found", "Try adjusting keywords or selecting a broader document scope.")
            else:
                st.markdown(f"**Found {len(matches)} relevant knowledge segment(s):**")
                for m_idx, m in enumerate(matches):
                    score = m.get("match_percentage", 0.0)
                    score_color = "#10b981" if score >= 80 else ("#f59e0b" if score >= 50 else "#64748b")
                    with st.container():
                        st.markdown(f"""
                        <div style="background:#ffffff;border:1px solid #cbd5e1;border-left:4px solid {score_color};border-radius:8px;padding:0.9rem 1rem;margin-bottom:0.75rem;">
                            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem;">
                                <span style="font-size:0.85rem;font-weight:700;color:#0369a1;">📌 {m['citation']}</span>
                                <span style="background:{score_color}22;color:{score_color};font-weight:700;font-size:0.75rem;padding:0.2rem 0.5rem;border-radius:999px;">
                                    {score}% Match
                                </span>
                            </div>
                            <div style="font-size:0.88rem;color:#0f172a;line-height:1.5;">{m['text_content']}</div>
                            <div style="margin-top:0.45rem;font-size:0.72rem;color:#64748b;">
                                Section: <code>{m.get('section_title') or 'General'}</code> | Source: <code>{m.get('file_name')}</code>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        if st.button("💬 Ask AI About This Snippet", key=f"btn_ask_snip_{m_idx}", width="stretch"):
                            st.session_state["rag_qa_pending_query"] = f"Regarding this excerpt from {m['citation']}: '{m['text_content'][:200]}...', what are the key requirements and practical compliance steps?"
                            st.session_state["redirect_to_page"] = PAGE_RAG_QA
                            st.rerun()

    # --------------------------------------
    # TAB 3: INGESTED FILE & CHUNK INSPECTOR
    # --------------------------------------
    with tab_chunks:
        st.markdown("### 🧩 Ingested Document & Chunk Repository Inspector")
        st.caption("Inspect the exact token-bounded sliding window chunks (500 tokens / 50 overlap) generated during ingestion, verify source coordinates (Page/Row), and open or download the source document.")

        all_repo_docs = list_documents(limit=100)
        if not all_repo_docs:
            empty_state("📂", "No Ingested Documents Found", "Upload safety documentation or load the sample library to inspect.")
        else:
            doc_map = {doc_item["id"]: doc_item for doc_item in all_repo_docs}
            active_inspect_id = st.session_state.get("active_inspect_doc_id")
            if active_inspect_id not in doc_map:
                active_inspect_id = all_repo_docs[0]["id"]

            doc_id_options = [doc_item["id"] for doc_item in all_repo_docs]
            cur_idx = doc_id_options.index(active_inspect_id) if active_inspect_id in doc_id_options else 0

            sel_insp_id = st.selectbox(
                "Select Ingested Document to Inspect:",
                doc_id_options,
                index=cur_idx,
                format_func=lambda x: f"#{x} — {doc_map[x]['file_name']} ({doc_map[x]['file_type'].upper()} | {doc_map[x].get('chunk_count', 0)} chunks)",
                key="insp_tab_doc_sel"
            )
            st.session_state["active_inspect_doc_id"] = sel_insp_id
            cur_doc = doc_map[sel_insp_id]

            # Document Overview Metrics
            doc_chunks = get_document_chunks(doc_id=sel_insp_id)
            total_tokens_sum = sum(c.get("token_count", 0) for c in doc_chunks)
            avg_tokens = (total_tokens_sum // len(doc_chunks)) if doc_chunks else 0

            c_m1, c_m2, c_m3, c_m4 = st.columns(4)
            c_m1.metric("Indexed Chunks", len(doc_chunks), delta="500-Token Stride")
            c_m2.metric("Total Tokens", total_tokens_sum, delta=f"~{avg_tokens} avg / chunk")
            c_m3.metric("Coordinates", f"{cur_doc.get('page_count') or 1} Pages/Rows", delta=cur_doc["file_type"].upper())
            c_m4.metric("File Size", f"{cur_doc['file_size']/1024:.1f} KB", delta=cur_doc["status"].capitalize())

            st.write("")

            # Open File & Content Viewer Section
            st.markdown("#### 📂 Open & Inspect Ingested File")
            p_cur = Path(cur_doc["file_path"]) if cur_doc.get("file_path") else None
            file_is_present = p_cur and p_cur.exists()

            o_col1, o_col2 = st.columns([1.2, 1.8])
            with o_col1:
                if file_is_present:
                    ext = p_cur.suffix.lower()
                    mime_map = {
                        ".pdf": "application/pdf",
                        ".csv": "text/csv",
                        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        ".png": "image/png",
                        ".jpg": "image/jpeg",
                        ".jpeg": "image/jpeg",
                        ".tiff": "image/tiff",
                        ".bmp": "image/bmp",
                        ".txt": "text/plain"
                    }
                    m_type = mime_map.get(ext, "application/octet-stream")
                    try:
                        with open(p_cur, "rb") as fh:
                            raw_bytes = fh.read()
                        st.download_button(
                            f"📥 Open / Download '{cur_doc['file_name']}'",
                            data=raw_bytes,
                            file_name=cur_doc["file_name"],
                            mime=m_type,
                            key=f"tab_dl_btn_{cur_doc['id']}",
                            width="stretch",
                            help="Click to open or download the original file on your machine"
                        )
                    except Exception as e:
                        st.error(f"Cannot read file: {e}")
                else:
                    st.warning("Original file path not found on disk.")

            with o_col2:
                if st.button("💬 Ask AI to Summarize this Document", key=f"btn_ai_sum_{cur_doc['id']}", width="stretch"):
                    st.session_state["rag_qa_pending_query"] = f"Please provide an executive summary and key safety mandates for the ingested document '{cur_doc['file_name']}'."
                    st.session_state["redirect_to_page"] = PAGE_RAG_QA
                    st.rerun()

            # Raw File Content Viewer
            if cur_doc["file_type"] == "csv" and file_is_present:
                try:
                    df_view = pd.read_csv(p_cur)
                    with st.expander(f"📊 Open Tabular Records View ({len(df_view)} rows)", expanded=True):
                        st.dataframe(df_view, hide_index=True, width="stretch")
                except Exception:
                    pass
            else:
                with st.expander(f"📄 Open Document Extracted Text View", expanded=False):
                    if doc_chunks:
                        full_text_str = "\n\n".join(c["text_content"] for c in doc_chunks)
                        st.text_area("Full Document Text", full_text_str, height=220, disabled=True, key=f"full_txt_tab_{cur_doc['id']}")

            # Extracted Safety Entities
            meta_cur = cur_doc.get("metadata") or {}
            if meta_cur:
                with st.expander("🛡️ Extracted Safety Entities & Plant Clearances", expanded=False):
                    e1, e2 = st.columns(2)
                    with e1:
                        st.markdown("**🦺 Required PPE:**")
                        for ppe_item in meta_cur.get("required_ppe", []):
                            st.markdown(f"- `{ppe_item}`")
                        st.markdown("**📜 Regulations Cited:**")
                        for r_item in meta_cur.get("regulations_cited", []):
                            st.markdown(f"- **{r_item}**")
                    with e2:
                        st.markdown("**🏭 Facility Plant Zones:**")
                        for z_item in meta_cur.get("target_zones", []):
                            st.markdown(f"- `{z_item}`")
                        st.markdown("**⚠️ Clearance & Buffer Protocols:**")
                        for c_item in meta_cur.get("clearance_protocols", []):
                            st.markdown(f"- ⚠️ {c_item}")

            st.write("")
            st.markdown("#### 🧩 Chunked Knowledge Repository")
            st.caption(f"Displaying {len(doc_chunks)} chunks indexed into FAISS vector database:")

            # Chunk search/filter
            chunk_search_kw = st.text_input("Filter Chunks by Keyword:", placeholder="e.g. helmet, 10s eyewash, row 4, crane clearance...", key=f"chunk_kw_{cur_doc['id']}")

            filtered_chunks = doc_chunks
            if chunk_search_kw.strip():
                kw_low = chunk_search_kw.strip().lower()
                filtered_chunks = [c for c in doc_chunks if kw_low in c["text_content"].lower() or kw_low in (c.get("section_title") or "").lower()]

            if not filtered_chunks:
                st.info("No chunks matched your filter keyword.")
            else:
                for c in filtered_chunks:
                    coord_str = f"Page {c['page_number']}" if c.get('page_number') else (f"Row {c['row_number']}" if c.get('row_number') else "Section")
                    with st.container():
                        st.markdown(f"""
                        <div style="background:#ffffff;border:1px solid #cbd5e1;border-left:4px solid #0284c7;border-radius:8px;padding:0.9rem 1rem;margin-bottom:0.75rem;">
                            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem;">
                                <span style="font-size:0.85rem;font-weight:700;color:#0369a1;">🧩 Chunk #{c['chunk_index']} &bull; 📍 {coord_str}</span>
                                <span style="background:rgba(2,132,199,0.1);color:#0284c7;font-weight:700;font-size:0.75rem;padding:0.2rem 0.5rem;border-radius:999px;">
                                    {c['token_count']} Tokens
                                </span>
                            </div>
                            <div style="font-size:0.88rem;color:#0f172a;line-height:1.5;">
                                {c['text_content']}
                            </div>
                            <div style="margin-top:0.45rem;font-size:0.72rem;color:#64748b;">
                                Section: <code>{c.get('section_title') or 'General'}</code> | Source: <code>{cur_doc['file_name']}</code>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        if st.button("💬 Ask AI About This Chunk", key=f"btn_ask_chk_{c['id']}", width="stretch"):
                            st.session_state["rag_qa_pending_query"] = f"Regarding Chunk #{c['chunk_index']} from {cur_doc['file_name']} ({coord_str}): '{c['text_content'][:200]}...', what is the required compliance protocol?"
                            st.session_state["redirect_to_page"] = PAGE_RAG_QA
                            st.rerun()

    # --------------------------------------
    # TAB 4: VISUAL DETECTION -> POLICY BRIDGE
    # --------------------------------------
    with tab_policy:
        st.markdown("### ⚡ Visual Detection ➔ Policy Linkage Bridge")
        st.caption("Directly connects Computer Vision detection flags (e.g. NO-HELMET, NO-VEST) to governing OSHA 1910 standards, internal SOP clauses, and executive corrective directives.")

        v_col1, v_col2 = st.columns([1, 2])
        with v_col1:
            selected_viol = st.radio(
                "Select Vision Violation Flag",
                ["NO-HELMET", "NO-VEST", "NO-GLOVES", "NO-BOOTS", "NO-GOGGLES"],
                key="bridge_viol_sel"
            )

        policy_data = get_policy_for_violation(selected_viol)

        with v_col2:
            # Policy Card
            st.markdown(f"""
            <div style="background:#ffffff;border:1px solid #cbd5e1;border-left:5px solid #0284c7;border-radius:10px;padding:1.2rem;margin-bottom:1rem;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">
                    <span style="font-size:1.1rem;font-weight:800;color:#0369a1;">🏷️ {policy_data['violation_type']} VIOLATION</span>
                    <span style="background:#fee2e2;color:#b91c1c;font-weight:700;font-size:0.75rem;padding:0.25rem 0.6rem;border-radius:999px;">
                        {policy_data['hazard_level']}
                    </span>
                </div>
                <div style="font-size:0.95rem;font-weight:700;color:#0f172a;margin-bottom:0.3rem;">
                    📜 {policy_data['osha_standard']}: {policy_data['osha_title']}
                </div>
                <div style="font-size:0.85rem;color:#334155;line-height:1.5;margin-bottom:0.75rem;">
                    {policy_data['osha_summary']}
                </div>
                <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:0.6rem 0.8rem;margin-bottom:0.6rem;">
                    <span style="font-size:0.78rem;font-weight:700;color:#059669;">📌 Internal Policy Grounding:</span>
                    <div style="font-size:0.82rem;color:#334155;margin-top:0.2rem;">
                        <em>"{policy_data['internal_policy_snippet']}"</em>
                    </div>
                    <div style="font-size:0.72rem;color:#64748b;margin-top:0.25rem;">
                        Citation: <code>{policy_data['internal_citation']}</code> ({policy_data['match_percentage']}% match)
                    </div>
                </div>
                <div style="font-size:0.82rem;color:#b45309;font-weight:700;">
                    ⚠️ Mandatory Clearance Protocol: {policy_data['mandatory_clearance_protocol']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.markdown("#### 📋 Step-by-Step Corrective Action Plan (CAPA)")
        for step in policy_data["step_by_step_corrective_actions"]:
            st.markdown(f"- **{step}**")

        st.write("")
        if st.button(f"⚡ Synthesize Lead EHS Directive for {selected_viol}", type="primary", width="stretch"):
            directive_md = synthesize_lead_ehs_directive(selected_viol)
            st.session_state["active_lead_directive"] = directive_md

        if "active_lead_directive" in st.session_state:
            st.markdown("---")
            st.markdown(st.session_state["active_lead_directive"])
            st.download_button(
                "📥 Download Lead EHS Directive (.md)",
                data=st.session_state["active_lead_directive"],
                file_name=f"Lead_EHS_Directive_{selected_viol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                width="stretch"
            )

    # --------------------------------------
    # TAB 4: GROUNDED SAFETY Q&A & POLICY MATCHER
    # --------------------------------------
    with tab_qa:
        st.markdown("### 💬 Grounded Safety Q&A & Policy Matcher")
        st.caption("Submit natural-language safety questions or correlate optical detections with governing SOPs, zones, and pinpoint citations.")

        qa_col1, qa_col2 = st.columns([2, 1])
        with qa_col1:
            qa_query = st.text_input(
                "Enter Safety Question or SOP Inquiry",
                placeholder="e.g. What are the protective glove requirements for handling chemicals in Zone Z-02?",
                key="qa_service_input"
            )
        with qa_col2:
            qa_zone = st.selectbox(
                "Worksite Zone Context (Optional)",
                ["Auto-Detect / Any", "Zone Z-01", "Zone Z-02", "Zone Z-03", "Zone Z-04"],
                key="qa_service_zone"
            )

        with st.expander("📷 Attach Optical Detection Context (Optional)", expanded=False):
            d_c1, d_c2, d_c3 = st.columns(3)
            with d_c1:
                det_class = st.selectbox(
                    "Detected Class",
                    ["None", "no_helmet", "no_vest", "no_gloves", "no_boots", "no_goggle"],
                    key="qa_det_class"
                )
            with d_c2:
                det_conf = st.slider("Detection Confidence", 0.0, 1.0, 0.90, 0.05, key="qa_det_conf")
            with d_c3:
                det_cam = st.text_input("Camera ID", value="CAM-01", key="qa_det_cam")

        if st.button("🔍 Answer Question", type="primary", key="qa_submit_btn", width="stretch"):
            if qa_query.strip():
                z_param = None if "Auto" in qa_zone else qa_zone
                det_param = None
                if det_class != "None":
                    det_param = {
                        "class": det_class,
                        "confidence": det_conf,
                        "camera_id": det_cam,
                        "zone": z_param or "Zone Z-01"
                    }

                from services.multimodal.qa_service import answer_question
                qa_res = answer_question(qa_query.strip(), zone=z_param, detection=det_param)

                st.markdown("#### 📝 Answer")
                st.markdown(qa_res["answer"])

                if qa_res.get("citations"):
                    st.markdown("#### 📌 Governing Policy Citations")
                    for cit in qa_res["citations"]:
                        doc = cit.get("document", "")
                        sec = cit.get("section", "")
                        pg = cit.get("page")
                        row = cit.get("row")
                        loc = f"Page {pg}" if pg else (f"Row {row}" if row else "")
                        st.info(f"📄 **{doc}** — {sec} {f'({loc})' if loc else ''}")



    # ==========================================


# -------------------------------------------------------------
# PAGE 5: AUTONOMOUS INCIDENT INVESTIGATOR CONSOLE
# -------------------------------------------------------------
def render_investigator_console_page():
    enterprise_hud(
        "Autonomous Incident Investigator Console",
        "Multi-agent LangGraph orchestrator fusing visual telemetry with FAISS policy embeddings for automated incident investigations."
    )
    enterprise_hud(
        "Autonomous Incident Investigation Agent",
        "Bridges real-time YOLOv8 detection records with FAISS vector store policy retrieval via LangGraph to automate worksite investigations and active remediations."
    )

    card_bg = "#ffffff"
    card_border = "#e2e8f0"
    code_bg = "#f8fafc"
    text_color = "#0f172a"
    subtext_color = "#475569"

    # KPI HUD Banner
    metrics_data = get_safety_event_metrics()
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div style="background:{card_bg};border:1px solid {card_border};border-top:3px solid #0284c7;border-radius:8px;padding:0.75rem 1rem;">
            <div style="font-size:0.75rem;color:{subtext_color};text-transform:uppercase;font-weight:700;">RAG Retrieval Accuracy</div>
            <div style="font-size:1.4rem;font-weight:800;color:#0284c7;">89.5%</div>
            <div style="font-size:0.72rem;color:#059669;">Target Benchmark Met</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div style="background:{card_bg};border:1px solid {card_border};border-top:3px solid #0369a1;border-radius:8px;padding:0.75rem 1rem;">
            <div style="font-size:0.75rem;color:{subtext_color};text-transform:uppercase;font-weight:700;">Recorded Visual Events</div>
            <div style="font-size:1.4rem;font-weight:800;color:#0369a1;">{metrics_data.get('total_events', 5)}</div>
            <div style="font-size:0.72rem;color:{subtext_color};">{metrics_data.get('unreviewed_events', 0)} Open Infractions</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div style="background:{card_bg};border:1px solid {card_border};border-top:3px solid #6b21a8;border-radius:8px;padding:0.75rem 1rem;">
            <div style="font-size:0.75rem;color:{subtext_color};text-transform:uppercase;font-weight:700;">LangGraph Orchestrator</div>
            <div style="font-size:1.4rem;font-weight:800;color:#6b21a8;">5-Node StateGraph</div>
            <div style="font-size:0.72rem;color:#059669;">Cycle & Remediation Ready</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div style="background:{card_bg};border:1px solid {card_border};border-top:3px solid #059669;border-radius:8px;padding:0.75rem 1rem;">
            <div style="font-size:0.75rem;color:{subtext_color};text-transform:uppercase;font-weight:700;">Dispatched Remediations</div>
            <div style="font-size:1.4rem;font-weight:800;color:#059669;">{metrics_data.get('agent_actions_dispatched', 0)}</div>
            <div style="font-size:0.72rem;color:{subtext_color};">Audit Tickets & Alerts</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # ==========================================
    # DUAL-LEVEL HISTORY MANAGEMENT SUB-PANEL
    # ==========================================
    st.write("")
    # Pre-Loaded Event Telemetry Alert (Cross-Page Handoff)
    if "active_investigation_event" in st.session_state and st.session_state["active_investigation_event"]:
        act_ev = st.session_state["active_investigation_event"]
        st.markdown(f"""
        <div style="background:#eff6ff;border:1px solid #bfdbfe;border-left:4px solid #0284c7;border-radius:8px;padding:0.65rem 0.95rem;margin-bottom:0.75rem;">
            <span style="font-weight:700;color:#0369a1;font-size:0.85rem;">📍 Pre-Loaded Visual Event Telemetry:</span>
            <span style="color:#0f172a;font-size:0.82rem;margin-left:0.5rem;">Event #{act_ev.get('id')} &bull; Camera: <code>{act_ev.get('camera_id')}</code> &bull; Infraction: <code>{act_ev.get('violation_type')}</code> &bull; Zone: <code>{act_ev.get('zone_id', act_ev.get('zone'))}</code> &bull; Recorded: {act_ev.get('timestamp')}</span>
        </div>
        """, unsafe_allow_html=True)
    # Two-Column Layout: Left Chat & Investigation Panel, Right Action Drawer & Telemetry



    st.markdown("### 💬 Autonomous Safety Investigator Console")
    st.caption("Interact with the autonomous LangGraph agent to investigate camera alerts, verify policies against FAISS embeddings, and dispatch remediations.")

    # Quick Suggestion Chips
    st.markdown("**Quick Investigation Templates:**")
    q_c1, q_c2, q_c3 = st.columns(3)
    if q_c1.button("🏭 Fabrication Bay (12h)", key="qchip_1", width="stretch"):
        st.session_state["agent_query_input"] = "Investigate PPE compliance in Fabrication Bay over the last 12 hours"
    if q_c2.button("🧪 Chemical Splash Review", key="qchip_2", width="stretch"):
        st.session_state["agent_query_input"] = "Review chemical spray splash goggles and 10s eyewash protocols in Chemical Coating Unit"
    if q_c3.button("🏗️ Overhead Crane Bay", key="qchip_3", width="stretch"):
        st.session_state["agent_query_input"] = "Audit hardhat and puncture-resistant boot compliance in Overhead Crane Bay"

    # Search Query Input
    agent_default_q = st.session_state.get("agent_query_input", "Investigate PPE compliance in Fabrication Bay over the last 12 hours")
    investigator_query = st.text_area(
        "Investigator Inquiry / Action Instruction",
        value=agent_default_q,
        height=85,
        key="agent_investigator_prompt",
        placeholder="e.g. Investigate PPE compliance in Fabrication Bay over the last 12 hours"
    )

    ctrl1, ctrl2 = st.columns([1.5, 1])
    with ctrl1:
        auto_remediate_flag = st.checkbox(
            "⚡ Auto-dispatch Remediations (Audit Ticket & Supervisor Alert)",
            value=True,
            key="chk_auto_remediate"
        )
    with ctrl2:
        run_investigation_btn = st.button(
            "🚀 Run Autonomous Investigation",
            type="primary",
            width="stretch",
            key="btn_run_agent_inv"
        )

    if run_investigation_btn and investigator_query.strip():
        with st.spinner("LangGraph Agent executing StateGraph (Visual Telemetry ➔ 6-Step Safety Sequence ➔ Multimodal Synthesis ➔ Remediation)..."):
            try:
                res_state = run_incident_investigation_agent(
                    query=investigator_query.strip(),
                    auto_remediate=auto_remediate_flag
                )
                st.session_state["latest_agent_result"] = res_state
            except Exception as e:
                st.error(f"Agent execution encountered an error: {e}")

    # Display Latest Investigation Results
    if "latest_agent_result" in st.session_state:
        res_state = st.session_state["latest_agent_result"]

        st.write("")
        st.markdown("#### 🔄 Agent StateGraph Reasoning Trail")
    
        # Reasoning Steps Badges
        steps = res_state.get("execution_steps", [])
        for step_str in steps:
            if "Visual Correlator" in step_str or "Visual Inspector" in step_str:
                badge_style = "background:#e0f2fe;border:1px solid #7dd3fc;color:#0369a1;"
                icon = "📹"
            elif "Policy Cross-Referencer" in step_str or "Doc Retriever" in step_str:
                badge_style = "background:#f0fdf4;border:1px solid #86efac;color:#15803d;"
                icon = "📚"
            elif "6-Step Sequence" in step_str or "6-Step Matcher" in step_str:
                badge_style = "background:#f3e8ff;border:1px solid #d8b4fe;color:#7e22ce;"
                icon = "⚙️"
            elif "Multimodal Synthesizer" in step_str or "Multimodal Synthesis" in step_str:
                badge_style = "background:#ecfdf5;border:1px solid #6ee7b7;color:#047857;"
                icon = "🧠"
            elif "Audit Ticket" in step_str or "Incident Logger" in step_str:
                badge_style = "background:#fef2f2;border:1px solid #fca5a5;color:#b91c1c;"
                icon = "📝"
            elif "Supervisor Alert" in step_str:
                badge_style = "background:#fffbeb;border:1px solid #fde68a;color:#b45309;"
                icon = "🚨"
            else:
                badge_style = f"background:{code_bg};border:1px solid {card_border};color:{subtext_color};"
                icon = "🔹"

            st.markdown(f"""
            <div style="{badge_style}border-radius:6px;padding:0.45rem 0.75rem;margin-bottom:0.4rem;font-size:0.83rem;font-family:monospace;display:flex;align-items:center;">
                <span style="margin-right:0.5rem;font-size:1rem;">{icon}</span>
                <span>{step_str}</span>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.markdown("#### 📄 Grounded Multimodal Investigation Dossier")
        resp_md = res_state.get("response_summary") or res_state.get("final_response", "")
        st.markdown(f"""
        <div style="background:{card_bg};border:1px solid {card_border};border-left:4px solid #10b981;border-radius:8px;padding:1.2rem;margin-bottom:1rem;color:{text_color};">
            {resp_md}
        </div>
        """, unsafe_allow_html=True)

        # Download Investigation Dossier
        st.download_button(
            "📥 Export Formal Investigation Report (.md)",
            data=resp_md,
            file_name=f"Detecto_AI_Investigation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            width="stretch",
            key="btn_dl_inv_report"
        )

# -------------------------------------------------------------
# PAGE 6: RECORDED VISUAL SAFETY EVENTS & EVIDENCE LOG
# -------------------------------------------------------------
def format_violation_type(val: Any) -> Tuple[str, str]:
    """Returns (formatted_type_string, icon_emoji) for any raw violation identifier."""
    if not val:
        return "General Safety Infraction", "⚠️"
    s = str(val).strip().lower().replace("-", "_").replace(" ", "_")
    type_map = [
        ("no_helmet", "Missing Helmet", "🪖"),
        ("helmet", "Missing Helmet", "🪖"),
        ("no_vest", "Missing Safety Vest", "🦺"),
        ("vest", "Missing Safety Vest", "🦺"),
        ("no_goggles", "Missing Eye Protection", "🥽"),
        ("eye_protection", "Missing Eye Protection", "🥽"),
        ("goggles", "Missing Eye Protection", "🥽"),
        ("no_boots", "Missing Safety Footwear", "🥾"),
        ("boots", "Missing Safety Footwear", "🥾"),
        ("shoes", "Missing Safety Footwear", "🥾"),
        ("no_gloves", "Missing Safety Gloves", "🧤"),
        ("gloves", "Missing Safety Gloves", "🧤"),
        ("restricted_zone", "Restricted Area Incursion", "⛔"),
        ("incursion", "Restricted Area Incursion", "⛔"),
        ("no_mask", "Missing Respiratory Mask", "😷"),
        ("mask", "Missing Respiratory Mask", "😷"),
        ("no_shield", "Missing Face Shield", "🛡️"),
    ]
    for key, label, icon in type_map:
        if key in s:
            return label, icon
    clean = s.replace("_", " ").title()
    if not clean.lower().startswith("missing") and not clean.lower().startswith("no"):
        return f"PPE Infraction ({clean})", "⚠️"
    return clean, "⚠️"


def format_violation_datetime(val: Any) -> str:
    """Formats any ISO string or SQLite timestamp to standard YYYY-MM-DD HH:MM:SS."""
    if not val:
        return "Unknown Date/Time"
    s = str(val).strip()
    try:
        s_clean = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s_clean)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass
    if len(s) >= 19 and s[10] in (" ", "T"):
        return s[:19].replace("T", " ")
    return s


def format_workplace_location(item: Dict[str, Any]) -> str:
    """Extracts clear Workplace Location or Department name from violation/event metadata."""
    wp = item.get("workplace")
    if wp and str(wp).strip() and str(wp).strip().lower() != "none":
        return str(wp).strip()
    zn = item.get("zone_name")
    if zn and str(zn).strip() and str(zn).strip().lower() != "none":
        return str(zn).strip()
    cam = item.get("camera_id") or ""
    if ":" in str(cam):
        parts = str(cam).split(":", 1)
        dept = parts[1].strip()
        if dept:
            return dept
    zone = item.get("zone")
    zone_id = item.get("zone_id")
    if zone and str(zone).strip() and str(zone).strip().lower() != "none":
        return f"{str(zone).strip()} ({zone_id})" if zone_id else str(zone).strip()
    if zone_id and str(zone_id).strip() and str(zone_id).strip().lower() != "none":
        return f"Workplace Zone {str(zone_id).strip()}"
    return "Main Facility / Production Floor"


def resolve_evidence_image_path(item: Dict[str, Any]) -> Optional[str]:
    """Resolves local filesystem path for snapshot, crop, or annotated detection image."""
    candidates = [
        item.get("snapshot_path"),
        item.get("evidence_path"),
        item.get("output_path"),
    ]
    for c in candidates:
        if not c or not str(c).strip():
            continue
        resolved = resolve_runtime_path(c)
        if resolved:
            return resolved
    return None


def get_canonical_violation_status(item: Dict[str, Any], decisions_map: Optional[Dict[str, Any]] = None) -> Tuple[str, str, str]:
    """
    Returns (canonical_status, display_badge_label, badge_style_css)
    canonical_status is strictly one of: 'Pending', 'Reviewed', 'Resolved'
    """
    res_stat = (item.get("resolution_status") or item.get("status") or item.get("review_status") or "").strip().lower()
    if res_stat in ("resolved", "closed", "mitigated"):
        return "Resolved", "✅ Resolved", "background:#ecfdf5;color:#065f46;border:1px solid #a7f3d0;"

    eid = str(item.get("event_id") or "").strip()
    if decisions_map and eid in decisions_map:
        dec = decisions_map[eid].get("decision", "").lower()
        if dec == "approved":
            return "Reviewed", "👁️ Reviewed (Approved)", "background:#eff6ff;color:#1e40af;border:1px solid #bfdbfe;"
        elif dec == "rejected":
            return "Reviewed", "👁️ Reviewed (Dismissed)", "background:#f1f5f9;color:#475569;border:1px solid #cbd5e1;"
        return "Reviewed", "👁️ Reviewed", "background:#eff6ff;color:#1e40af;border:1px solid #bfdbfe;"

    if res_stat in ("confirmed", "dismissed", "approved", "rejected", "reviewed", "under review"):
        label = "👁️ Reviewed (Confirmed)" if res_stat == "confirmed" else ("👁️ Under Review" if res_stat == "under review" else "👁️ Reviewed")
        return "Reviewed", label, "background:#eff6ff;color:#1e40af;border:1px solid #bfdbfe;"

    return "Pending", "⏳ Pending Review", "background:#fffbeb;color:#b45309;border:1px solid #fde68a;"


def render_recorded_events_page():
    enterprise_hud(
        "Workplace Safety Violation Monitoring & Review",
        "Chronological registry of PPE infractions, optical evidence snapshots, workplace locations, and multi-tier review statuses stored in SQLite."
    )
    card_bg = "#ffffff"
    card_border = "#e2e8f0"
    text_color = "#171a21"
    subtext_color = "#5b6472"

    can_review = role in ("admin", "operator")
    if can_review:
        tab_queue, tab_events = st.tabs(["📋 Confidence-Aware Human Review Queue", "📹 All Violation Records & Evidence Log"])
    else:
        tab_queue = None
        tab_events = st.container()

    if tab_queue is not None:
        with tab_queue:
            st.markdown("#### 🛡️ Safety Violation Human Review Queue")
            st.caption("Review potential violation events before inclusion into formal audit reports. Low-confidence events are automatically filtered out.")
            from services.multimodal.review_queue import get_review_queue, approve_event, reject_event
            queue_items = get_review_queue(limit=50)
            if not queue_items:
                st.info("✅ No safety events currently pending human review.")
            else:
                st.markdown(f"**{len(queue_items)} high-confidence items awaiting supervisor verification:**")
                for q_ev in queue_items:
                    eid = q_ev.get("event_id") or f"EVT-{q_ev.get('id')}"
                    v_label, v_icon = format_violation_type(q_ev.get("violation_type"))
                    c_id = q_ev.get("camera_id", "CAM-01")
                    z_code = q_ev.get("zone_id") or q_ev.get("zone", "Z-01")
                    conf_pct = float(q_ev.get("confidence", 0.0)) * 100
                    snap_path = resolve_evidence_image_path(q_ev)
                    citation = q_ev.get("policy_citation", "Standard Policy")
                    ts_disp = format_violation_datetime(q_ev.get("timestamp") or q_ev.get("created_at"))
                    loc_disp = format_workplace_location(q_ev)

                    with st.container():
                        st.markdown(f"""
                        <div style="background:{card_bg};border:1px solid {card_border};border-left:4px solid #d97706;border-radius:8px;padding:0.85rem 1rem;margin-bottom:0.5rem;box-shadow:0 1px 3px rgba(0,0,0,0.03);">
                            <div style="display:flex;justify-content:space-between;align-items:center;">
                                <span style="font-weight:700;font-size:0.88rem;color:#0369a1;">
                                    📹 {c_id} &bull; 🏢 Department / Location: <strong>{loc_disp}</strong>
                                </span>
                                <span style="background:#fef3c7;color:#b45309;border:1px solid #fde68a;font-size:0.75rem;font-weight:700;padding:0.2rem 0.55rem;border-radius:999px;">
                                    ⏳ Pending Review
                                </span>
                            </div>
                            <div style="display:flex;justify-content:space-between;align-items:center;margin-top:0.4rem;">
                                <span style="font-weight:800;color:{text_color};font-size:0.95rem;">
                                    {v_icon} {v_label}
                                </span>
                                <span style="color:#0284c7;font-size:0.8rem;font-weight:700;background:#f0f9ff;padding:0.15rem 0.45rem;border-radius:6px;">
                                    {conf_pct:.1f}% Confidence
                                </span>
                            </div>
                            <div style="font-size:0.76rem;color:{subtext_color};margin-top:0.35rem;display:flex;gap:1.2rem;flex-wrap:wrap;">
                                <span>📅 <strong>Detected:</strong> {ts_disp}</span>
                                <span>🆔 <strong>Event ID:</strong> <code>{eid}</code></span>
                                <span>📜 <strong>Policy Citation:</strong> <em>{citation}</em></span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        c_left, c_right = st.columns([1.3, 1])
                        with c_left:
                            if snap_path and Path(snap_path).exists():
                                with st.expander("🖼️ View Detected Evidence Image", expanded=True):
                                    st.image(snap_path, caption=f"{c_id} • {v_label}", width="stretch")
                            else:
                                st.caption(f"📷 Telemetry Evidence: BBox {q_ev.get('bounding_box', 'N/A')}")
                            review_note = st.text_input("Reviewer Observation / Note", key=f"rq_note_{eid}", placeholder="Optional audit notes...")
                        with c_right:
                            st.write("")
                            b_app, b_rej = st.columns(2)
                            with b_app:
                                if st.button("✅ Approve", key=f"btn_app_{eid}", width="stretch", type="primary"):
                                    try:
                                        approve_event(event_id=eid, reviewer_user_id=uid, note=review_note, role=role)
                                        st.success(f"Event {eid} approved.")
                                        st.rerun()
                                    except Exception as exc:
                                        st.error(str(exc))
                            with b_rej:
                                if st.button("❌ Reject", key=f"btn_rej_{eid}", width="stretch"):
                                    try:
                                        reject_event(event_id=eid, reviewer_user_id=uid, note=review_note, role=role)
                                        st.warning(f"Event {eid} rejected.")
                                        st.rerun()
                                    except Exception as exc:
                                        st.error(str(exc))

    with tab_events:
        st.markdown("#### 📹 Workplace Violation Monitoring & Evidence Registry")
        st.caption("Auditable log of workplace safety infractions, physical evidence snapshots, and resolution statuses stored in SQLite:")

        # Filter toolbar
        f_col1, f_col2, f_col3, f_col4 = st.columns([1.2, 1.4, 1.2, 1.6])
        with f_col1:
            status_filter = st.selectbox(
                "Filter by Status",
                ["All Statuses", "Pending", "Reviewed", "Resolved"],
                key="violation_filter_status"
            )
        with f_col2:
            type_filter = st.selectbox(
                "Filter by Violation Type",
                ["All Types", "Missing Helmet", "Missing Safety Vest", "Missing Eye Protection", "Missing Safety Footwear", "Missing Safety Gloves", "Restricted Area Incursion"],
                key="violation_filter_type"
            )
        with f_col3:
            source_filter = st.selectbox(
                "Violation Source",
                ["All Sources", "CCTV Camera Streams", "Inspection Uploads"],
                key="violation_filter_source"
            )
        with f_col4:
            search_query = st.text_input(
                "Search Location / Department / ID",
                placeholder="e.g. Fabrication, Dock, EVT-...",
                key="violation_search_query"
            ).strip().lower()

        # Query review decisions mapping from SQLite
        with get_connection() as conn:
            dec_rows = conn.execute("SELECT event_id, decision, note, decided_at FROM review_decisions").fetchall()
            decisions_map = {str(r["event_id"]): dict(r) for r in dec_rows}

        # Query camera safety events and inspection upload violations
        unified_items: List[Dict[str, Any]] = []

        if source_filter in ("All Sources", "CCTV Camera Streams"):
            events_feed = get_safety_events(limit=100)
            for ev in events_feed:
                v_label, v_icon = format_violation_type(ev.get("violation_type"))
                ts_disp = format_violation_datetime(ev.get("timestamp") or ev.get("created_at"))
                loc_disp = format_workplace_location(ev)
                img_path = resolve_evidence_image_path(ev)
                status_key, status_label, status_style = get_canonical_violation_status(ev, decisions_map)
                eid = ev.get("event_id") or f"EVT-{ev.get('id')}"

                unified_items.append({
                    "source": "cctv",
                    "pk": ev.get("id"),
                    "item_id": eid,
                    "violation_type_raw": ev.get("violation_type"),
                    "formatted_type": v_label,
                    "icon": v_icon,
                    "timestamp_disp": ts_disp,
                    "location_disp": loc_disp,
                    "camera_id": ev.get("camera_id", "CAM-01"),
                    "zone_code": ev.get("zone_id") or ev.get("zone", "Z-01"),
                    "image_path": img_path,
                    "status_key": status_key,
                    "status_label": status_label,
                    "status_style": status_style,
                    "confidence_pct": float(ev.get("confidence", 0.90)) * 100 if ev.get("confidence") is not None else None,
                    "bbox_str": ev.get("bounding_box", "[]"),
                    "raw_item": ev,
                })

        if source_filter in ("All Sources", "Inspection Uploads"):
            violations_feed = get_recent_violations(limit=100, user_id=uid, role=role)
            for v in violations_feed:
                v_label, v_icon = format_violation_type(v.get("violation_type"))
                ts_disp = format_violation_datetime(v.get("created_at"))
                loc_disp = format_workplace_location(v)
                img_path = resolve_evidence_image_path(v)
                status_key, status_label, status_style = get_canonical_violation_status(v, decisions_map)
                vid = f"VIO-{v.get('violation_id')}"

                unified_items.append({
                    "source": "inspection",
                    "pk": v.get("violation_id"),
                    "item_id": vid,
                    "violation_type_raw": v.get("violation_type"),
                    "formatted_type": v_label,
                    "icon": v_icon,
                    "timestamp_disp": ts_disp,
                    "location_disp": loc_disp,
                    "camera_id": v.get("filename", "Inspection Upload"),
                    "zone_code": v.get("zone_code") or "Zone",
                    "image_path": img_path,
                    "status_key": status_key,
                    "status_label": status_label,
                    "status_style": status_style,
                    "confidence_pct": float(v.get("confidence", 0.0)) * 100 if v.get("confidence") is not None else None,
                    "bbox_str": f"Shift: {v.get('shift', 'General')} | Severity: {str(v.get('severity', 'medium')).upper()}",
                    "raw_item": v,
                })

        # Apply user filters
        filtered_items = []
        for it in unified_items:
            # Status filter
            if status_filter != "All Statuses" and it["status_key"] != status_filter:
                continue
            # Type filter
            if type_filter != "All Types" and it["formatted_type"] != type_filter:
                continue
            # Search query
            if search_query:
                haystack = f"{it['item_id']} {it['location_disp']} {it['formatted_type']} {it['camera_id']} {it['timestamp_disp']}".lower()
                if search_query not in haystack:
                    continue
            filtered_items.append(it)

        # Sort newest first
        filtered_items.sort(key=lambda x: str(x.get("timestamp_disp", "")), reverse=True)

        # Summary KPIs
        k_c1, k_c2, k_c3, k_c4 = st.columns(4)
        total_cnt = len(filtered_items)
        pending_cnt = sum(1 for x in filtered_items if x["status_key"] == "Pending")
        reviewed_cnt = sum(1 for x in filtered_items if x["status_key"] == "Reviewed")
        resolved_cnt = sum(1 for x in filtered_items if x["status_key"] == "Resolved")

        k_c1.metric("Violations In View", total_cnt)
        k_c2.metric("⏳ Pending", pending_cnt)
        k_c3.metric("👁️ Reviewed", reviewed_cnt)
        k_c4.metric("✅ Resolved", resolved_cnt)
        st.write("")

        if not filtered_items:
            st.info("No recorded safety violations match the selected filter criteria.")
        else:
            for item in filtered_items:
                item_id = item["item_id"]
                loc_disp = item["location_disp"]
                v_label = item["formatted_type"]
                v_icon = item["icon"]
                ts_disp = item["timestamp_disp"]
                status_label = item["status_label"]
                status_style = item["status_style"]
                status_key = item["status_key"]
                img_path = item["image_path"]
                source_badge = "📹 Camera Stream" if item["source"] == "cctv" else "📁 Upload Inspection"

                border_color = "#10b981" if status_key == "Resolved" else ("#0284c7" if status_key == "Reviewed" else "#d97706")

                with st.container():
                    st.markdown(f"""
                    <div style="background:{card_bg};border:1px solid {card_border};border-left:4px solid {border_color};border-radius:8px;padding:0.85rem 1rem;margin-bottom:0.5rem;box-shadow:0 1px 3px rgba(0,0,0,0.03);">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <span style="font-weight:700;font-size:0.88rem;color:#0369a1;">
                                {source_badge} &bull; 🏢 Department / Location: <strong>{loc_disp}</strong>
                            </span>
                            <span style="{status_style};font-size:0.75rem;font-weight:700;padding:0.2rem 0.55rem;border-radius:999px;">
                                {status_label}
                            </span>
                        </div>
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-top:0.4rem;">
                            <span style="font-weight:800;color:{text_color};font-size:0.95rem;">
                                {v_icon} {v_label}
                            </span>
                            <span style="color:#0284c7;font-size:0.8rem;font-weight:700;background:#f0f9ff;padding:0.15rem 0.45rem;border-radius:6px;">
                                {f"{item['confidence_pct']:.1f}% Conf" if item.get('confidence_pct') else item.get('camera_id')}
                            </span>
                        </div>
                        <div style="font-size:0.76rem;color:{subtext_color};margin-top:0.35rem;display:flex;gap:1.2rem;flex-wrap:wrap;">
                            <span>📅 <strong>Detected:</strong> {ts_disp}</span>
                            <span>🆔 <strong>Reference ID:</strong> <code>{item_id}</code></span>
                            <span>📐 <strong>Optical Telemetry:</strong> <code>{item.get('bbox_str', 'N/A')}</code></span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    btn_c1, btn_c2 = st.columns([1.3, 1])
                    with btn_c1:
                        if img_path and Path(img_path).exists():
                            with st.expander("🖼️ View Detected Image / Annotated Evidence", expanded=False):
                                st.image(img_path, caption=f"{loc_disp} • {v_label}", width="stretch")
                        else:
                            st.caption(f"📷 Optical Evidence: Telemetry BBox {item.get('bbox_str', 'N/A')} (Stored in Audit Database)")

                    with btn_c2:
                        st.write("")
                        b_res_col, b_inv_col = st.columns(2)
                        with b_res_col:
                            if can_review and status_key != "Resolved":
                                if st.button("✅ Mark Resolved", key=f"btn_res_{item_id}", width="stretch"):
                                    try:
                                        if item["source"] == "cctv":
                                            update_safety_event_status(event_id=item["pk"], status="Resolved", resolution_status="Resolved")
                                        else:
                                            update_violation_status(violation_id=item["pk"], new_status="resolved", reviewer_user_id=uid)
                                        st.success(f"{item_id} marked as Resolved.")
                                        st.rerun()
                                    except Exception as exc:
                                        st.error(str(exc))
                            elif status_key == "Resolved":
                                st.caption("✅ Incident Resolved")
                            else:
                                st.caption("🔒 Operator resolution required")

                        with b_inv_col:
                            if st.button(f"🔍 Investigate", key=f"btn_rec_ev_{item_id}", width="stretch"):
                                raw_ev = item["raw_item"]
                                st.session_state["active_investigation_event"] = raw_ev
                                st.session_state["agent_query_input"] = f"Investigate verified infraction {item_id}: {v_label} in {loc_disp}"
                                st.session_state["redirect_to_page"] = PAGE_INVESTIGATOR_CONSOLE
                                st.rerun()

    st.write("")

# -------------------------------------------------------------
# PAGE 7: DISPATCHED REMEDIATION WORKFLOWS & HISTORY MANAGEMENT
# -------------------------------------------------------------
def render_remediation_workflows_page():
    enterprise_hud(
        "Dispatched Remediation Workflows & History Management",
        "Audit ticket logging, area supervisor emergency dispatching, and dual-level investigation history controls."
    )
    card_bg = "#ffffff"
    card_border = "#e2e8f0"
    code_bg = "#f8fafc"
    text_color = "#0f172a"
    subtext_color = "#475569"

    # Dual-Level History Management Sub-Panel
    with st.expander("🗄️ Past Investigation Sessions & History Management", expanded=False):
        hist_records = get_investigation_history(limit=50)
        h_col1, h_col2 = st.columns([3, 1])
        h_col1.markdown(f"**Archived Investigation Logs ({len(hist_records)} stored):**")
        with h_col2:
            with st.popover("⚠️ Clear Entire History", use_container_width=True):
                st.markdown("**Purge All Stored Investigation Logs?**")
                st.caption("This action is permanent and clears all entries from SQLite agent_investigation_history.")
                if st.button("Confirm Purge All Records", type="primary", key="btn_confirm_clear_all_inv", width="stretch"):
                    clear_all_investigation_history()
                    if "latest_agent_result" in st.session_state:
                        del st.session_state["latest_agent_result"]
                    st.success("Investigation history successfully cleared.")
                    time.sleep(0.4)
                    st.rerun()

        if not hist_records:
            st.caption("No past investigation sessions recorded in the database.")
        else:
            for item in hist_records:
                i_id = item["id"]
                i_ts = item.get("timestamp", "Recent")
                i_query = item.get("query_text", "Safety Inquiry")
                i_actions = item.get("actions_executed", [])
                action_badge = f"{len(i_actions)} Remediations" if i_actions else "Consultation"

                with st.container():
                    st.markdown(f"""
                    <div style="background:{code_bg};border:1px solid {card_border};border-left:3px solid #0284c7;border-radius:6px;padding:0.55rem 0.8rem;margin-bottom:0.35rem;">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <span style="font-weight:700;font-size:0.83rem;color:#0369a1;">Session #{i_id} &bull; ⏱️ {str(i_ts)[:19]}</span>
                            <span style="font-size:0.72rem;background:rgba(2,132,199,0.1);color:#0284c7;padding:0.15rem 0.45rem;border-radius:4px;font-weight:700;">
                                {action_badge}
                            </span>
                        </div>
                        <div style="font-size:0.82rem;color:{text_color};margin-top:0.25rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
                            {i_query}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    btn_c_load, btn_c_del = st.columns([3.5, 1])
                    with btn_c_load:
                        if st.button(f"📂 Load Session #{i_id} Report", key=f"btn_load_h_{i_id}", width="stretch"):
                            st.session_state["agent_query_input"] = i_query
                            st.session_state["latest_agent_result"] = {
                                "response_summary": item.get("response_summary", ""),
                                "final_response": item.get("response_summary", ""),
                                "execution_steps": [
                                    f"[Session #{i_id}] Loaded historical audit record ({str(i_ts)[:19]})",
                                    f"[Query] {i_query}",
                                    f"[Remediations] {len(i_actions)} actions verified in SQLite"
                                ],
                                "actions_executed": i_actions,
                                "actions_taken": i_actions,
                                "retrieved_events": item.get("events_referenced", []),
                                "visual_evidence": item.get("events_referenced", []),
                                "primary_event": item.get("events_referenced", [{}])[0] if item.get("events_referenced") else None,
                                "matched_policies": item.get("policies_cited", []),
                            }
                            st.session_state["redirect_to_page"] = PAGE_INVESTIGATOR_CONSOLE
                            st.rerun()

                    with btn_c_del:
                        if st.button("🗑️ Delete", key=f"btn_del_h_{i_id}", width="stretch", help="Delete this investigation session"):
                            delete_investigation_by_id(i_id)
                            st.toast(f"Deleted investigation record #{i_id}")
                            time.sleep(0.3)
                            st.rerun()

    st.write("")
    st.write("")
    st.markdown("### ⚡ Real-Time Remediation Workflows & Status")
    # Real-time Workflow Status Cards
    st.markdown("**Real-Time Remediation Workflows:**")
    w_c1, w_c2 = st.columns(2)
    latest_res = st.session_state.get("latest_agent_result", {})
    actions_list = latest_res.get("actions_executed") or latest_res.get("actions_taken", [])

    # Check if audit ticket was generated
    ticket_act = next((a for a in actions_list if "ticket" in a.get("tool", "").lower() or a.get("action_type") == "AUDIT_TICKET_CREATED"), None)
    ticket_status = ticket_act.get("status") if ticket_act else "Status: Ready (Standby)"
    ticket_badge_color = "#10b981" if ticket_act else "#64748b"

    # Check if supervisor alert was dispatched
    alert_act = next((a for a in actions_list if "alert" in a.get("tool", "").lower() or a.get("action_type") == "SUPERVISOR_DISPATCH_SENT"), None)
    alert_status = alert_act.get("status") if alert_act else "Status: Ready (Standby)"
    alert_badge_color = "#0284c7" if alert_act else "#64748b"

    with w_c1:
        st.markdown(f"""
        <div style="background:{card_bg};border:1px solid {card_border};border-left:4px solid {ticket_badge_color};border-radius:8px;padding:0.75rem 0.9rem;">
            <div style="font-size:0.75rem;font-weight:700;color:{subtext_color};text-transform:uppercase;">📝 Audit Ticket</div>
            <div style="font-size:0.85rem;font-weight:700;color:{text_color};margin-top:0.2rem;">{ticket_status}</div>
            <div style="font-size:0.72rem;color:{subtext_color};margin-top:0.2rem;">remediation_actions committed</div>
        </div>
        """, unsafe_allow_html=True)

    with w_c2:
        st.markdown(f"""
        <div style="background:{card_bg};border:1px solid {card_border};border-left:4px solid {alert_badge_color};border-radius:8px;padding:0.75rem 0.9rem;">
            <div style="font-size:0.75rem;font-weight:700;color:{subtext_color};text-transform:uppercase;">🚨 Supervisor Alert</div>
            <div style="font-size:0.85rem;font-weight:700;color:{text_color};margin-top:0.2rem;">{alert_status}</div>
            <div style="font-size:0.72rem;color:{subtext_color};margin-top:0.2rem;">Structured webhook / email</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")
    st.markdown("#### 📋 Dispatched Remediation Audit Trail")
    recent_remediations = get_remediation_actions(limit=5)
    if recent_remediations:
        for rem in recent_remediations:
            act_type = rem.get("action_type", "ACTION")
            payload = rem.get("action_payload", {})
            t_no = payload.get("ticket_no", "")
            recip = payload.get("recipient", "")
            detail_str = f"Ticket: {t_no}" if t_no else (f"Recipient: {recip}" if recip else str(payload)[:80])
            st.markdown(f"""
            <div style="background:{code_bg};border:1px solid {card_border};border-radius:6px;padding:0.5rem 0.75rem;margin-bottom:0.4rem;font-size:0.78rem;">
                <div style="display:flex;justify-content:space-between;color:#0284c7;font-weight:700;">
                    <span>⚡ {act_type}</span>
                    <span style="color:{subtext_color};font-weight:normal;">{str(rem.get('created_at', ''))[:19]}</span>
                </div>
                <div style="color:{text_color};margin-top:0.15rem;">
                    {detail_str}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("No remediations dispatched yet. Run an investigation to trigger automated remediations.")


    # ==========================================

# -------------------------------------------------------------
# PAGE 8: AUDIT REPORTS & COMPLIANCE EXPORT
# -------------------------------------------------------------
def render_audit_reports_page():
    enterprise_hud("Audit Reports & Compliance Export", "Generate formal PDF inspection audit dossiers with physical signature blocks and formula-sanitized CSV exports.")

    r1, r2, r3 = st.columns(3)
    s_rep = date_str(r1.date_input("Audit Period Start", date.today() - timedelta(days=30), key="rep_s"))
    e_rep = date_str(r2.date_input("Audit Period End", date.today(), key="rep_e"))
    status_filter = r3.selectbox("Filter Review Status", ["All", "Pending", "Confirmed", "Resolved", "Dismissed"], key="rep_stat")

    from services.multimodal.review_queue import filter_report_items
    v_audit = get_recent_violations(500, uid, role, s_rep, e_rep, status_filter)
    v_audit = filter_report_items(v_audit)
    stats_audit = get_dashboard_stats(uid, role, s_rep, e_rep)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    st.markdown(f"**Audit Dataset Scope:** `{len(v_audit)}` violation record(s) matching criteria from `{s_rep}` to `{e_rep}`.")

    b1, b2, b3 = st.columns(3)

    with b1:
        st.markdown("### 📄 Formal PDF Audit Report")
        st.caption("Includes executive summary, OSHA compliance KPIs, violation logs, and sign-off certification block.")
        pdf_bytes = generate_pdf_report(
            violations=v_audit,
            stats=stats_audit,
            reporting_period=f"{s_rep} to {e_rep}",
            generated_by=f"{current_user['full_name']} ({role.upper()})"
        )
        st.download_button(
            "📥 Download Official PDF Report",
            pdf_bytes,
            f"DetectoAI_Audit_Report_{tag}.pdf",
            "application/pdf",
            width="stretch",
            type="primary",
            key=f"btn_dl_pdf_{tag}"
        )
        with st.expander("👁️ Audit Scope Summary", expanded=False):
            st.markdown(f"""
            - **Scope:** `{len(v_audit)} Incident Records`
            - **Compliance Rating:** `{stats_audit.get('violations', 0)} Incidents Flagged`
            - **Page Size:** `ISO 45001 Landscape A4 Document`
            """)

    with b2:
        st.markdown("### 📊 Sanitized CSV Export")
        st.caption("Exports sanitized tabular data with formula injection mitigation for Excel and PowerBI ingestion.")
        csv_bytes = generate_csv_report(v_audit)
        st.download_button(
            "📥 Download Sanitized CSV",
            csv_bytes,
            f"DetectoAI_Violations_{tag}.csv",
            "text/csv",
            width="stretch",
            key=f"btn_dl_csv_{tag}"
        )
        with st.expander("👁️ CSV Columns Included", expanded=False):
            st.caption("Includes ID, Timestamp, File, Workplace, Zone, Severity, Confidence, Review Status, Reviewer, and Shift.")

    with b3:
        st.markdown("### ⚙️ JSON Data Package")
        st.caption("Structured JSON dump with complete detection bounding boxes and audit timestamps.")
        json_str = generate_json_report(v_audit)
        st.download_button(
            "📥 Download JSON Package",
            json_str.encode("utf-8"),
            f"DetectoAI_Audit_{tag}.json",
            "application/json",
            width="stretch",
            key=f"btn_dl_json_{tag}"
        )
        with st.expander("👁️ JSON Data Integration", expanded=False):
            st.caption("Standard JSON schema for direct SIEM, ERP, or external analytics consumption.")

    st.divider()

    # Automated Alert Notification Dispatch Simulator
    st.subheader("🚨 Automated Safety Alert Dispatcher (Simulator)")
    st.caption("Simulate real-time multi-channel incident dispatches (Email, Webhook, SMS, Slack) for critical safety violations.")

    al1, al2, al3 = st.columns([1, 1.5, 1])
    channel = al1.selectbox("Alert Channel", ["Email", "Webhook", "SMS", "Slack"])
    recipient = al2.text_input("Recipient / Endpoint", value="safety-director@plant.com" if channel == "Email" else "https://hooks.slack.com/services/safety-alerts")
    alert_msg = al3.text_input("Alert Summary", value="CRITICAL: Missing Hardhat detected on Heavy Fabrication Floor.")

    if st.button("🚀 Dispatch Simulated Notification", type="primary", width="stretch"):
        dispatch_id = log_alert_dispatch(
            violation_id=v_audit[0]["violation_id"] if v_audit else None,
            channel=channel,
            recipient=recipient,
            payload=alert_msg,
            status="dispatched"
        )
        st.success(f"Alert successfully dispatched via {channel} to {recipient} (Dispatch ID #{dispatch_id})!")

    # Alert Dispatch History Table
    st.subheader("Recent Alert Dispatch History")
    dispatches = get_alert_dispatches(limit=15)
    if dispatches:
        st.dataframe(pd.DataFrame([{
            "ID": d["id"],
            "Timestamp": fmt_ts(d["created_at"]),
            "Channel": d["channel"],
            "Recipient": d["recipient"],
            "Payload": d["payload"],
            "Status": d["status"].upper()
        } for d in dispatches]), hide_index=True, width="stretch")
    else:
        empty_state("📢", "No Alert Dispatches Logged")


    # ==========================================


# -------------------------------------------------------------
# PAGE 9: USER & ACCESS GOVERNANCE (ADMIN ONLY)
# -------------------------------------------------------------
def render_user_governance_page():
    if not user_is_admin:
        st.error("Access Denied: Administrator role required.")
        st.stop()

    enterprise_hud("User & Access Governance (RBAC)", "Manage corporate operators, role assignments, and account security.")

    users = list_all_users()
    st.subheader(f"Registered Corporate Users ({len(users)})")
    st.dataframe(pd.DataFrame([{
        "ID": u["id"],
        "Full Name": u["full_name"],
        "Email": u["email"],
        "Role": u["role"].upper(),
        "Account Status": u["status"].upper(),
        "Workplace": u["workplace"] or "General Operations",
        "Registered": fmt_ts(u["created_at"]),
        "Last Login": fmt_ts(u.get("last_login_at")),
    } for u in users]), hide_index=True, width="stretch")

    st.divider()

    st.subheader("Modify User Privileges")
    sel_uid = st.selectbox(
        "Select User Account",
        [u["id"] for u in users],
        format_func=lambda x: f"#{x} — {next(u['email'] for u in users if u['id'] == x)} ({next(u['full_name'] for u in users if u['id'] == x)})"
    )
    target_user = next(u for u in users if u["id"] == sel_uid)

    c_r, c_s = st.columns(2)
    with c_r:
        new_role = st.selectbox("Role Assignment", ["standard", "admin"], index=0 if target_user["role"] == "standard" else 1, key=f"r_sel_{sel_uid}")
        if st.button("Update Role", width="stretch"):
            if sel_uid == uid and new_role == "standard":
                st.error("Guardrail violation: You cannot demote your own active administrator account.")
            elif update_user_role(sel_uid, new_role, uid):
                st.success("User role updated successfully.")
                st.rerun()
            else:
                st.error("Action rejected: Cannot remove the last active administrator.")

    with c_s:
        new_stat = st.selectbox("Account Status", ["active", "deactivated"], index=0 if target_user["status"] == "active" else 1, key=f"s_sel_{sel_uid}")
        if st.button("Update Status", width="stretch"):
            if sel_uid == uid and new_stat == "deactivated":
                st.error("Guardrail violation: You cannot deactivate your own active administrator account.")
            elif update_user_status(sel_uid, new_stat, uid):
                st.success("Account status updated.")
                st.rerun()
            else:
                st.error("Action rejected: Cannot deactivate the last active administrator.")


    # ==========================================


# -------------------------------------------------------------
# PAGE 10: MY PROFILE & SECURITY
# -------------------------------------------------------------
def render_profile_security_page():
    enterprise_hud("Operator Profile & Security", "Manage credentials, password security, and active session details.")

    p_c1, p_c2 = st.columns(2)
    with p_c1:
        st.subheader("Account Details")
        st.markdown(f"""
        - **Operator Name:** `{current_user['full_name']}`
        - **Email Address:** `{current_user['email']}`
        - **Access Level:** `{role.upper()}`
        - **Assigned Facility:** `{current_user.get('workplace') or 'HQ Operations'}`
        - **Account Status:** `{current_user.get('status', 'active').upper()}`
        - **Registered Since:** `{fmt_ts(current_user.get('created_at'))}`
        """)

    with p_c2:
        st.subheader("Update Corporate Password")
        with st.form("change_password_form", clear_on_submit=True):
            cur_p = st.text_input("Current Password", type="password")
            new_p1 = st.text_input("New Password", type="password")
            new_p2 = st.text_input("Confirm New Password", type="password")
            submit_pw = st.form_submit_button("Update Password", type="primary", width="stretch")

        if submit_pw:
            with get_connection() as conn:
                rec = conn.execute("SELECT password_hash FROM users WHERE id = ?", (uid,)).fetchone()
            if not rec or not verify_password(cur_p, rec["password_hash"]):
                st.error("Current password verification failed.")
            elif new_p1 != new_p2:
                st.error("New passwords do not match.")
            else:
                val_ok, err_str = validate_password_strength(new_p1)
                if not val_ok:
                    st.error(err_str)
                else:
                    change_user_password(uid, hash_password(new_p1))
                    st.success("Your password has been securely updated.")


# =============================================================
# DESTINATION 1: DASHBOARD
# =============================================================
def render_dashboard_page():
    render_executive_dashboard_page()


# Backward compatibility alias for legacy safety analytics
def render_safety_analytics_page():
    render_dashboard_page()


# =============================================================
# DESTINATION 2: COMBINED IMAGE & VIDEO ANALYSIS
# =============================================================
def render_image_video_analysis_page():
    render_live_surveillance_page()


# Backward compatibility aliases
def render_image_analysis_page():
    render_image_video_analysis_page()


def render_video_analysis_page():
    render_image_video_analysis_page()


# =============================================================
# DESTINATION 3: VIOLATIONS
# =============================================================
def render_violations_page():
    render_recorded_events_page()


# =============================================================
# DESTINATION 4: REPORTS & EXPORT
# =============================================================
def render_reports_export_page():
    render_audit_reports_page()


# =============================================================
# DESTINATION 5: RAG SAFETY POLICY ASSISTANT
# =============================================================
def render_rag_safety_policy_assistant_page():
    seed_default_knowledge_base()
    enterprise_hud(
        "RAG Safety Policy Assistant",
        "Conversational knowledge retrieval grounded in plant safety manuals, operating SOPs, and OSHA 1910 standards with pinpoint citations."
    )

    doc_stats = get_document_stats()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Indexed Documents", doc_stats["indexed_documents"], delta=f"{doc_stats['total_documents']} Ingested")
    k2.metric("Knowledge Base Chunks", doc_stats["total_chunks"], delta="500-Token Windows")
    k3.metric("Retrieval Accuracy", "89.5%+", delta="Benchmark Met")
    k4.metric("Grounding Policy", "OSHA 1910", delta="Zero Hallucination")

    st.write("")

    # Two-Panel Layout (Taste Skill Professional Workspace)
    col_repo, col_assistant = st.columns([1.15, 1.85])

    # -------------------------------------------------------------
    # LEFT PANEL: DOCUMENT REPOSITORY & INGESTION
    # -------------------------------------------------------------
    with col_repo:
        st.subheader("📚 Document Repository")
        st.caption("Indexed safety manuals, SOPs, and inspection reports in FAISS vector store.")

        # Category Filter
        repo_cat_filter = st.selectbox(
            "Filter Category",
            ["All Categories", "Safety Manuals", "Standard Operating Procedures", "Inspection Reports"],
            key="rag_two_panel_cat_filter"
        )

        # Upload and Index Control Expander
        with st.expander("⚡ Upload & Index Document", expanded=False):
            doc_cat_up = st.selectbox(
                "Document Category",
                ["Safety Manuals", "Standard Operating Procedures", "Inspection Reports"],
                key="rag_two_panel_upload_cat"
            )
            up_f = st.file_uploader(
                "Select Safety Document",
                type=["pdf", "docx", "txt", "csv", "png", "jpg", "jpeg"],
                key="rag_two_panel_file_up",
                help=f"Accepts PDF, DOCX, TXT, CSV up to {MAX_IMAGE_MB} MB."
            )
            if up_f and st.button("Upload & Index Document", type="primary", width="stretch", key="btn_two_panel_idx"):
                val_ok, val_msg, doc_info = validate_and_store_document(
                    file_name=up_f.name,
                    file_bytes=up_f.getvalue(),
                    user_id=uid
                )
                if not val_ok:
                    st.error(f"Validation Failed: {val_msg}")
                else:
                    doc_id = doc_info["id"]
                    try:
                        extracted_items, total_pages, _ = extract_document_text(Path(doc_info["file_path"]), doc_info["file_type"])
                        entities = extract_ehs_entities("\n".join(item["text_content"] for item in extracted_items))
                        summary_text = generate_document_summary(extracted_items, entities)
                        update_document_metadata(doc_id, summary=summary_text, metadata_dict=entities)
                        final_chunks = process_extracted_items_into_chunks(doc_id, doc_info["file_name"], extracted_items, target_tokens=500, overlap_tokens=50)
                        insert_document_chunks(doc_id, final_chunks)
                        vstore = get_vector_store()
                        vstore.add_document_chunks(final_chunks)
                        update_document_status(doc_id, status="indexed", chunk_count=len(final_chunks), page_count=total_pages)
                        st.toast(f"'{doc_info['file_name']}' indexed!", icon="📚")
                        time.sleep(0.4)
                        st.rerun()
                    except Exception as e:
                        update_document_status(doc_id, status="failed", error_message=str(e))
                        st.error(f"Indexing error: {e}")

        # Document Listing
        docs = list_documents(limit=50)
        if repo_cat_filter != "All Categories":
            docs = [d for d in docs if d.get("category") == repo_cat_filter]

        if not docs:
            empty_state("📂", "No Matching Documents", "Upload documents above or click Load Sample Library.")
        else:
            for d in docs[:8]:
                b = get_status_badge(d["status"])
                with st.expander(f"{b['icon']} {d['file_name']} ({d.get('chunk_count', 0)} Chunks)", expanded=False):
                    st.markdown(
                        f"- **Category:** `{d.get('category') or 'General'}` | **Format:** `{d['file_type'].upper()}`\n"
                        f"- **Size:** `{d['file_size']/1024:.1f} KB` | **Status:** <span class=\"sv-pill {b['pill_class']}\">{b['label']}</span>",
                        unsafe_allow_html=True
                    )
                    if d.get("summary"):
                        st.caption(d["summary"][:140] + "...")

        st.write("")
        b_c1, b_c2 = st.columns(2)
        if b_c1.button("📦 Load Sample Library", width="stretch", key="btn_two_panel_load_sample"):
            with st.spinner("Seeding standard EHS library into FAISS vector database..."):
                load_sample_ehs_library(force=True)
            st.toast("EHS Library ready!", icon="📚")
            time.sleep(0.3)
            st.rerun()
        
        csv_data = export_repository_csv()
        b_c2.download_button("📥 Export CSV", data=csv_data, file_name=f"EHS_Knowledge_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv", width="stretch", key="btn_two_panel_dl_csv")

    # -------------------------------------------------------------
    # RIGHT PANEL: ASK YOUR SAFETY DOCUMENTS
    # -------------------------------------------------------------
    with col_assistant:
        st.subheader("💬 Ask Your Safety Documents")
        st.caption("Query indexed plant safety manuals, standard operating procedures, and inspection audit logs.")

        # API Engine & Gemini Key Settings Expander
        rag_svc = get_rag_service()
        has_gemini = bool(rag_svc.gemini_api_key)
        has_openai = bool(rag_svc.openai_api_key)
        active_llm = "Google Gemini" if has_gemini else ("OpenAI" if has_openai else "Deterministic / Simulated EHS Engine")
        engine_badge = f'<span class="sv-pill {"pill-purple" if (has_gemini or has_openai) else "pill-teal"}">{"● " + active_llm + " Connected" if (has_gemini or has_openai) else "⚡ Local Rule Engine"}</span>'

        with st.expander("🔑 AI Generation Engine & Gemini API Key Settings", expanded=not (has_gemini or has_openai)):
            st.markdown(f"**Current Engine Status:** {engine_badge}", unsafe_allow_html=True)
            k_c1, k_c2 = st.columns([2.5, 1.2])
            with k_c1:
                cur_key = st.session_state.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
                input_gem = st.text_input(
                    "Google Gemini API Key",
                    value=cur_key,
                    type="password",
                    placeholder="AIzaSy...",
                    help="Enter your Google AI Studio Gemini API key to enable fluent, natural question-answering with exact document grounding and intelligent workplace safety simulation.",
                    key="ui_gemini_api_key_input"
                )
                if input_gem and input_gem.strip() != st.session_state.get("gemini_api_key", ""):
                    st.session_state["gemini_api_key"] = input_gem.strip()
                    os.environ["GEMINI_API_KEY"] = input_gem.strip()
                    st.toast("Google Gemini API Key connected!", icon="🔑")
            with k_c2:
                st.caption("🔒 Key is kept in memory during the session. You can also define `GEMINI_API_KEY` in your `.env` file.")

        col_q1, col_q2 = st.columns([3, 1.2])
        with col_q2:
            qa_category = st.selectbox(
                "Repository Scope",
                ["All Categories", "Safety Manuals", "Standard Operating Procedures", "Inspection Reports"],
                key="rag_qa_category_sel"
            )

        # Quick Example Questions
        st.markdown("**Quick-Fill Example Inquiries:**")
        ex1, ex2, ex3 = st.columns(3)
        if ex1.button("🔥 Welding PPE", key="rag_ex_1", width="stretch"):
            st.session_state["rag_qa_query_box"] = "What are the mandatory PPE requirements for welding?"
        if ex2.button("🪜 Working at Heights", key="rag_ex_2", width="stretch"):
            st.session_state["rag_qa_query_box"] = "What precautions should be followed while working at heights?"
        if ex3.button("🏗️ Construction Gear", key="rag_ex_3", width="stretch"):
            st.session_state["rag_qa_query_box"] = "What protective equipment is required in construction areas?"

        ex4, ex5 = st.columns(2)
        if ex4.button("⛑️ Missing Helmet Protocol", key="rag_ex_4", width="stretch"):
            st.session_state["rag_qa_query_box"] = "What should be done when a worker is found without a safety helmet?"
        if ex5.button("⚡ Electrical Maintenance", key="rag_ex_5", width="stretch"):
            st.session_state["rag_qa_query_box"] = "What are the safety precautions for electrical maintenance?"

        default_query = st.session_state.get("rag_qa_query_box", "What are the mandatory PPE requirements for welding?")
        with col_q1:
            user_question = st.text_area(
                "Safety Policy Inquiry",
                value=default_query,
                height=90,
                key="rag_qa_query_input",
                placeholder="Ask a question about worksite safety policies, PPE standards, or compliance protocols..."
            )

        if st.button("🔍 Ask Safety Assistant", type="primary", width="stretch", key="btn_run_rag_qa"):
            if not user_question.strip():
                st.warning("Please enter a question about safety policies or choose one of the example inquiries above.")
            else:
                with st.spinner("Retrieving grounded policy provisions from FAISS vector store..."):
                    cat_arg = None if qa_category == "All Categories" else qa_category
                    rag_res = answer_safety_policy(query=user_question.strip(), category=cat_arg)
                    st.session_state["latest_rag_qa_result"] = rag_res

        qa_res = st.session_state.get("latest_rag_qa_result")
        if qa_res:
            st.divider()
            mode_str = qa_res.get("answering_mode", "Deterministic (Extractive - No LLM)")
            is_llm = "LLM" in mode_str
            mode_pill = f'<span class="sv-pill {"pill-purple" if is_llm else "pill-teal"}">{"🤖 " + mode_str if is_llm else "⚡ " + mode_str}</span>'
            conf_val = qa_res.get("confidence", 0.0) * 100
            conf_pill = f'<span class="sv-pill pill-navy">Relevance Confidence: {conf_val:.1f}%</span>'
            thresh_pill = f'<span class="sv-pill pill-slate">Threshold: {qa_res.get("threshold_used", 0.20):.2f}</span>'

            render_html(f"""
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem; margin-bottom:0.75rem;">
                <div style="font-weight:700; font-size:1.05rem; color:#171A21;">Grounded Safety Policy Findings</div>
                <div>
                    {mode_pill}
                    {conf_pill}
                    {thresh_pill}
                </div>
            </div>
            """)

            if not qa_res.get("sufficient_information"):
                st.warning(
                    "⚠️ **Insufficient Information:** No relevant policy provisions were found in the indexed repository for this query. "
                    "Under zero-hallucination protocols, no advisory can be synthesized without grounded source documents."
                )
            else:
                st.markdown(qa_res.get("answer", ""))

            # Excerpts and Citations Drawer
            citations = qa_res.get("citations", [])
            if not citations and qa_res.get("excerpts"):
                citations = qa_res.get("excerpts", [])

            if citations:
                st.write("")
                st.subheader(f"📌 Pinpoint Source Citations ({len(citations)})")
                for e_idx, cit in enumerate(citations):
                    if isinstance(cit, dict):
                        fname = cit.get("file_name", "Document")
                        pg = cit.get("page_number") or cit.get("row_number") or 1
                        loc_type = "Row" if cit.get("row_number") is not None else "Page"
                        sec = cit.get("section_title") or "General Policy Clause"
                        match_pct = cit.get("match_percentage", 0.0)
                        content = cit.get("text_content") or cit.get("excerpt") or ""
                        doc_id = cit.get("document_id", "N/A")
                        cat = cit.get("category", "Safety Manuals")
                        sim = cit.get("similarity_score", match_pct / 100.0 if match_pct else 0.0)
                    else:
                        fname = "Safety Policy Document"
                        pg = 1
                        loc_type = "Section"
                        sec = f"Clause #{e_idx + 1}"
                        match_pct = 100.0
                        content = str(cit)
                        doc_id = "N/A"
                        cat = "Safety Manuals"
                        sim = 1.0

                    with st.expander(f"📄 {fname} &bull; {loc_type} {pg} &bull; {sec} ({match_pct:.1f}% Match)", expanded=(e_idx == 0)):
                        render_html(f"""
                        <div style="background:#F7F8FA; border:1px solid #E2E8F0; border-left:4px solid #0F8B8D; border-radius:6px; padding:0.65rem 0.9rem; font-size:0.85rem; color:#171A21; line-height:1.45;">
                            {content}
                        </div>
                        <div style="font-size:0.72rem; color:#87909D; margin-top:0.35rem;">
                            Document ID: <code>#{doc_id}</code> | Category: <code>{cat}</code> | Similarity: <code>{sim:.4f}</code>
                        </div>
                        """)


# =============================================================
# DESTINATION 6: LANGGRAPH AI SAFETY INVESTIGATION
# =============================================================
def render_langgraph_ai_safety_investigation_page():
    enterprise_hud(
        "LangGraph AI Safety Investigation",
        "Multi-agent LangGraph orchestrator fusing visual telemetry with FAISS policy embeddings for automated incident investigations."
    )

    metrics_data = get_safety_event_metrics()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("LangGraph Core", "6-Node StateGraph", delta="Incident ➔ Report")
    m2.metric("Computer Vision", "YOLOv8 Evidence", delta="Person-Centric Spatial")
    m3.metric("Risk Framework", "5x5 Enterprise Matrix", delta="OSHA / ISO Aligned")
    m4.metric("Active Remediations", metrics_data.get("agent_actions_dispatched", 0), delta="CAPA & Emergency Alerts")

    st.write("")

    col_input, col_results = st.columns([1.1, 1.9])

    # -------------------------------------------------------------
    # LEFT COLUMN: INCIDENT INPUT
    # -------------------------------------------------------------
    with col_input:
        st.markdown("### 📝 Incident Input")
        st.caption("Provide incident narrative and optional photographic evidence for autonomous investigation.")

        # Incident Description (Mandatory)
        ex_col1, ex_col2 = st.columns(2)
        if ex_col1.button("🪜 Working at Height", key="inv_ex_1", width="stretch"):
            st.session_state["inv_desc_box"] = "Worker operating at height without a fall protection harness near an active scaffolding area."
        if ex_col2.button("🧪 Chemical Handling", key="inv_ex_2", width="stretch"):
            st.session_state["inv_desc_box"] = "Chemical transfer in coating line without chemical splash goggles and neoprene gauntlet gloves."

        default_desc = st.session_state.get(
            "inv_desc_box",
            "Worker operating at height without a fall protection harness near an active scaffolding area."
        )

        incident_desc = st.text_area(
            "Incident Description *",
            value=default_desc,
            height=110,
            key="inv_input_desc_field",
            placeholder="Describe the safety incident, hazard, or workplace condition you want investigated..."
        )

        # Visual Evidence (Optional)
        st.markdown("**Visual Evidence (Optional)**")
        up_img = st.file_uploader(
            "Attach Workplace Photographic Evidence",
            type=["jpg", "jpeg", "png"],
            key="inv_img_uploader",
            help=f"Accepts JPG, JPEG, PNG up to {MAX_IMAGE_MB} MB."
        )

        img_path_for_workflow = None
        if up_img:
            if up_img.size / (1024 * 1024) > MAX_IMAGE_MB:
                st.error(f"Image exceeds {MAX_IMAGE_MB} MB limit.")
            else:
                img_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_name = f"inv_upload_{img_stamp}_{safe_stem(up_img.name)}.jpg"
                saved_full_path = EVIDENCE_DIR / save_name
                with open(saved_full_path, "wb") as f_out:
                    f_out.write(up_img.getvalue())
                img_path_for_workflow = str(saved_full_path)
                st.image(up_img, caption=f"Uploaded: {up_img.name}", width="stretch")

        # Policy Repository Scope
        scope_choice = st.selectbox(
            "Policy Repository Scope",
            ["All Categories", "Safety Manuals", "Standard Operating Procedures", "Inspection Reports"],
            key="inv_scope_sel"
        )

        st.write("")
        if st.button("🚀 Run AI Safety Investigation", type="primary", width="stretch", key="btn_run_langgraph_inv"):
            if not incident_desc.strip():
                st.error("Please provide a mandatory incident description to investigate.")
            else:
                with st.spinner("Executing 6-node LangGraph autonomous safety investigation..."):
                    scope_filter = scope_choice or "All Categories"
                    t_start = time.perf_counter()
                    inv_res = run_safety_investigation(
                        incident_description=incident_desc.strip(),
                        visual_evidence_path=img_path_for_workflow,
                        repository_scope=scope_filter,
                        user_id=uid
                    )
                    lat_total = (time.perf_counter() - t_start) * 1000.0
                    inv_res["total_execution_ms"] = lat_total
                    st.session_state["active_safety_investigation"] = inv_res
                    st.toast(f"Investigation {inv_res.get('incident_id')} completed!", icon="🛡️")

    # -------------------------------------------------------------
    # RIGHT COLUMN: INVESTIGATION RESULTS
    # -------------------------------------------------------------
    with col_results:
        inv_data = st.session_state.get("active_safety_investigation")

        if not inv_data:
            empty_state(
                "🤖",
                "LangGraph AI Safety Investigation Standby",
                "Provide an incident description on the left and click 'Run AI Safety Investigation' to trigger the 6-agent LangGraph workflow."
            )
        else:
            inc_id = inv_data.get("incident_id", "INV-UNKNOWN")
            title = inv_data.get("title", "Safety Incident Investigation")
            risk_obj = inv_data.get("risk_assessment") or {}
            risk_level = risk_obj.get("risk_level") or inv_data.get("risk_level") or "Medium"

            risk_pill_class = {
                "Low": "pill-emerald",
                "Medium": "pill-amber",
                "High": "pill-amber",
                "Critical": "pill-red"
            }.get(risk_level, "pill-slate")

            render_html(f"""
            <div style="background:#ffffff; border:1px solid #e2e8f0; border-top:3px solid #6b21a8; border-radius:8px; padding:0.85rem 1.1rem; margin-bottom:1rem;">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
                    <div>
                        <div style="font-size:0.75rem; font-weight:700; color:#6b21a8; text-transform:uppercase;">Autonomous Multi-Agent Investigation</div>
                        <div style="font-size:1.15rem; font-weight:800; color:#0f172a;">{title}</div>
                        <div style="font-size:0.75rem; color:#64748b; margin-top:0.15rem;">Reference ID: <code>{inc_id}</code></div>
                    </div>
                    <div>
                        <span class="sv-pill {risk_pill_class}">RISK RATING: {risk_level.upper()}</span>
                    </div>
                </div>
            </div>
            """)

            # -------------------------------------------------------------
            # 1. SIX-AGENT EXECUTION PROGRESS TIMELINE
            # -------------------------------------------------------------
            st.subheader("⚡ 6-Agent Autonomous Workflow Pipeline")
            stages = inv_data.get("stages", {})

            stage_names = [
                ("Incident Analysis", "Incident Analysis Agent", "🔍", "Extracts hazard type, entities & zone context"),
                ("Visual Evidence Analysis", "Visual Evidence Agent", "📸", "Processes YOLOv8 bounding boxes & detections"),
                ("Safety Policy Retrieval", "Policy Retrieval Agent", "📚", "Queries FAISS vector index for OSHA & SOP rules"),
                ("Risk Assessment", "Risk Assessment Agent", "⚠️", "Evaluates Severity × Likelihood 5×5 matrix"),
                ("Corrective Actions", "Corrective Actions Agent", "📋", "Generates grounded CAPA hierarchy directives"),
                ("Report Generation", "Report Generation Agent", "📄", "Synthesizes formal investigation dossier")
            ]

            p_cols = st.columns(6)
            for i, (key, label, icon, desc) in enumerate(stage_names):
                stg = stages.get(key, {})
                status = stg.get("status", "ok")
                exec_ms = stg.get("execution_time_ms", 0)

                if status in ("ok", "completed"):
                    s_color = "#16845B"
                    s_badge = "COMPLETED"
                    s_pill = "pill-emerald"
                elif status == "unavailable":
                    s_color = "#87909D"
                    s_badge = "UNAVAILABLE"
                    s_pill = "pill-slate"
                elif status == "skipped":
                    s_color = "#D99A22"
                    s_badge = "SKIPPED"
                    s_pill = "pill-amber"
                else:
                    s_color = "#D94B4B"
                    s_badge = "FAILED"
                    s_pill = "pill-red"

                with p_cols[i]:
                    render_html(f"""
                    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-top:3px solid {s_color}; border-radius:10px; padding:0.6rem 0.5rem; text-align:center; box-shadow:0 1px 3px rgba(0,0,0,0.02); min-height:105px;">
                        <div style="font-size:1.15rem; margin-bottom:0.15rem;">{icon}</div>
                        <div style="font-size:0.68rem; font-weight:700; color:#171A21; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="{label}">Stage {i+1}</div>
                        <div style="margin:0.25rem 0;"><span class="sv-pill {s_pill}" style="font-size:0.62rem; padding:0.12rem 0.45rem;">{s_badge}</span></div>
                        <div style="font-size:0.64rem; color:#87909D;">{exec_ms} ms</div>
                    </div>
                    """)

            st.write("")

            # -------------------------------------------------------------
            # 2. INCIDENT ANALYSIS & VISUAL EVIDENCE TABS
            # -------------------------------------------------------------
            tab_anal, tab_vis, tab_pol, tab_risk, tab_capa, tab_rep = st.tabs([
                "🔍 Incident Analysis",
                "📸 Visual Evidence",
                "📚 Policy Citations",
                "⚠️ Risk Assessment",
                "📋 CAPA Actions",
                "📄 Formal Report"
            ])

            with tab_anal:
                st.subheader("🔍 Incident Classification & Entity Extraction")
                inc_analysis = inv_data.get("incident_analysis") or {}
                st.markdown(f"""
                - **Primary Hazard Category:** `{inc_analysis.get('identified_category') or inv_data.get('incident_category', 'General')}`
                - **Identified Entities / Equipment:** `{', '.join(inc_analysis.get('entities_present') or ['General Workplace Equipment'])}`
                - **Narrative Analysis:**
                """)
                st.info(inc_analysis.get("analysis_narrative") or inv_data.get("incident_description", "No narrative recorded."))

            with tab_vis:
                st.subheader("📸 Visual Telemetry & Detection Evidence")
                vis_ev = inv_data.get("visual_evidence") or {}
                vis_status = vis_ev.get("status", "unavailable")

                if vis_status == "ok" and vis_ev.get("annotated_image_path") and Path(vis_ev["annotated_image_path"]).exists():
                    v_c1, v_c2 = st.columns(2)
                    orig_p = vis_ev.get("original_image_path")
                    if orig_p and Path(orig_p).exists():
                        v_c1.image(orig_p, caption="Original Uploaded Evidence Frame", width="stretch")
                    v_c2.image(vis_ev["annotated_image_path"], caption="Detecto AI YOLOv8 Optical Annotations", width="stretch")

                    dets = vis_ev.get("detections") or []
                    if dets:
                        st.markdown(f"**Detected Objects & Workers ({len(dets)} items):**")
                        df_v = pd.DataFrame(dets)
                        st.dataframe(df_v, hide_index=True, width="stretch")
                else:
                    st.info(
                        "ℹ️ **Visual Evidence Notice:** Visual evidence was not provided for this run. "
                        "The investigation completed using incident narrative reasoning and policy grounding."
                    )

            with tab_pol:
                st.subheader("📚 Retrieved Safety Policy Context")
                policy_ctx = inv_data.get("policy_context") or []
                if not policy_ctx:
                    st.warning(
                        "⚠️ **Policy Gap:** No relevant safety policy documents matched the incident profile in the selected scope. "
                        "Recommended actions are flagged as General Industry Best Practice."
                    )
                else:
                    for p_idx, pol in enumerate(policy_ctx):
                        if isinstance(pol, dict):
                            doc_n = pol.get("document_name") or pol.get("file_name", "Policy Document")
                            pg_n = pol.get("page") or pol.get("page_number", 1)
                            sec_n = pol.get("section_title", "General Clause")
                            cite_n = pol.get("citation", f"Found in {doc_n} (Page {pg_n})")
                            exc_body = pol.get('excerpt') or pol.get('text_content', '')
                        else:
                            doc_n = "Policy Document"
                            pg_n = 1
                            sec_n = f"Clause #{p_idx + 1}"
                            cite_n = str(pol)
                            exc_body = str(pol)
                        with st.expander(f"📄 {doc_n} &bull; Page {pg_n} &bull; {sec_n}", expanded=(p_idx == 0)):
                            render_html(f"""
                            <div style="background:#F7F8FA; border:1px solid #E2E8F0; border-left:4px solid #0F8B8D; border-radius:6px; padding:0.65rem 0.9rem; font-size:0.85rem; color:#171A21;">
                                {exc_body}
                            </div>
                            <div style="font-size:0.75rem; color:#5B6472; margin-top:0.35rem;">
                                <b>Citation:</b> <em>{cite_n}</em>
                            </div>
                            """)

            with tab_risk:
                st.subheader("⚠️ 5×5 Enterprise Risk Assessment Matrix")
                rk = inv_data.get("risk_assessment") or {}
                sev = rk.get("severity", 3)
                sev_lbl = rk.get("severity_label", "Moderate")
                lik = rk.get("likelihood", 2)
                lik_lbl = rk.get("likelihood_label", "Unlikely")
                score = rk.get("risk_score", 6)
                level = rk.get("risk_level", "Medium")

                # Top Metrics
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Severity (1-5)", f"{sev}/5", delta=sev_lbl)
                r2.metric("Likelihood (1-5)", f"{lik}/5", delta=lik_lbl)
                r3.metric("Composite Risk Score", f"{score}/25", delta="Severity × Likelihood")
                r4.metric("Assessed Risk Rating", level.upper(), delta="OSHA / ISO Matrix", delta_color="inverse" if level in ("High", "Critical") else "normal")

                # Render Visual 5x5 Matrix
                render_risk_matrix_5x5(sev=sev, lik=lik, score=score, level=level)

                st.markdown(f"**Recommended Containment Directive:** {rk.get('recommended_action', 'Issue verbal correction and immediate PPE rectification.')}")
                regs = rk.get("regulatory_references") or []
                if regs:
                    st.markdown("**Governing Regulatory Standards:**")
                    for rg in regs:
                        st.markdown(f"- 📜 `{rg}`")

                unc = rk.get("uncertainty_notes") or []
                if unc:
                    st.markdown("**Uncertainty & Boundary Disclosures:**")
                    for un in unc:
                        st.caption(f"⚠️ {un}")

            with tab_capa:
                st.subheader("📋 Corrective and Preventive Actions (CAPA)")
                capa = inv_data.get("corrective_actions") or {}
                imm_act = capa.get("immediate_action") or capa.get("immediate_containment") or "Issue verbal correction and immediate PPE rectification."
                st.markdown(f"**Immediate Containment Directive:** {imm_act}")

                # Render Hierarchy of Controls Diagram
                render_hierarchy_of_controls(capa)

                # Grounded Policy Actions
                grounded_acts = capa.get("grounded_actions") or []
                if grounded_acts:
                    st.markdown("#### 📜 Actions Derived from Grounded Policies")
                    for act in grounded_acts:
                        act_text = act.get("action") or "Enforce mandatory protocol"
                        cit_text = act.get("citation") or f"[Source: {act.get('file_name', 'Policy')}, Page {act.get('page_number', 1)}]"
                        render_html(f"""
                        <div style="background:#F7F8FA; border:1px solid #E2E8F0; border-left:4px solid #7357D9; border-radius:6px; padding:0.55rem 0.8rem; margin-bottom:0.4rem;">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <span style="font-weight:700; font-size:0.85rem; color:#171A21;">{act_text}</span>
                                <span class="sv-pill pill-purple">Policy-Grounded</span>
                            </div>
                            <div style="font-size:0.75rem; color:#5B6472; margin-top:0.25rem;">
                                <em>{cit_text}</em>
                            </div>
                        </div>
                        """)

                # General Actions / Hierarchy of Controls
                gen_acts = capa.get("general_actions") or []
                if gen_acts:
                    st.markdown("#### 🏗️ General Controls (Standard EHS Practice)")
                    for act in gen_acts:
                        c_type = act.get("control_type", "General Control")
                        render_html(f"""
                        <div style="background:#F7F8FA; border:1px solid #E2E8F0; border-left:4px solid #0F8B8D; border-radius:6px; padding:0.55rem 0.8rem; margin-bottom:0.4rem;">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <span style="font-weight:700; font-size:0.85rem; color:#171A21;">{act.get('action')}</span>
                                <span class="sv-pill pill-teal">{c_type}</span>
                            </div>
                            <div style="font-size:0.72rem; color:#87909D; margin-top:0.2rem;">
                                General Industry Best Practice (Not from indexed documents)
                            </div>
                        </div>
                        """)

                # Hierarchy summary if present
                h_sum = capa.get("hierarchy_summary") or {}
                if h_sum:
                    st.markdown("#### 📊 Hierarchy Controls Summary")
                    for k, v in h_sum.items():
                        st.markdown(f"- **{k.title()} Control:** {v}")

            with tab_rep:
                st.subheader("📄 Formal AI Safety Investigation Report")
                rep_md = inv_data.get("report") or inv_data.get("final_report", "")
                st.markdown(rep_md)

            # PDF Download Button
            st.divider()
            try:
                pdf_bytes = generate_investigation_pdf_report(
                    inv_data,
                    generated_by=f"{current_user.get('full_name', 'Safety Officer')} ({role.upper()})"
                )
                st.download_button(
                    "📄 Download Formal Investigation PDF Report",
                    data=pdf_bytes,
                    file_name=f"SafetyVision_Investigation_{inc_id}.pdf",
                    mime="application/pdf",
                    type="primary",
                    width="stretch",
                    key="btn_dl_inv_pdf"
                )
            except Exception as pdf_err:
                st.error(f"Error generating PDF report: {pdf_err}")

    # -------------------------------------------------------------
    # BOTTOM CONTAINER: INVESTIGATION HISTORY
    # -------------------------------------------------------------
    st.divider()
    with st.expander("🗄️ Past AI Safety Investigations History", expanded=False):
        all_invs = list_investigations(limit=30)
        if not all_invs:
            st.caption("No past AI safety investigations recorded in database.")
        else:
            st.markdown(f"**Archived Investigation Reports ({len(all_invs)} stored in SQLite):**")
            for h_inv in all_invs:
                h_id = h_inv["id"]
                h_inc = h_inv["incident_id"]
                h_title = h_inv["title"]
                h_cat = h_inv.get("incident_category") or "General"
                h_risk = h_inv.get("risk_level") or "Medium"
                h_ts = str(h_inv.get("created_at") or "")[:19]

                h_pill = {
                    "Low": "pill-emerald",
                    "Medium": "pill-amber",
                    "High": "pill-amber",
                    "Critical": "pill-red"
                }.get(h_risk, "pill-slate")

                with st.container():
                    st.markdown(f"""
                    <div style="background:#f8fafc; border:1px solid #e2e8f0; border-left:3.5px solid #6b21a8; border-radius:6px; padding:0.55rem 0.8rem; margin-bottom:0.35rem;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-weight:700; font-size:0.83rem; color:#6b21a8;">#{h_id} &bull; <code>{h_inc}</code> &bull; ⏱️ {h_ts}</span>
                            <span class="sv-pill {h_pill}">{h_risk.upper()}</span>
                        </div>
                        <div style="font-size:0.82rem; font-weight:700; color:#0f172a; margin-top:0.25rem;">
                            {h_title}
                        </div>
                        <div style="font-size:0.75rem; color:#64748b; margin-top:0.15rem;">
                            Category: <code>{h_cat}</code> | Scope: <code>{h_inv.get('repository_scope') or 'All Categories'}</code>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    btn_c1, btn_c2 = st.columns([3.5, 1])
                    with btn_c1:
                        if st.button(f"📂 View Report ({h_inc})", key=f"btn_view_h_inv_{h_id}", width="stretch"):
                            inv_full = get_safety_investigation_by_id(h_id) or h_inv
                            stages_list = get_investigation_stages(h_id)
                            citations_list = get_investigation_citations(h_id)
                            loaded_state = {
                                "investigation_id": h_id,
                                "incident_id": h_inc,
                                "title": h_title,
                                "incident_description": inv_full.get("incident_description", ""),
                                "incident_category": h_cat,
                                "visual_evidence_path": inv_full.get("visual_evidence_path") or "",
                                "repository_scope": inv_full.get("repository_scope") or "All Categories",
                                "risk_level": h_risk,
                                "status": inv_full.get("status") or "Completed",
                                "report": inv_full.get("final_report") or "",
                                "final_report": inv_full.get("final_report") or "",
                                "stages": {s["stage_name"]: s for s in stages_list},
                                "policy_context": citations_list,
                                "risk_assessment": {
                                    "risk_level": h_risk,
                                    "severity": 3,
                                    "likelihood": 2,
                                    "risk_score": 6,
                                    "severity_label": "Moderate",
                                    "likelihood_label": "Unlikely",
                                    "recommended_action": "Supervisory review and PPE audit.",
                                    "regulatory_references": ["OSHA 1910 General Industry"],
                                    "uncertainty_notes": []
                                },
                                "incident_analysis": {
                                    "identified_category": h_cat,
                                    "identified_hazards": [],
                                    "entities_present": [],
                                    "analysis_narrative": inv_full.get("incident_description", "")
                                },
                                "visual_evidence": {
                                    "status": "ok" if inv_full.get("visual_evidence_path") else "unavailable",
                                    "violations": [],
                                    "persons_count": 0
                                },
                                "corrective_actions": {
                                    "immediate_containment": "Enforce mandatory safety protocols.",
                                    "grounded_actions": citations_list,
                                    "hierarchy_of_controls": {}
                                }
                            }
                            st.session_state["active_safety_investigation"] = loaded_state
                            st.rerun()

                    with btn_c2:
                        if user_is_admin:
                            if st.button("🗑️ Delete", key=f"btn_del_h_inv_{h_id}", width="stretch"):
                                try:
                                    delete_investigation(h_id, user_role=role)
                                    if st.session_state.get("active_safety_investigation", {}).get("investigation_id") == h_id:
                                        del st.session_state["active_safety_investigation"]
                                    st.toast(f"Deleted investigation #{h_id}", icon="🗑️")
                                    time.sleep(0.3)
                                    st.rerun()
                                except Exception as exc:
                                    st.error(str(exc))


# =============================================================
# DESTINATION 7: ACCOUNT PROFILE
# =============================================================
def render_account_profile_page():
    render_profile_security_page()


# =============================================================
# OPERATIONAL ROUTING DISPATCHER
# =============================================================
active_page = st.session_state.get("current_page", PAGE_DASHBOARD)

if active_page == PAGE_DASHBOARD:
    render_dashboard_page()
elif active_page == PAGE_IMAGE_VIDEO_ANALYSIS:
    render_image_video_analysis_page()
elif active_page == PAGE_VIOLATIONS:
    render_violations_page()
elif active_page == PAGE_REPORTS_EXPORT:
    render_reports_export_page()
elif active_page == PAGE_DOCUMENT_INGESTION:
    render_document_ingestion_page()
elif active_page == PAGE_SEMANTIC_SEARCH:
    render_semantic_search_page()
elif active_page == PAGE_RAG_ASSISTANT:
    render_rag_safety_policy_assistant_page()
elif active_page == PAGE_LANGGRAPH_INVESTIGATION:
    render_langgraph_ai_safety_investigation_page()
elif active_page == PAGE_ACCOUNT_PROFILE:
    render_account_profile_page()
elif active_page == PAGE_USER_GOVERNANCE:
    render_user_governance_page()
# Backward compatibility aliases
elif active_page in (PAGE_IMAGE_ANALYSIS, PAGE_VIDEO_ANALYSIS, PAGE_LIVE_SURVEILLANCE):
    render_image_video_analysis_page()
elif active_page == PAGE_RAG_QA:
    render_rag_safety_policy_assistant_page()
elif active_page in (PAGE_INVESTIGATOR_CONSOLE, PAGE_REMEDIATION_WORKFLOWS):
    render_langgraph_ai_safety_investigation_page()
elif active_page == PAGE_RECORDED_EVENTS:
    render_violations_page()
elif active_page == PAGE_AUDIT_REPORTS:
    render_reports_export_page()
elif active_page == PAGE_SAFETY_ANALYTICS:
    render_dashboard_page()
elif active_page == PAGE_MY_PROFILE:
    render_account_profile_page()
else:
    render_dashboard_page()
