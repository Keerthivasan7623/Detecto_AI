"""SafetyVision AI - Workplace Safety KPI Cards Component.
Reusable high-contrast enterprise KPI cards adhering to Taste Skill design principles.
"""

from typing import Any, Dict, Optional
import streamlit as st


def _clean_html_render(html_content: str) -> None:
    """Renders HTML safely in Streamlit without triggering Markdown indented code block formatting."""
    cleaned = [
        line.strip() for line in html_content.splitlines()
        if line.strip() and not line.strip().startswith("<!--")
    ]
    st.markdown(" ".join(cleaned), unsafe_allow_html=True)


def _render_card_markup(
    title: str,
    value_display: str,
    icon: str,
    subtext: str,
    pill_text: str,
    pill_class: str,
    accent_border_color: str = "#0F8B8D"
) -> str:
    """Generates clean HTML string for a single KPI card."""
    return f"""
    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-top:3px solid {accent_border_color}; border-radius:12px; padding:1rem 1.15rem; box-shadow:0 1px 3px rgba(0,0,0,0.03); min-height:130px; display:flex; flex-direction:column; justify-content:space-between; margin-bottom:0.75rem;">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.04em; color:#5B6472;">
                {title}
            </div>
            <div style="width:34px; height:34px; border-radius:8px; background:#F1F3F5; display:flex; align-items:center; justify-content:center; font-size:1.15rem; color:#172B4D;">
                {icon}
            </div>
        </div>
        <div style="font-size:1.95rem; font-weight:800; color:#171A21; letter-spacing:-0.03em; margin:0.35rem 0 0.15rem 0; line-height:1.15;">
            {value_display}
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.35rem; margin-top:0.25rem;">
            <span style="font-size:0.75rem; color:#87909D;">{subtext}</span>
            <span class="sv-pill {pill_class}" style="font-size:0.68rem; padding:0.12rem 0.5rem;">{pill_text}</span>
        </div>
    </div>
    """


def render_kpi_cards(
    kpi_data: Optional[Dict[str, Any]] = None,
    user_id: Optional[int] = None,
    role: str = "admin",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    """Renders the four critical workplace safety KPI cards in a responsive horizontal layout:
    1. Total Inspections (Clipboard icon)
    2. Total PPE Violations (Alert triangle icon)
    3. Resolved Violations (Shield check icon)
    4. Compliance Percentage (Chart line icon)

    Args:
        kpi_data: Optional dictionary containing pre-fetched KPI figures.
                  If omitted, figures are retrieved directly from SQLite via database.get_workplace_safety_kpis.
        user_id: User ID for RBAC scoping.
        role: User role ("admin" or "standard").
        start_date: Optional ISO date string for start of filter window.
        end_date: Optional ISO date string for end of filter window.

    Returns:
        The dictionary of KPI metrics used to populate the cards.
    """
    if kpi_data is None:
        try:
            from database import get_workplace_safety_kpis
            kpi_data = get_workplace_safety_kpis(
                user_id=user_id,
                role=role,
                start_date=start_date,
                end_date=end_date
            )
        except Exception:
            # Graceful fallback if database function is unavailable
            kpi_data = {
                "total_inspections": 0,
                "total_violations": 0,
                "resolved_violations": 0,
                "compliant_inspections": 0,
                "compliance_percentage": None
            }

    # Extract metrics safely with defaults
    total_insp = kpi_data.get("total_inspections", 0) or 0
    total_viols = kpi_data.get("total_violations", 0) or 0
    resolved_viols = kpi_data.get("resolved_violations", 0) or 0
    compliant_insp = kpi_data.get("compliant_inspections", max(0, total_insp - total_viols))
    comp_pct = kpi_data.get("compliance_percentage")
    is_demo = bool(kpi_data.get("is_demo", False))

    # Calculate compliance percentage if missing but inspections exist
    if comp_pct is None and total_insp > 0:
        comp_pct = round((compliant_insp / total_insp) * 100.0, 1)

    # -------------------------------------------------------------
    # 1. TOTAL INSPECTIONS METRICS
    # -------------------------------------------------------------
    insp_val = f"{total_insp:,}"
    insp_sub = f"{compliant_insp:,} Compliant" if total_insp > 0 else "No inspections in range"
    insp_pill = "DEMO DATA" if is_demo else ("Audited" if total_insp > 0 else "Zero Records")
    insp_pill_cls = "pill-amber" if is_demo else ("pill-navy" if total_insp > 0 else "pill-slate")

    # -------------------------------------------------------------
    # 2. TOTAL PPE VIOLATIONS METRICS
    # -------------------------------------------------------------
    unresolved_count = max(0, total_viols - resolved_viols)
    viol_val = f"{total_viols:,}"
    viol_sub = f"{unresolved_count} Action Required" if unresolved_count > 0 else "All Resolved / Clear"
    viol_pill = "Action Needed" if unresolved_count > 0 else ("All Clear" if total_viols > 0 else "Zero Hazards")
    viol_pill_cls = "pill-red" if unresolved_count > 0 else ("pill-emerald" if total_viols > 0 else "pill-slate")

    # -------------------------------------------------------------
    # 3. RESOLVED VIOLATIONS METRICS
    # -------------------------------------------------------------
    res_val = f"{resolved_viols:,}"
    if total_viols > 0:
        res_rate = (resolved_viols / total_viols) * 100.0
        res_sub = f"{res_rate:.1f}% Resolution Rate"
        res_pill = f"{res_rate:.0f}% Closed"
        res_pill_cls = "pill-emerald" if res_rate >= 80.0 else ("pill-amber" if res_rate >= 50.0 else "pill-red")
    else:
        res_sub = "No violations to resolve"
        res_pill = "100% Closed"
        res_pill_cls = "pill-emerald"

    # -------------------------------------------------------------
    # 4. COMPLIANCE PERCENTAGE METRICS
    # -------------------------------------------------------------
    if comp_pct is not None:
        comp_val = f"{comp_pct:.1f}%"
        if comp_pct >= 90.0:
            comp_sub = "Exceeds Safety Threshold"
            comp_pill = "Exemplary (Grade A)"
            comp_pill_cls = "pill-emerald"
            comp_accent = "#16845B"
        elif comp_pct >= 80.0:
            comp_sub = "OSHA Standard Met"
            comp_pill = "Compliant (Grade B)"
            comp_pill_cls = "pill-teal"
            comp_accent = "#0F8B8D"
        elif comp_pct >= 65.0:
            comp_sub = "Corrective Review Advised"
            comp_pill = "Warning (Grade C)"
            comp_pill_cls = "pill-amber"
            comp_accent = "#D99A22"
        else:
            comp_sub = "Critical Incident Ratio"
            comp_pill = "Non-Compliant"
            comp_pill_cls = "pill-red"
            comp_accent = "#D94B4B"
    else:
        comp_val = "N/A"
        comp_sub = "No inspections in period"
        comp_pill = "Insufficient Data"
        comp_pill_cls = "pill-slate"
        comp_accent = "#87909D"

    # -------------------------------------------------------------
    # RENDER FOUR HORIZONTAL CARDS
    # -------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        card1_html = _render_card_markup(
            title="Total Inspections",
            value_display=insp_val,
            icon="📋",
            subtext=insp_sub,
            pill_text=insp_pill,
            pill_class=insp_pill_cls,
            accent_border_color="#172B4D"
        )
        _clean_html_render(card1_html)

    with col2:
        card2_html = _render_card_markup(
            title="Total PPE Violations",
            value_display=viol_val,
            icon="⚠️",
            subtext=viol_sub,
            pill_text=viol_pill,
            pill_class=viol_pill_cls,
            accent_border_color="#D94B4B" if unresolved_count > 0 else "#16845B"
        )
        _clean_html_render(card2_html)

    with col3:
        card3_html = _render_card_markup(
            title="Resolved Violations",
            value_display=res_val,
            icon="✅",
            subtext=res_sub,
            pill_text=res_pill,
            pill_class=res_pill_cls,
            accent_border_color="#16845B"
        )
        _clean_html_render(card3_html)

    with col4:
        card4_html = _render_card_markup(
            title="Compliance Percentage",
            value_display=comp_val,
            icon="📈",
            subtext=comp_sub,
            pill_text=comp_pill,
            pill_class=comp_pill_cls,
            accent_border_color=comp_accent
        )
        _clean_html_render(card4_html)

    return kpi_data
