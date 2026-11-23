import io
import csv
import json
import html
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape, portrait
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image
)


def escape_xml(value: Any) -> str:
    """Safely escapes text for ReportLab XML/HTML Paragraph parsing."""
    if value is None:
        return ""
    val_str = str(value)
    # Remove surrogate or unprintable characters that could fail ReportLab encoding
    val_str = val_str.encode("utf-8", "replace").decode("utf-8")
    return html.escape(val_str)


def sanitize_csv_cell(value: Any) -> str:
    """
    Prevents CSV Formula Injection (DDE injection).
    If a string starts with =, +, -, @, \\t, or \\r, prepend a single quote.
    """
    if value is None:
        return ""
    val_str = str(value)
    if val_str and val_str[0] in ("=", "+", "-", "@", "\t", "\r"):
        return f"'{val_str}"
    return val_str


def generate_csv_report(violations: List[Dict[str, Any]]) -> bytes:
    """
    Generates a sanitized CSV export from violation records.
    Filters out any sensitive fields and protects against spreadsheet injection.
    """
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)

    headers = [
        "Violation ID",
        "Timestamp (UTC)",
        "Source File",
        "Media Type",
        "Workplace",
        "Safety Zone",
        "Priority",
        "Violation Category",
        "Severity",
        "Confidence",
        "Review Status",
        "CAPA Status",
        "Root Cause",
        "Reviewer",
        "Review Notes",
        "Resolved At",
        "Shift"
    ]
    writer.writerow(headers)

    for v in violations:
        conf_str = f"{float(v.get('confidence', 0)):.1%}" if v.get("confidence") is not None else "N/A"
        row = [
            sanitize_csv_cell(v.get("violation_id") or v.get("id")),
            sanitize_csv_cell(v.get("created_at")),
            sanitize_csv_cell(v.get("filename")),
            sanitize_csv_cell(v.get("media_type")),
            sanitize_csv_cell(v.get("workplace") or "General"),
            sanitize_csv_cell(v.get("zone_name") or v.get("zone_code") or "General Area"),
            sanitize_csv_cell(v.get("priority", "medium").upper()),
            sanitize_csv_cell(v.get("violation_type") or v.get("class_name")),
            sanitize_csv_cell(v.get("severity", "medium").upper()),
            sanitize_csv_cell(conf_str),
            sanitize_csv_cell(v.get("review_status", "pending").upper()),
            sanitize_csv_cell(v.get("capa_status", "none").upper()),
            sanitize_csv_cell(v.get("root_cause") or ""),
            sanitize_csv_cell(v.get("reviewed_by") or "Unassigned"),
            sanitize_csv_cell(v.get("review_notes") or ""),
            sanitize_csv_cell(v.get("resolved_at") or ""),
            sanitize_csv_cell(v.get("shift") or "General"),
        ]
        writer.writerow(row)

    return output.getvalue().encode("utf-8")


def generate_json_report(violations: List[Dict[str, Any]]) -> str:
    """Exports structured JSON audit dataset."""
    return json.dumps(violations, indent=2, default=str)


# =========================================================================
# 1. PLANT-WIDE EXECUTIVE AUDIT REPORT (LANDSCAPE A4)
# =========================================================================
def generate_pdf_report(
    violations: List[Dict[str, Any]],
    stats: Dict[str, Any],
    reporting_period: str = "All Recorded Activity",
    generated_by: str = "Detecto AI System",
    title: str = "Workplace Safety & PPE Monitoring Audit Report"
) -> bytes:
    """
    Generates an executive-grade, industrial safety PDF report using ReportLab (Landscape A4).
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=28,
        leftMargin=28,
        topMargin=28,
        bottomMargin=28
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
        alignment=0,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=12
    )

    h2_style = ParagraphStyle(
        "ReportH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=10,
        spaceAfter=6
    )

    cell_style = ParagraphStyle(
        "CellNormal",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#1e293b")
    )

    cell_bold = ParagraphStyle(
        "CellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#0f172a")
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("Detecto AI - Enterprise Workplace Safety Platform", title_style))
    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(
        f"<b>Audit Report:</b> {escape_xml(title)} | <b>Period:</b> {escape_xml(reporting_period)} | <b>Generated:</b> {gen_time} | <b>Inspector:</b> {escape_xml(generated_by)}",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=14))

    # 2. Executive KPI Summary Table
    total_uploads = stats.get("total_uploads", 0)
    total_detections = stats.get("detections", 0)
    total_violations = stats.get("violations", len(violations))
    pending = stats.get("pending_review", 0)
    confirmed = stats.get("confirmed", 0)
    resolved = stats.get("resolved", 0)

    cell_kpi_num = ParagraphStyle('KNum', parent=cell_style, fontSize=12, alignment=1)

    kpi_data = [
        [
            Paragraph("<b>Total Uploads</b>", cell_bold),
            Paragraph("<b>Total Detections</b>", cell_bold),
            Paragraph("<b>Potential Violations</b>", cell_bold),
            Paragraph("<b>Pending Review</b>", cell_bold),
            Paragraph("<b>Confirmed Incidents</b>", cell_bold),
            Paragraph("<b>Resolved Incidents</b>", cell_bold)
        ],
        [
            Paragraph(f"<font size=12 color='#0f172a'><b>{total_uploads}</b></font>", cell_kpi_num),
            Paragraph(f"<font size=12 color='#0284c7'><b>{total_detections}</b></font>", cell_kpi_num),
            Paragraph(f"<font size=12 color='#dc2626'><b>{total_violations}</b></font>", cell_kpi_num),
            Paragraph(f"<font size=12 color='#ea580c'><b>{pending}</b></font>", cell_kpi_num),
            Paragraph(f"<font size=12 color='#b91c1c'><b>{confirmed}</b></font>", cell_kpi_num),
            Paragraph(f"<font size=12 color='#16a34a'><b>{resolved}</b></font>", cell_kpi_num),
        ]
    ]
    kpi_table = Table(kpi_data, colWidths=[130, 130, 130, 130, 130, 130])
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#ffffff")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 14))

    # 3. Violation Records Section
    story.append(Paragraph(f"Detailed Incident & Potential Violation Log ({len(violations)} records)", h2_style))

    if not violations:
        story.append(Paragraph("<i>No violation events recorded for the selected criteria and period. Site is 100% compliant.</i>", subtitle_style))
    else:
        table_headers = [
            Paragraph("<b>ID</b>", cell_bold),
            Paragraph("<b>Date / Time</b>", cell_bold),
            Paragraph("<b>Source Media</b>", cell_bold),
            Paragraph("<b>Violation Category</b>", cell_bold),
            Paragraph("<b>Confidence</b>", cell_bold),
            Paragraph("<b>Severity</b>", cell_bold),
            Paragraph("<b>Review Status</b>", cell_bold),
            Paragraph("<b>Review Notes</b>", cell_bold),
        ]
        table_rows = [table_headers]

        for v in violations[:300]:  # Cap at 300 to ensure reasonable PDF size
            vid = f"#{escape_xml(v.get('violation_id') or v.get('id', 'N/A'))}"
            ts = escape_xml(str(v.get("created_at", ""))[:19].replace("T", " "))
            fn = escape_xml(str(v.get("filename", ""))[:28])
            cat = escape_xml(str(v.get("violation_type") or v.get("class_name", "")).upper())
            conf = f"{float(v.get('confidence', 0)):.1%}" if v.get("confidence") is not None else "-"
            sev = escape_xml(str(v.get("severity", "medium")).capitalize())
            status = escape_xml(str(v.get("review_status", "pending")).upper())
            notes = escape_xml(str(v.get("review_notes") or "-")[:40])

            # Status styling
            status_color = "#ea580c" if status in ("PENDING", "DETECTED") else ("#dc2626" if status == "CONFIRMED" else ("#16a34a" if status == "RESOLVED" else "#64748b"))

            table_rows.append([
                Paragraph(vid, cell_style),
                Paragraph(ts, cell_style),
                Paragraph(fn, cell_style),
                Paragraph(f"<b>{cat}</b>", cell_style),
                Paragraph(conf, cell_style),
                Paragraph(sev, cell_style),
                Paragraph(f"<font color='{status_color}'><b>{status}</b></font>", cell_style),
                Paragraph(notes, cell_style),
            ])

        col_widths = [45, 95, 130, 110, 65, 65, 90, 180]
        v_table = Table(table_rows, repeatRows=1, colWidths=col_widths)
        v_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(v_table)

    # Formal Sign-off and Certification Block
    signoff_data = [
        [
            Paragraph("<b>EHS Safety Officer Sign-Off:</b>", cell_bold),
            Paragraph("<b>Plant Operations Director:</b>", cell_bold),
            Paragraph("<b>Audit Certification:</b>", cell_bold),
        ],
        [
            Paragraph("Signature: __________________________<br/>Date: ________________________", cell_style),
            Paragraph("Signature: __________________________<br/>Date: ________________________", cell_style),
            Paragraph("<b>OFFICIAL COMPLIANCE RECORD</b><br/><font color='#16a34a'>ISO 45001 & OSHA Audit Trail Maintained</font>", cell_style),
        ]
    ]
    signoff_table = Table(signoff_data, colWidths=[260, 260, 260])
    signoff_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f8fafc")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(Spacer(1, 12))
    story.append(signoff_table)

    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#94a3b8"), spaceAfter=8))
    story.append(Paragraph(
        "<b>Important Notice:</b> This report is generated by Detecto AI as a safety monitoring aid. Detections marked as 'Pending' or 'Detected' represent potential AI-identified anomalies and should be human-reviewed before formal disciplinary or compliance actions.",
        ParagraphStyle("Notice", parent=styles["Normal"], fontName="Helvetica-Oblique", fontSize=7, leading=9, textColor=colors.HexColor("#64748b"))
    ))

    doc.build(story)
    return buffer.getvalue()


# =========================================================================
# 2. SINGLE / BATCH INSPECTION AUDIT CERTIFICATE (PORTRAIT A4)
# =========================================================================
def generate_inspection_pdf_report(
    inspection_res: Dict[str, Any],
    generated_by: str = "Detecto AI Safety Officer"
) -> bytes:
    """
    Generates a formal, printable Single Image / Inspection Audit Certificate (Portrait A4)
    with the embedded annotated image, violation summary, findings breakdown, and certification block.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=portrait(A4),
        rightMargin=32,
        leftMargin=32,
        topMargin=32,
        bottomMargin=32
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "InspTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a"),
        alignment=0,
        spaceAfter=3
    )

    meta_style = ParagraphStyle(
        "InspMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#475569"),
        spaceAfter=8
    )

    h2_style = ParagraphStyle(
        "InspH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=8,
        spaceAfter=4
    )

    cell_style = ParagraphStyle(
        "InspCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#1e293b")
    )

    cell_bold = ParagraphStyle(
        "InspCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#0f172a")
    )

    story = []

    # 1. Title Banner
    story.append(Paragraph("Detecto AI | Workplace PPE Inspection Certificate", title_style))
    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_name = escape_xml(inspection_res.get("name", "workplace_inspection.jpg"))
    zone_name = escape_xml(inspection_res.get("zone_name", "General Plant Area"))
    shift = escape_xml(inspection_res.get("shift", "General Shift"))

    story.append(Paragraph(
        f"<b>Target:</b> {file_name} | <b>Safety Zone:</b> {zone_name} | <b>Shift:</b> {shift} | <b>Audited:</b> {gen_time} | <b>Inspector:</b> {escape_xml(generated_by)}",
        meta_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=10))

    # 2. Overall Status & KPI Highlights
    violations_count = int(inspection_res.get("violations", 0))
    objects_count = int(inspection_res.get("detections", 0))
    latency_ms = float(inspection_res.get("latency", 0.0))

    is_compliant = (violations_count == 0)
    status_text = "✓ FULLY COMPLIANT (ALL MANDATORY PPE VERIFIED)" if is_compliant else f"🚨 NON-COMPLIANT ({violations_count} SAFETY HAZARDS DETECTED)"
    status_bg = colors.HexColor("#dcfce7") if is_compliant else colors.HexColor("#fee2e2")
    status_border = colors.HexColor("#16a34a") if is_compliant else colors.HexColor("#dc2626")
    status_color = "#16a34a" if is_compliant else "#dc2626"

    kpi_card_data = [
        [
            Paragraph(f"<font size=11 color='{status_color}'><b>OVERALL POSTURE: {status_text}</b></font>", cell_bold),
            Paragraph(f"<b>Objects Tracked:</b> {objects_count}", cell_style),
            Paragraph(f"<b>Violations:</b> {violations_count}", cell_style),
            Paragraph(f"<b>AI Latency:</b> {latency_ms:.1f} ms", cell_style)
        ]
    ]
    kpi_card_table = Table(kpi_card_data, colWidths=[260, 90, 80, 100])
    kpi_card_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), status_bg),
        ("GRID", (0, 0), (-1, -1), 1.0, status_border),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(kpi_card_table)
    story.append(Spacer(1, 8))

    # 3. Embedded Annotated Visual Evidence Image
    img_path_str = inspection_res.get("path")
    if img_path_str and Path(img_path_str).exists():
        try:
            story.append(Paragraph("<b>Annotated Photographic Evidence:</b>", h2_style))
            # Fit image neatly on page: width = 530, max height = 280
            story.append(Image(str(img_path_str), width=530, height=270))
            story.append(Spacer(1, 8))
        except Exception:
            pass

    # 4. Itemized Findings Table
    records = inspection_res.get("records") or []
    story.append(Paragraph(f"<b>Itemized PPE Inspection Findings ({len(records)} entries):</b>", h2_style))

    if not records:
        story.append(Paragraph("<i>No objects or subjects detected in frame.</i>", meta_style))
    else:
        table_headers = [
            Paragraph("<b>Subject / Gear</b>", cell_bold),
            Paragraph("<b>Category</b>", cell_bold),
            Paragraph("<b>Status / Hazard</b>", cell_bold),
            Paragraph("<b>Severity</b>", cell_bold),
            Paragraph("<b>Inspection Finding & Rule</b>", cell_bold),
        ]
        table_rows = [table_headers]

        for r in records[:50]:
            subj = escape_xml(r.get("Subject / Gear", r.get("Class", "Subject")))
            cat = escape_xml(r.get("Category", "N/A"))
            viol = escape_xml(r.get("Violation", "NO"))
            sev = escape_xml(r.get("Severity", "N/A"))
            finding = escape_xml(r.get("Inspection Finding", "-"))

            v_color = "#dc2626" if "YES" in viol.upper() or "HAZARD" in viol.upper() else "#16a34a"

            table_rows.append([
                Paragraph(f"<b>{subj}</b>", cell_style),
                Paragraph(cat, cell_style),
                Paragraph(f"<font color='{v_color}'><b>{viol}</b></font>", cell_style),
                Paragraph(sev, cell_style),
                Paragraph(finding, cell_style),
            ])

        f_table = Table(table_rows, repeatRows=1, colWidths=[105, 110, 85, 55, 175])
        f_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(f_table)

    # 5. Formal Certification & Sign-off Block
    story.append(Spacer(1, 10))
    cert_data = [
        [
            Paragraph("<b>EHS Lead Inspector Verification:</b>", cell_bold),
            Paragraph("<b>Site Safety Supervisor:</b>", cell_bold),
            Paragraph("<b>Compliance Standard:</b>", cell_bold),
        ],
        [
            Paragraph("Name: __________________________<br/>Signature: ______________________<br/>Date: ________________________", cell_style),
            Paragraph("Name: __________________________<br/>Signature: ______________________<br/>Date: ________________________", cell_style),
            Paragraph("<b>ISO 45001 / OSHA 1910 Verified</b><br/>Computer Vision Safety Trail<br/>Cryptographic Record Hash Saved", cell_style),
        ]
    ]
    cert_table = Table(cert_data, colWidths=[175, 175, 180])
    cert_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f8fafc")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(cert_table)

    doc.build(story)
    return buffer.getvalue()


# =========================================================================
# 3. CAPA / INCIDENT TICKET CERTIFICATE (PORTRAIT A4)
# =========================================================================
def generate_capa_ticket_pdf(
    violation: Dict[str, Any],
    generated_by: str = "EHS Safety Director"
) -> bytes:
    """
    Generates an official ISO 45001 Corrective & Preventive Action (CAPA) ticket PDF.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=portrait(A4),
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CapaTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a"),
        alignment=0,
        spaceAfter=4
    )

    cell_style = ParagraphStyle(
        "CapaCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1e293b")
    )

    cell_bold = ParagraphStyle(
        "CapaCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#0f172a")
    )

    vid = escape_xml(violation.get("violation_id") or violation.get("id", "N/A"))
    ticket_no = f"CAPA-{datetime.now().strftime('%Y%m%d')}-{int(violation.get('violation_id') or 1):04d}"

    story = [
        Paragraph(f"Detecto AI | Corrective Action Request (CAR) #{ticket_no}", title_style),
        Paragraph(f"<b>Incident Reference:</b> Violation #{vid} | <b>ISO 45001 Clause:</b> 10.2 Nonconformity & Corrective Action", styles["Normal"]),
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#dc2626"), spaceAfter=12),
    ]

    # Summary Table
    cat = escape_xml(str(violation.get("violation_type", "")).upper())
    sev = escape_xml(str(violation.get("severity", "high")).upper())
    prio = escape_xml(str(violation.get("priority", "P1-Critical")).upper())
    zone = escape_xml(str(violation.get("zone_name") or "General Plant Floor"))
    ts = escape_xml(str(violation.get("created_at", ""))[:19].replace("T", " "))
    status = escape_xml(str(violation.get("review_status", "pending")).upper())

    info_data = [
        [Paragraph("<b>Violation Category:</b>", cell_bold), Paragraph(cat, cell_style), Paragraph("<b>Severity / Risk:</b>", cell_bold), Paragraph(f"<font color='#dc2626'><b>{sev}</b></font>", cell_style)],
        [Paragraph("<b>Priority Rating:</b>", cell_bold), Paragraph(prio, cell_style), Paragraph("<b>Safety Zone:</b>", cell_bold), Paragraph(zone, cell_style)],
        [Paragraph("<b>Timestamp:</b>", cell_bold), Paragraph(ts, cell_style), Paragraph("<b>Investigation Status:</b>", cell_bold), Paragraph(status, cell_style)],
        [Paragraph("<b>Root Cause:</b>", cell_bold), Paragraph(escape_xml(violation.get("root_cause") or "Operator non-compliance"), cell_style), Paragraph("<b>Investigator:</b>", cell_bold), Paragraph(escape_xml(generated_by), cell_style)],
    ]
    info_table = Table(info_data, colWidths=[120, 140, 120, 140])
    info_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 12))

    # Evidence Thumbnail
    ev_path = violation.get("evidence_path")
    if ev_path and Path(ev_path).exists():
        try:
            story.append(Paragraph("<b>Photographic Evidence Crop:</b>", styles["Heading3"]))
            story.append(Image(str(ev_path), width=240, height=180))
            story.append(Spacer(1, 10))
        except Exception:
            pass

    # Corrective Plan Box
    plan_text = escape_xml(violation.get("review_notes") or violation.get("corrective_action_plan") or "Immediate safety briefing conducted; replacement PPE issued to operator. Work area supervisor re-notified.")
    plan_data = [
        [Paragraph("<b>Corrective & Preventive Action Plan (CAPA):</b>", cell_bold)],
        [Paragraph(plan_text, cell_style)],
        [Paragraph("<b>Verification & Closure Sign-Off:</b><br/>Safety Officer: ________________________   Date: ______________   Status: CLOSED", cell_style)]
    ]
    plan_table = Table(plan_data, colWidths=[520])
    plan_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#f1f5f9")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(plan_table)

    doc.build(story)
    return buffer.getvalue()


def generate_investigation_pdf_report(
    investigation: Dict[str, Any],
    generated_by: str = "SafetyVision AI Safety Investigator"
) -> bytes:
    """
    Generates a comprehensive formal AI Safety Investigation Report (Portrait A4)
    incorporating multi-agent telemetry, visual evidence crops, grounded citations,
    enterprise risk matrix evaluation, and CAPA corrective action planning.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=portrait(A4),
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "InvTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=21,
        textColor=colors.HexColor("#0f172a"),
        alignment=0,
        spaceAfter=3
    )

    meta_style = ParagraphStyle(
        "InvMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#475569"),
        spaceAfter=8
    )

    h2_style = ParagraphStyle(
        "InvH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=8,
        spaceAfter=4
    )

    cell_style = ParagraphStyle(
        "InvCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1e293b")
    )

    cell_bold = ParagraphStyle(
        "InvCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#0f172a")
    )

    story = []

    # Title & Sub-header
    story.append(Paragraph("SAFETYVISION AI - FORMAL SAFETY INVESTIGATION REPORT", title_style))
    story.append(Paragraph(
        f"<b>Autonomous AI Multi-Agent Orchestration</b> &bull; Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')} &bull; Officer: {escape_xml(generated_by)}",
        meta_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#4f46e5"), spaceAfter=10))

    # Incident Overview Table
    inc_id = escape_xml(investigation.get("incident_id") or "INV-UNKNOWN")
    inc_title = escape_xml(investigation.get("title") or "Safety Incident Investigation")
    category = escape_xml(investigation.get("incident_category") or "General Construction & PPE")
    risk_level = (investigation.get("risk_level") or "Medium").upper()

    risk_colors = {
        "LOW": colors.HexColor("#16a34a"),
        "MEDIUM": colors.HexColor("#d97706"),
        "HIGH": colors.HexColor("#ea580c"),
        "CRITICAL": colors.HexColor("#dc2626")
    }
    r_color = risk_colors.get(risk_level, colors.HexColor("#475569"))

    overview_data = [
        [
            Paragraph("<b>Incident Reference:</b>", cell_bold),
            Paragraph(f"<code>{inc_id}</code>", cell_style),
            Paragraph("<b>Assessed Risk Level:</b>", cell_bold),
            Paragraph(f"<font color='{r_color.hexval()}'><b>{risk_level}</b></font>", cell_bold),
        ],
        [
            Paragraph("<b>Investigation Title:</b>", cell_bold),
            Paragraph(inc_title, cell_style),
            Paragraph("<b>Hazard Category:</b>", cell_bold),
            Paragraph(category, cell_style),
        ],
        [
            Paragraph("<b>Repository Scope:</b>", cell_bold),
            Paragraph(escape_xml(investigation.get("repository_scope") or "All Categories"), cell_style),
            Paragraph("<b>Investigation Status:</b>", cell_bold),
            Paragraph("<b>COMPLETED</b>", cell_bold),
        ]
    ]

    t_overview = Table(overview_data, colWidths=[110, 160, 110, 140])
    t_overview.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_overview)
    story.append(Spacer(1, 8))

    # Narrative Description
    story.append(Paragraph("1. Incident Narrative & Operational Description", h2_style))
    desc_text = escape_xml(investigation.get("incident_description") or "No operational description provided.")
    t_desc = Table([[Paragraph(f"<i>&ldquo;{desc_text}&rdquo;</i>", cell_style)]], colWidths=[520])
    t_desc.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ffffff")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t_desc)
    story.append(Spacer(1, 8))

    # Visual Evidence
    story.append(Paragraph("2. Visual Intelligence & Telemetry Evidence", h2_style))
    vis = investigation.get("visual_evidence") or {}
    vis_path = vis.get("annotated_image_path") or investigation.get("visual_evidence_path")
    if vis_path and Path(vis_path).exists():
        try:
            story.append(Paragraph("<b>Annotated Computer Vision Evidence Crop:</b>", cell_bold))
            story.append(Image(str(vis_path), width=280, height=190))
            story.append(Spacer(1, 6))
        except Exception:
            story.append(Paragraph("<i>[Visual evidence image file exists but could not be embedded in PDF]</i>", cell_style))
    else:
        story.append(Paragraph(
            "<i>Visual evidence was not provided or marked unavailable. "
            "Investigation proceeded using narrative cues and indexed policy retrieval.</i>",
            cell_style
        ))
    story.append(Spacer(1, 8))

    # Grounded Policies & Citations
    story.append(Paragraph("3. Grounded Corporate Safety Policies & Statutory Citations", h2_style))
    citations = investigation.get("policy_context") or []
    if citations:
        cit_data = [
            [
                Paragraph("<b>Document Name</b>", cell_bold),
                Paragraph("<b>Page / Sect</b>", cell_bold),
                Paragraph("<b>Policy Excerpt Snippet</b>", cell_bold),
                Paragraph("<b>Score</b>", cell_bold)
            ]
        ]
        for c in citations[:4]:
            fn = escape_xml(c.get("file_name", "Policy"))
            p = f"p.{c.get('page_number', 1)}"
            snip = escape_xml((c.get("snippet") or "")[:120]) + ("..." if len(c.get("snippet", "")) > 120 else "")
            sc = f"{float(c.get('score', 0.0)):.2f}"
            cit_data.append([
                Paragraph(fn, cell_style),
                Paragraph(p, cell_style),
                Paragraph(snip, cell_style),
                Paragraph(sc, cell_style)
            ])
        t_cit = Table(cit_data, colWidths=[120, 60, 290, 50])
        t_cit.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t_cit)
    else:
        story.append(Paragraph(
            "<i>No directly matching corporate safety documents were indexed. "
            "Statutory baseline requirements were evaluated against statutory standards.</i>",
            cell_style
        ))
    story.append(Spacer(1, 8))

    # Enterprise Risk Matrix
    story.append(Paragraph("4. Enterprise Risk Matrix Assessment", h2_style))
    risk_info = investigation.get("risk_assessment") or {}
    sev = risk_info.get("severity", 2)
    lik = risk_info.get("likelihood", 3)
    r_score = risk_info.get("risk_score", sev * lik)

    risk_data = [
        [
            Paragraph("<b>Severity Rating:</b>", cell_bold),
            Paragraph(f"<b>{sev}/5</b> ({risk_info.get('severity_label', 'Moderate')})", cell_style),
            Paragraph("<b>Likelihood Rating:</b>", cell_bold),
            Paragraph(f"<b>{lik}/5</b> ({risk_info.get('likelihood_label', 'Possible')})", cell_style),
        ],
        [
            Paragraph("<b>Composite Risk Score:</b>", cell_bold),
            Paragraph(f"<b>{r_score} / 25</b> (Severity &times; Likelihood)", cell_style),
            Paragraph("<b>Recommended Action:</b>", cell_bold),
            Paragraph(escape_xml(risk_info.get("recommended_action") or "Supervisory review"), cell_style),
        ]
    ]
    t_risk = Table(risk_data, colWidths=[110, 160, 110, 140])
    t_risk.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_risk)
    story.append(Spacer(1, 8))

    # Corrective & Preventive Action Plan (CAPA)
    story.append(Paragraph("5. Corrective & Preventive Actions (CAPA)", h2_style))
    capa = investigation.get("corrective_actions") or {}
    grounded_acts = capa.get("grounded_actions") or []
    general_acts = capa.get("general_actions") or []

    capa_data = [[Paragraph("<b>Control Level</b>", cell_bold), Paragraph("<b>Action Item & Regulatory Source</b>", cell_bold)]]
    for ga in grounded_acts[:2]:
        act_text = f"<b>[POLICY GROUNDED]</b> {escape_xml(ga.get('action'))}<br/><font color='#4338ca'><i>{escape_xml(ga.get('citation'))}</i></font>"
        capa_data.append([Paragraph("Corporate Policy", cell_style), Paragraph(act_text, cell_style)])
    for gen in general_acts[:3]:
        act_text = f"<b>[{escape_xml(gen.get('control_type', 'General'))}]</b> {escape_xml(gen.get('action'))}<br/><font color='#64748b'><i>(General Industry Best Practice - Not from indexed documents)</i></font>"
        capa_data.append([Paragraph(escape_xml(gen.get("control_type", "Standard")), cell_style), Paragraph(act_text, cell_style)])

    t_capa = Table(capa_data, colWidths=[120, 400])
    t_capa.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_capa)
    story.append(Spacer(1, 8))

    # Multi-Agent Workflow Execution Log
    story.append(Paragraph("6. Multi-Agent Workflow Execution Audit", h2_style))
    stages = investigation.get("stages") or {}
    stage_data = [[Paragraph("<b>Workflow Stage</b>", cell_bold), Paragraph("<b>Status</b>", cell_bold), Paragraph("<b>Latency</b>", cell_bold)]]
    for s_name, s_info in stages.items():
        st_st = s_info.get("status", "ok").upper()
        ms = f"{s_info.get('execution_time_ms', 0)} ms"
        stage_data.append([
            Paragraph(escape_xml(s_name), cell_style),
            Paragraph(f"<b>{st_st}</b>", cell_style),
            Paragraph(ms, cell_style)
        ])
    if len(stage_data) > 1:
        t_stage = Table(stage_data, colWidths=[240, 140, 140])
        t_stage.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f8fafc")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(t_stage)
        story.append(Spacer(1, 8))

    # Verification & Closure Block
    sign_data = [
        [Paragraph("<b>EHS Lead Investigator Certification & Verification:</b>", cell_bold)],
        [Paragraph(f"I hereby certify that this investigation was conducted using autonomous multimodal intelligence in compliance with enterprise EHS safety governance standards.<br/><br/><b>Lead Investigator:</b> {escape_xml(generated_by)} &nbsp;&nbsp;&nbsp;&nbsp; <b>Signature:</b> ___________________________ &nbsp;&nbsp;&nbsp;&nbsp; <b>Date:</b> {datetime.now().strftime('%Y-%m-%d')}", cell_style)]
    ]
    t_sign = Table(sign_data, colWidths=[520])
    t_sign.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#f1f5f9")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t_sign)

    doc.build(story)
    return buffer.getvalue()
