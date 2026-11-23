"""
Detecto AI - Document Intelligence: Enterprise EHS Knowledge Base Seeder
Milestone 2: Populates the full 4-document Enterprise EHS Knowledge Base:
1. OSHA_1910_General_Industry_PPE_Standard.pdf (Regulatory statutory benchmark)
2. CCTV_Overhead_Crane_Inspection_Audit_Report.csv (16+ tabular inspection records with coordinates and CAPA)
3. Chemical_Coating_Line_Safety_SOP_Rev4.docx (Zone Z-02 chemical splash & 10s eyewash rule)
4. Logistics_Forklift_Pedestrian_Safety_Manual.pdf (Class 2/3 high-vis vests, 5 km/h limits, walkways)
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json

from database import (
    list_documents, create_document, insert_document_chunks,
    update_document_status, update_document_metadata, delete_document,
    get_connection
)
from services.document_intelligence.uploader import get_storage_directory
from services.document_intelligence.vector_store import get_vector_store
from services.document_intelligence.extractor import extract_ehs_entities, generate_document_summary


# ==========================================
# 1. OSHA 1910 GENERAL INDUSTRY PPE STANDARD
# ==========================================
OSHA_1910_STANDARD_TEXT = """
UNITED STATES DEPARTMENT OF LABOR - OCCUPATIONAL SAFETY & HEALTH ADMINISTRATION
STANDARD: 29 CFR 1910 - GENERAL INDUSTRY PERSONAL PROTECTIVE EQUIPMENT
Document Code: OSHA-1910-FED-STD-2026 | Statutory Regulatory Authority

SECTION 1: 1910.132 GENERAL REQUIREMENTS FOR PERSONAL PROTECTIVE EQUIPMENT (PPE)
1.1 Application: Protective equipment, including personal protective equipment for eyes, face, head, and extremities, protective clothing, respiratory devices, and protective shields and barriers, shall be provided, used, and maintained in a sanitary and reliable condition wherever it is necessary by reason of hazards of processes or environment, chemical hazards, radiological hazards, or mechanical irritants encountered in a manner capable of causing injury or impairment in the function of any part of the body through absorption, inhalation or physical contact.
1.2 Hazard Assessment & Equipment Selection: The employer shall assess the workplace to determine if hazards are present, or are likely to be present, which necessitate the use of personal protective equipment (PPE). If such hazards are present, or likely to be present, the employer shall select, and have each affected employee use, the types of PPE that will protect the affected employee from the hazards identified in the hazard assessment.
1.3 Defective and Damaged Equipment: Defective or damaged personal protective equipment shall not be used. Any item failing integrity standards must be immediately removed from service, quarantined, and destroyed.
1.4 Training Requirements: The employer shall provide training to each employee who is required by this section to use PPE. Each such employee shall be trained to know at least the following: when PPE is necessary; what PPE is necessary; how to properly don, doff, adjust, and wear PPE; the limitations of the PPE; and the proper care, maintenance, useful life and disposal of the PPE.

SECTION 2: 1910.135 HEAD PROTECTION & INDUSTRIAL HARDHAT STANDARDS (ANSI Z89.1)
2.1 General Requirements: The employer shall ensure that each affected employee wears a protective helmet when working in areas where there is a potential for injury to the head from falling objects, flying particles, or electrical shock.
2.2 Consensus Standard Compliance: Head protection must comply with ANSI Z89.1 ("American National Standard for Industrial Head Protection"):
    - Type I helmets are designed to reduce the force of impact resulting only from a blow only to the top of the head.
    - Type II helmets are designed to reduce the force of impact resulting from a blow which may be received on the top or lateral sides of the head.
    - Class G (General) helmets are tested to withstand 2,200 volts.
    - Class E (Electrical) helmets are tested to withstand 20,000 volts.
    - Class C (Conductive) helmets offer zero electrical protection and are strictly prohibited near electrical substations or live panels.
2.3 Impact Protocol & Useful Life: Any hardhat subjected to a severe impact, dropped from a height exceeding 3 meters, or exhibiting visible hairline cracking or chalking must be immediately removed from service and destroyed. Shells must be routinely replaced every 3 years; internal suspension harnesses must be replaced every 12 months.

SECTION 3: 1910.133 EYE AND FACE PROTECTION (ANSI Z87.1)
3.1 General Requirements: The employer shall ensure that each affected employee uses appropriate eye or face protection when exposed to eye or face hazards from flying particles, molten metal, liquid chemicals, acids or caustic liquids, chemical gases or vapors, or potentially injurious light radiation.
3.2 Side Protection: The employer shall ensure that each affected employee uses eye protection that provides side protection when there is a hazard from flying objects. Detachable side protectors (e.g., clip-on or slide-on side shields) meeting the pertinent requirements of this section are acceptable.
3.3 Prescription Lenses: The employer shall ensure that each affected employee who wears prescription lenses while engaged in operations that involve eye hazards wears eye protection that incorporates the prescription in its design, or wears eye protection that can be worn over the prescription lenses without disturbing the proper position of the prescription lenses or the protective lenses.
3.4 Engineering Markings: Eye and face protective gear must carry the permanent 'Z87+' impact certification stamp. Chemical splash operations require ANSI Z87.1 D3 indirect-vent splash goggles.

SECTION 4: 1910.136 OCCUPATIONAL FOOT PROTECTION (ASTM F2412 & ASTM F2413)
4.1 General Requirements: The employer shall ensure that each affected employee uses protective footwear when working in areas where there is a danger of foot injuries due to falling or rolling objects, or objects piercing the sole, or when the use of protective footwear will protect the affected employee from an electrical hazard, such as a static-discharge or electric-shock hazard, that remains after the employer takes other necessary protective measures.
4.2 Performance Criteria: Protective footwear must comply with ASTM F2413-18. Impact resistance must be rated at least I/75 (delivering protection against 75 foot-pounds of impact), and compression resistance must be rated at least C/75 (withstanding compressive loads up to 2,500 pounds).
4.3 Puncture Resistance: In demolition, staging, and recycling corridors where sharp objects are present, footwear equipped with steel or Kevlar puncture-resistant midsoles tested to withstand 270 lbs of penetration force is mandatory.

SECTION 5: 1910.138 HAND PROTECTION (ANSI/ISEA 105)
5.1 General Requirements: Employers shall select and require employees to use appropriate hand protection when employees' hands are exposed to hazards such as those from skin absorption of harmful substances; severe cuts or lacerations; severe abrasions; punctures; chemical burns; thermal burns; and harmful temperature extremes.
5.2 Selection Framework: Employers shall base the selection of the appropriate hand protection on an evaluation of the performance characteristics of the hand protection relative to the task(s) to be performed, conditions present, duration of use, and the hazards and potential hazards identified. ANSI/ISEA 105 Level A4+ cut-resistant Kevlar gloves are mandatory for metal fabrication. Chemical operations require verified permeation breakthrough times exceeding 120 minutes.

SECTION 6: ISO 45001 OCCUPATIONAL HEALTH & SAFETY & STATUTORY PENALTIES
6.1 Hierarchy of Controls (ISO 45001 Clause 8.1.2): Organizations must apply controls in the order: Elimination -> Substitution -> Engineering Controls -> Administrative Controls -> Personal Protective Equipment (PPE). PPE is the final barrier of protection, not the primary safety control.
6.2 Statutory Enforcement: OSHA serious citations carry federal civil penalties up to $16,131 per violation. Failure to abate carries penalties of up to $16,131 per day until compliance is verified.
"""


# ==========================================
# 2. CCTV OVERHEAD CRANE INSPECTION AUDIT REPORT (CSV)
# ==========================================
OVERHEAD_CRANE_CSV_TEXT = """Audit_ID,Timestamp,Facility_Zone,Component_Inspected,PPE_Audit_Check,Defect_Finding,Severity_Rating,Clearance_Radius_Status,CAPA_Ticket_Required,Immediate_Action_Taken
AUD-CRANE-101,2026-10-15 08:30:00,Zone Z-04 Overhead Crane Bay,Hoist Wire Rope,NO-HELMET,Rigging technician spotted beneath boom with no hardhat,CRITICAL,5m stand-down zone breached,CAPA-CR-401,Work halted immediately. Technician evacuated and issued ANSI Z89.1 Type I hardhat.
AUD-CRANE-102,2026-10-15 09:45:00,Zone Z-04 Overhead Crane Bay,Crane Hook Safety Latch,COMPLIANT,Safety tongue latch spring broken on 10-ton south hoist hook,HIGH,5m zone maintained,CAPA-CR-402,Hook tagged out of service. Lockout/Tagout applied until spring replacement.
AUD-CRANE-103,2026-10-16 11:10:00,Zone Z-01 Heavy Fabrication,Gantry Rail Crane A,NO-VEST,Floor spotter guiding steel I-beam without high-vis apparel,HIGH,3m standoff breached,CAPA-CR-403,Spotter removed from line. Issued ANSI/ISEA 107 Class 2 high-vis reflective vest.
AUD-CRANE-104,2026-10-17 14:00:00,Zone Z-04 Overhead Crane Bay,Synthetic Rigging Slings,COMPLIANT,2-inch nylon web sling has visible edge tears exceeding 1/4 inch,CRITICAL,5m zone clear,CAPA-CR-404,Sling condemned and sliced in half on site per OSHA 1910.184 rules.
AUD-CRANE-105,2026-10-18 16:20:00,Zone Z-01 Heavy Fabrication,Overhead Crane Control Pendant,NO-GLOVES,Operator handling pendant cable and guide rope without gloves,MEDIUM,Safe clearance,CAPA-CR-405,Work paused. Operator issued ANSI Level A4 Kevlar cut-resistant gloves.
AUD-CRANE-106,2026-10-19 08:15:00,Zone Z-04 Overhead Crane Bay,Bridge Travel Brakes,COMPLIANT,Bridge motor drift exceeds 6 inches on 5-ton travel stop,HIGH,Safe zone maintained,CAPA-CR-406,Brake shoes adjusted and torque recalibrated by certified crane technician.
AUD-CRANE-107,2026-10-20 10:30:00,Zone Z-04 Overhead Crane Bay,Crane Hook B,NO-BOOTS,Contractor assisting rigging wearing lightweight running shoes,HIGH,Stand-down radius breached,CAPA-CR-407,Contractor escorted off manufacturing floor until ASTM F2413 boots donned.
AUD-CRANE-108,2026-10-21 13:45:00,Zone Z-02 Chemical Coating,Monorail Dip Hoist,NO-GOGGLES,Acid tank dip hoist operator wearing ordinary prescription glasses,CRITICAL,10s eyewash accessible,CAPA-CR-408,Immediate stop work order. Operator fitted with ANSI Z87.1 D3 chemical splash goggles.
AUD-CRANE-109,2026-10-22 15:50:00,Zone Z-04 Overhead Crane Bay,Limit Switch Upper Hoist,COMPLIANT,Paddle limit switch failed to cut power at upper travel limit,CRITICAL,Exclusion zone active,CAPA-CR-409,Hoist locked out. Electrical limit switch wiring replaced and tested.
AUD-CRANE-110,2026-10-23 09:20:00,Zone Z-03 Logistics Staging,Mobile Yard Crane C,NO-VEST,Truck driver walking across active staging lane with no vest,HIGH,5m crane radius breached,CAPA-CR-410,Driver restricted to cab; Class 3 high-vis reflective vest issued before exit.
AUD-CRANE-111,2026-10-24 11:35:00,Zone Z-04 Overhead Crane Bay,Wire Rope Drum Sheaves,COMPLIANT,Sheave groove corrugated causing heavy wire rope distortion,MEDIUM,Safe clearance,CAPA-CR-411,Sheave re-grooved and inspected during scheduled weekend maintenance.
AUD-CRANE-112,2026-10-25 14:15:00,Zone Z-01 Heavy Fabrication,Heavy Jib Crane 2,NO-HELMET,Welder apprentice grinding under jib boom without hardhat,HIGH,3m radius breached,CAPA-CR-412,Welder ordered to don ANSI Z89.1 hardhat with integrated grinding face shield.
AUD-CRANE-113,2026-10-26 16:40:00,Zone Z-04 Overhead Crane Bay,Motion Warning Horn & Strobe,COMPLIANT,Motion siren audio level measured at 74 dB (below 85 dB requirement),MEDIUM,Safe clearance,CAPA-CR-413,High-intensity dual siren and 360-degree amber strobe installed on crane trolley.
AUD-CRANE-114,2026-10-27 10:05:00,Zone Z-03 Logistics Staging,Rigging Hardware Shackles,COMPLIANT,3/4-inch bow shackle missing manufacturer WLL rated capacity stamp,HIGH,Safe zone maintained,CAPA-CR-414,Non-rated shackle confiscated and destroyed; replaced with Crosby forged alloy shackle.
AUD-CRANE-115,2026-10-28 13:20:00,Zone Z-04 Overhead Crane Bay,Suspended Load Clearance,NO-VEST,Maintenance apprentice stepped inside 5-meter exclusion radius under 8-ton load,CRITICAL,5m zone violated,CAPA-CR-415,Emergency horn sounded. Crane halted. Apprentice reassigned to mandatory safety retraining.
AUD-CRANE-116,2026-10-29 15:30:00,Zone Z-01 Heavy Fabrication,Runaway End Stop Bumpers,COMPLIANT,End-stop polyurethane rubber bumper sheared off north rail,HIGH,Clear,CAPA-CR-416,Heavy structural polyurethane end-stop buffer welded and load-tested.
"""


# ==========================================
# 3. CHEMICAL COATING LINE SAFETY SOP REV 4 (DOCX)
# ==========================================
CHEMICAL_COATING_SOP_TEXT = """
DETECTO AI ENTERPRISE HEALTH & SAFETY STANDARD OPERATING PROCEDURE
DOCUMENT CONTROL: DET-SOP-CHM-2026-R4 | REVISION: 4.0 | STATUS: APPROVED
EFFECTIVE ZONE: ZONE Z-02 (CHEMICAL COATING, ETCHING & ACID SPRAY BAY)
GOVERNING STANDARDS: OSHA 29 CFR 1910.133, 1910.138, ANSI Z87.1, ANSI Z358.1, ISO 45001

SECTION 1: PURPOSE AND APPLICABILITY
1.1 Purpose: This Standard Operating Procedure defines mandatory protocols for chemical baths, sulfuric acid etching, solvent degreasing, and electrostatic spray coating in Zone Z-02.
1.2 Scope: Applicable to all process operators, maintenance technicians, chemical handlers, and quality inspectors entering Zone Z-02. Zero exceptions permitted.

SECTION 2: MANDATORY CHEMICAL PPE MATRIX
2.1 Eye & Face Protection: Standard safety spectacles are strictly prohibited in Zone Z-02. All personnel within 10 meters of chemical baths must wear ANSI Z87.1-2020 D3 indirect-vent chemical splash goggles paired with a full polycarbonate face shield (minimum 8-inch length).
2.2 Hand Protection:
    - Chemical Transfer & Quality Testing: Nitrile gloves (minimum 8 mil thickness) with textured grip.
    - Bath Immersion & Tank Cleaning: Heavy butyl or neoprene gauntlet gloves (minimum 18 mil thickness, 14-inch sleeve) with verified chemical permeation breakthrough times exceeding 120 minutes against sulfuric acid (30%) and methyl ethyl ketone.
2.3 Body & Foot Protection: Chemical-resistant neoprene/PVC aprons covering from neck to below knees. ASTM F2413 protective boots with acid-resistant Vibram soles and composite safety toe caps.

SECTION 3: EMERGENCY EYEWASH AND SAFETY SHOWER COMPLIANCE (ANSI Z358.1)
3.1 10-Second Unobstructed Access Rule: In strict compliance with ANSI Z358.1, certified plumbed emergency eyewash and deluge shower stations must be located within a 10-second unobstructed travel path (maximum 16 meters / 55 feet distance) from all chemical tanks.
3.2 Flow Rate Specifications: Eyewash units must deliver a continuous minimum flow of 0.4 gallons per minute (1.5 L/min) of tepid water (60°F to 100°F / 16°C to 38°C) for a minimum duration of 15 minutes.
3.3 Weekly Flow Inspection Protocol: Facilities engineering must activate each emergency eyewash station every Monday at 07:00 to flush sediment, inspect spray nozzles, and log flow rate telemetry. Stations blocked by pallets, barrels, or equipment constitute a Level 1 Critical Violation subject to an immediate stop-work order.

SECTION 4: ACCIDENTAL SPLASH & EXPOSURE PROTOCOL
4.1 Immediate Actions:
    - Step 1: Immediately escort the exposed individual to the nearest emergency eyewash station within 10 seconds.
    - Step 2: Hold eyelids forcibly open and irrigate both eyes continuously with tepid water for a minimum of 15 minutes. Do not use neutralizing agents.
    - Step 3: Remove all contaminated clothing, aprons, and gloves while irrigation is ongoing.
    - Step 4: Dispatch Code Blue Medical Emergency on radio channel 3 and notify Site EHS Lead.
    - Step 5: Secure Safety Data Sheet (SDS) for the specific chemical agent and transfer with the emergency response crew to hospital.

SECTION 5: VENTILATION AND SPILL CONTAINMENT
5.1 Local Exhaust Ventilation (LEV): Chemical dip tanks must operate with slot exhaust hoods maintaining a minimum capture velocity of 100 feet per minute (fpm) across the tank surface. If exhaust airflow drops below 85 fpm, automated interlocks must halt chemical transfer.
5.2 Secondary Containment: All chemical storage tanks must have 110% secondary containment berms. Acid-resistant spill response kits equipped with sodium bicarbonate neutralizer must be stationed every 20 meters.
"""


# ==========================================
# 4. LOGISTICS FORKLIFT & PEDESTRIAN SAFETY MANUAL (PDF)
# ==========================================
LOGISTICS_FORKLIFT_MANUAL_TEXT = """
DETECTO AI ENTERPRISE LOGISTICS & MATERIAL HANDLING SAFETY MANUAL
DOCUMENT ID: DET-SAF-LOG-2026-V1 | EFFECTIVE: JANUARY 2026
EFFECTIVE ZONES: ZONE Z-03 (LOGISTICS DOCKS) & ZONE Z-05 (PEDESTRIAN WALKWAYS)
GOVERNING STANDARDS: OSHA 29 CFR 1910.178, ANSI/ISEA 107-2020, ISO 45001:2018

SECTION 1: HIGH-VISIBILITY SAFETY APPAREL POLICY (ANSI/ISEA 107)
1.1 Minimum Garment Classifications:
    - General Warehouse Personnel & Walkway Transits: ANSI/ISEA 107 Class 2 high-visibility vests featuring fluorescent yellow-green background material and retroreflective striping.
    - Active Loading Docks, Yard Corridors, & Forklift Spotters: ANSI/ISEA 107 Class 3 high-visibility garments with full sleeves and 360-degree retroreflective striping are mandatory.
1.2 Vest Integrity & Fastening: High-visibility garments must be fully zipped or snapped closed at all times. Open or flapping vests create severe snag hazards on conveyor rollers and machinery. Soiled, faded, or torn vests must be surrendered to safety stores and replaced immediately.

SECTION 2: PEDESTRIAN TRAFFIC SEGREGATION & CLEARANCE STANDARDS
2.1 Painted Safety Walkways (Zone Z-05): Pedestrians must walk exclusively within designated safety aisles painted with durable safety green borders and high-contrast yellow hatched stripes.
2.2 Forklift Standoff Distance: Pedestrians must maintain a minimum 3-meter (10-foot) clearance radius from operating forklifts. When a forklift is lifting or lowering mast loads, the exclusion radius increases to 5 meters.
2.3 Maximum Operational Speed Limits:
    - Shared Pedestrian Corridors & Crossings: Maximum speed limit is 5 km/h (3 mph - walking pace).
    - Open Logistics Yard & Unobstructed Aisles: Maximum speed limit is 12 km/h (7.5 mph).
2.4 Blind Corner Protocols: Forklift drivers must come to a complete stop, sound their horn, and verify pedestrian clearance before proceeding through any blind doorway or aisle intersection.

SECTION 3: POWERED INDUSTRIAL TRUCK REGULATIONS (OSHA 1910.178)
3.1 Operator Licensing & Certification: Only personnel who have completed Detecto AI OSHA 1910.178 certification and hold a valid physical badge are authorized to operate forklifts. Licenses expire every 36 months.
3.2 Daily Pre-Shift Vehicle Inspection: Operators must complete the 24-point pre-shift inspection log before powering on any forklift: hydraulic fluid levels, tire pressure, mast chains, backup beepers, headlights, and horn function. Any defective vehicle must be tagged out immediately.
3.3 Safety Interlocks & Blue Spotlight: All forklifts must be equipped with forward and reverse blue safety spotlights projecting a warning beam onto the floor 5 meters ahead of travel direction. Backup audible beepers (minimum 85 dB) must be permanently engaged.
3.4 Operator Restraints: Seatbelts must be buckled at all times while the ignition is on. In the event of a vehicle tip-over, the operator must hold onto the steering wheel, brace their feet, and lean away from the point of impact.

SECTION 4: LOADING DOCK SAFETY & TRAILER WHEEL CHOCKING
4.1 Mechanical Dock Restraints: Tractor-trailers backed into loading bays must be secured with automatic dock hook restraints. In bays without automated hooks, two heavy rubber wheel chocks must be placed firmly against the rear wheels of the trailer before forklift loading commences.
4.2 Dock Leveler Overlap: Dock leveler lips must extend at least 4 inches into the trailer floor before forklifts traverse. Maximum dock gap permissible is 1 inch.
4.3 Edge Protection: Overhead dock bay doors not occupied by a trailer must have visual safety chains or automated yellow drop-down gates in place to prevent vehicles or pedestrians falling over the 4-foot dock ledge.
"""


def load_sample_ehs_library(force: bool = False) -> Dict[str, Any]:
    """
    Populates and indexes the complete 4-document Enterprise EHS Knowledge Base:
    1. OSHA 1910 General Industry PPE Standard (.pdf)
    2. CCTV Overhead Crane Inspection Audit Report (.csv)
    3. Chemical Coating Line Safety SOP Rev4 (.docx)
    4. Logistics Forklift & Pedestrian Safety Manual (.pdf)
    Returns summary statistics of the loaded library.
    """
    existing_docs = list_documents(limit=100)
    existing_names = {d["file_name"] for d in existing_docs}

    storage = get_storage_directory()
    vstore = get_vector_store()

    specs = [
        {
            "file_name": "OSHA_1910_General_Industry_PPE_Standard.pdf",
            "file_type": "pdf",
            "text": OSHA_1910_STANDARD_TEXT.strip(),
            "section_delimiter": "SECTION ",
            "default_section": "OSHA 1910 General Industry PPE Standard"
        },
        {
            "file_name": "CCTV_Overhead_Crane_Inspection_Audit_Report.csv",
            "file_type": "csv",
            "text": OVERHEAD_CRANE_CSV_TEXT.strip(),
            "section_delimiter": None,
            "default_section": "Tabular Overhead Crane Inspection Audit Record"
        },
        {
            "file_name": "Chemical_Coating_Line_Safety_SOP_Rev4.docx",
            "file_type": "docx",
            "text": CHEMICAL_COATING_SOP_TEXT.strip(),
            "section_delimiter": "SECTION ",
            "default_section": "Chemical Coating Line Safety SOP Rev 4"
        },
        {
            "file_name": "Logistics_Forklift_Pedestrian_Safety_Manual.pdf",
            "file_type": "pdf",
            "text": LOGISTICS_FORKLIFT_MANUAL_TEXT.strip(),
            "section_delimiter": "SECTION ",
            "default_section": "Logistics Forklift & Pedestrian Safety Manual"
        }
    ]

    loaded_count = 0
    total_chunks_added = 0

    for s in specs:
        fname = s["file_name"]
        if not force and fname in existing_names:
            continue

        # If forcing reload of existing document, clean it out first
        if force and fname in existing_names:
            for old_doc in existing_docs:
                if old_doc["file_name"] == fname:
                    delete_document(old_doc["id"])
                    vstore.delete_document(old_doc["id"])

        # Write genuine document artifact to storage
        doc_path = storage / fname
        try:
            if s["file_type"] == "pdf":
                from reportlab.lib.pagesizes import letter
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib import colors

                doc_pdf = SimpleDocTemplate(
                    str(doc_path),
                    pagesize=letter,
                    rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54
                )
                styles = getSampleStyleSheet()
                t_style = ParagraphStyle(
                    'TitleStyle',
                    parent=styles['Heading1'],
                    fontSize=13,
                    textColor=colors.HexColor('#0F2C59'),
                    spaceAfter=10
                )
                h_style = ParagraphStyle(
                    'HeadingStyle',
                    parent=styles['Heading2'],
                    fontSize=10.5,
                    textColor=colors.HexColor('#0284C7'),
                    spaceBefore=8,
                    spaceAfter=4
                )
                b_style = ParagraphStyle(
                    'BodyStyle',
                    parent=styles['Normal'],
                    fontSize=8.5,
                    leading=12,
                    textColor=colors.HexColor('#1E293B'),
                    spaceAfter=4
                )
                story = [Paragraph(s["default_section"], t_style), Spacer(1, 8)]
                for line in s["text"].strip().split("\n"):
                    l_str = line.strip()
                    if not l_str:
                        continue
                    if l_str.startswith("SECTION "):
                        story.append(Paragraph(l_str, h_style))
                    else:
                        safe_l = l_str.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                        story.append(Paragraph(safe_l, b_style))
                doc_pdf.build(story)

            elif s["file_type"] == "docx":
                import docx
                doc_word = docx.Document()
                doc_word.add_heading(s["default_section"], level=0)
                for line in s["text"].strip().split("\n"):
                    l_str = line.strip()
                    if not l_str:
                        continue
                    if l_str.startswith("SECTION "):
                        doc_word.add_heading(l_str, level=1)
                    else:
                        doc_word.add_paragraph(l_str)
                doc_word.save(str(doc_path))

            else:
                with open(doc_path, "w", encoding="utf-8") as f:
                    f.write(s["text"])
        except Exception:
            with open(doc_path, "w", encoding="utf-8") as f:
                f.write(s["text"])

        # Extract structured entities
        entities = extract_ehs_entities(s["text"])
        raw_items = []

        if s["file_type"] == "csv":
            csv_lines = [l.strip() for l in s["text"].split("\n") if l.strip()]
            headers = csv_lines[0].split(",")
            for r_idx, line in enumerate(csv_lines[1:], start=2):
                parts = line.split(",")
                row_str = " | ".join(f"{h.strip()}: {p.strip()}" for h, p in zip(headers, parts) if h)
                raw_items.append({
                    "text_content": f"Overhead Crane Audit (Row {r_idx}): {row_str}",
                    "page_number": None,
                    "row_number": r_idx,
                    "section_title": f"Crane Audit Record (Row {r_idx})",
                    "token_count": len(line.split())
                })
        else:
            # Section based splitting
            raw_sections = s["text"].split(s["section_delimiter"])
            for s_idx, sec_text in enumerate(raw_sections[1:], start=1):
                lines = [l.strip() for l in sec_text.strip().split("\n") if l.strip()]
                title = "Section " + lines[0] if lines else s["default_section"]
                body = "\n".join(lines[1:]).strip() if len(lines) > 1 else lines[0]
                raw_items.append({
                    "text_content": f"[{title}]\n{body}",
                    "page_number": s_idx,
                    "row_number": None,
                    "section_title": title,
                    "token_count": len(body.split())
                })

        # Register document in database
        doc_summary = generate_document_summary(raw_items, entities)
        meta_json = json.dumps(entities)

        doc_id = create_document(
            file_name=fname,
            file_type=s["file_type"],
            file_size=len(s["text"].encode("utf-8")),
            file_path=str(doc_path),
            status="indexed",
            extraction_method="dual_path_digital",
            summary=doc_summary,
            metadata_json=meta_json
        )

        # Assemble final chunks
        final_chunks = []
        for c_idx, item in enumerate(raw_items):
            final_chunks.append({
                "document_id": doc_id,
                "file_name": fname,
                "chunk_index": c_idx,
                "text_content": item["text_content"],
                "page_number": item.get("page_number"),
                "row_number": item.get("row_number"),
                "section_title": item.get("section_title") or s["default_section"],
                "token_count": item.get("token_count", 50)
            })

        # Insert chunks and index in FAISS
        insert_document_chunks(doc_id, final_chunks)
        vstore.add_document_chunks(final_chunks)
        update_document_status(doc_id, "indexed", chunk_count=len(final_chunks), page_count=len(final_chunks))

        loaded_count += 1
        total_chunks_added += len(final_chunks)

    return {
        "loaded_documents": loaded_count,
        "total_chunks_added": total_chunks_added,
        "total_repository_documents": len(list_documents(limit=100))
    }


def seed_default_knowledge_base():
    """Default startup entry point: Ensures all 4 enterprise documents exist and are indexed."""
    existing_docs = list_documents(limit=100)
    vstore = get_vector_store()
    if len(existing_docs) < 4 or vstore.total_chunks == 0:
        load_sample_ehs_library(force=True)
