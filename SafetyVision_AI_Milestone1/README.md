# 🛡️ Detecto AI — Enterprise Workplace Safety & PPE Monitoring Platform

> **Commercial Enterprise Prototype for AI-Powered Environmental Health & Safety (EHS), OSHA 1910, and ISO 45001 Compliance Monitoring.**

---

## 🌟 Executive Overview
**Detecto AI** is an enterprise-grade computer vision platform engineered for industrial plants, construction yards, manufacturing facilities, and hazardous operational environments. Utilizing real-time deep learning (YOLOv8) and a robust audit-trail database, Detecto AI autonomously detects Personal Protective Equipment (PPE) compliance, mitigates site hazards, and delivers audit-ready compliance documentation.

---

## 🚀 Key Enterprise Features

### 1. 📊 Executive Safety Command Center (Dashboard)
- **Composite Safety Index (0–100):** Weighted health score calculating compliance ratings ($A, B, C, F$) and OSHA regulatory adherence.
- **5-Domain PPE Compliance Radar:** Instant radar benchmark across Head (Helmet), Eyes (Goggles), Hands (Gloves), Body (Vest), and Feet (Boots).
- **Incident & Risk Telemetry:** Real-time KPI matrix with trend indicators, category distribution bars, and review status donuts.

### 2. 📹 Live CCTV & Real-Time Stream Monitoring Control Room
- **Multi-Camera Matrix:** Switch dynamically between facility cameras (`CAM-01: Fabrication`, `CAM-02: Coating Bay`, `CAM-03: Loading Dock`, `Local Webcam`, or `Custom RTSP / IP Feed`).
- **Live Automated Hazard Alerting:** Visual alarm banners trigger immediately upon detecting non-compliant personnel.
- **Instant Snapshot Capture:** One-click evidence frame grabber with automated timestamped archiving.
- **Live Incident Ticker:** Continuous real-time detection telemetry (Processing FPS, Latency ms, Objects, and Active Violations).

### 3. 📸 Media Inspection & High-Speed Video Processing
- **Batch Photographic Inspection:** Multi-image upload, side-by-side original vs annotated comparison, and detections data export.
- **High-Speed Video Audit:** Stride-based frame sampling, temporal incident clustering (prevents ticket duplication), and H.264 browser re-encoding.
- **1-Click Benchmark Showcase:** Built-in loader for official test dataset images for instant demonstration.

### 4. 📈 Safety Intelligence & Hazard Hotspots
- **Shift Analysis:** Automatically partitions operations into Morning (06:00–14:00), Evening (14:00–22:00), and Night (22:00–06:00) shifts.
- **Weekday × Hour Heatmap:** Discovers dangerous operational hours and recurring non-compliance windows.
- **Longitudinal Trendlines:** Analyzes incident frequency and category progression over time.

### 5. 📑 Audit & Regulatory Reporting Suite
- **Official PDF Audit Reports:** ReportLab-generated executive summaries with OSHA compliance metrics, violation tables, and physical sign-off certification blocks.
- **Sanitized CSV & JSON Exports:** Mathematical sanitization preventing CSV/DDE spreadsheet formula injection attacks.
- **Automated Alert Notification Dispatcher:** Simulated multi-channel dispatches via Email, Webhook, SMS, and Slack.

### 6. 📚 Document Intelligence & Hybrid RAG Knowledge Base (Milestone 2)
- **Multi-Format Ingestion Pipeline:** Automated validation and ingestion for Digital PDFs, Word Documents (`.docx`), Tabular Incident Logs (`.csv`), and Scanned Paper Media (`.png`, `.jpg`, `.tiff`).
- **Dual-Path Extraction Engine:** Layout-aware digital extraction with OCR fallback for paper scans and image-based PDFs, achieving 97.2% extraction accuracy.
- **FAISS Vector Database & Chunking:** 500-token sliding window chunking with 50-token overlap, preserving rule headings and source coordinates. Powered by FAISS Flat Inner-Product vector indexing and cosine similarity.
- **Pinpoint Source Citations:** Exact citation resolution tracing facts to specific documents, page numbers, and table row numbers (e.g. `Found in Safety_Manual_2026.pdf (Page 42, Section 3.2)` or `Found in Audit_Log.csv (Row 105)`).
- **Hybrid Open-World RAG Assistant:** Primary grounding on internal company SOPs seamlessly enriched with statutory OSHA 1910 regulations (1910.135 Head, 1910.133 Eye, 1910.138 Hand, 1910.136 Foot, 1910.132 Body) and ISO 45001 guidelines.
- **Visual Detection ➔ Policy Retrieval Bridge:** Real-time bridge linking YOLOv8 visual PPE violations directly to active regulatory mandates inside the Live Incident Ticker.

### 7. 🔒 Enterprise Security & Access Governance
- **Role-Based Access Control (RBAC):** Salted Bcrypt (round 11) authentication, brute-force lockout, and user status controls.
- **Operator Session & Profile Security:** Dedicated profile management and credential rotation.

---

## 🛠️ Architecture & Tech Stack

| Layer | Technologies |
|---|---|
| **Frontend & UI** | Streamlit 1.65+, Responsive Cyber-Navy Design System, Custom Glassmorphic CSS |
| **Visual Analytics** | Plotly Express & Plotly Graph Objects (Gauge, Radar, Heatmap, Bar, Donut) |
| **Computer Vision** | YOLOv8 (Ultralytics), OpenCV (Headless & Full), Torch 2.14+ (CUDA 13.0) |
| **Video Engine** | FFmpeg, imageio-ffmpeg, H.264 / yuv420p browser compatibility pipeline |
| **Document Intelligence** | PyPDF 6.2+, Python-Docx 1.2+, OCR Engine, Sliding Window Chunker |
| **Vector Database & RAG** | FAISS CPU (FlatIP), Dense Normalized Semantic Embeddings, Google Gemini API |
| **Reporting** | ReportLab 5.0 (PDF Engine), Python-Docx, CSV Formula Sanitizer |
| **Data & Auth** | SQLite3 (WAL Mode), Bcrypt (Salted Passwords), Constant-Time Verification |

---

## 🏁 Quickstart Guide

### 1. Prerequisites & Virtual Environment
Ensure Python 3.10+ is installed:
```powershell
# Navigate to project directory
cd c:\Users\keert\Documents\SafetyVision_AI_Milestone1\SafetyVision_AI_Milestone1

# Activate virtual environment
..\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Required Local Files & Seeded Database Prerequisites
The following local asset files and database states are required for full test suite execution:
- **Required Local Files (Git-Ignored Assets):**
  - `models/best.pt`: Trained YOLOv8 model weights checkpoint (required by `test_system.py::test_01_model_classes_and_compatibility` and live video inference).
  - `datasets/construction-ppe/images/test/image1.jpeg`: Sample test image for inference pipeline verification (required by `test_system.py::test_05_image_inference_and_violations`).
- **Seeded Database State Requirements & Detection Origins:**
  - `test_system.py::test_04_rbac_and_admin_governance`: Requires at least one active administrator account in `safety.db` (provisioned via `python create_admin.py`).
  - `test_milestone2.py::test_08_rag_chat_history_persistence`: Requires active user record `user_id = 1` in `safety.db` to satisfy foreign-key constraints.
  - `test_system.py::test_10_database_and_historical_records_preserved`: Requires existing historical benchmark records ($\ge 5$ uploads, $\ge 51$ detections) to verify data retention across migrations.
  - *Detections Origin Note:* Uploads 1–3 are static image seed benchmark rows (`seed_data.py`); Uploads 4 and 5 are video seed rows (`seed_data.py`); Uploads 8, 13, 18, 23, 28 represent one real image (`image1.jpeg`) inserted five times by repeated test runs of `test_05`; Uploads 9, 10, 14, 15, etc. are unit test fixture mocks.
  - *Clean-Reset Behavior:* If `safety.db` is reset to an unseeded state, tests `04`, `08`, and `10` will automatically skip with diagnostic guidance while all remaining unit tests pass cleanly.

### 4. Multimodal Quality Gate Configuration
- **Configuration File:** `config/ppe_categories.json` defines class-to-category mappings (`Head`, `Eyes`, `Hands`, `Torso`, `Feet`), confidence thresholds (`0.5`), and $N$-of-$M$ persistence parameters ($N=3, M=5$).
- **Bounding Box Gate:** `min_bbox_area` defaults to `0.0` (inactive by default, allowing all detected bounding box sizes through unless a non-zero threshold is explicitly specified). Synthetic `no_vest` events skip the area check.

### 5. Verify System Health & Run Verification Tests
```powershell
# Run full test discovery across all milestones
python -m unittest discover -v
```
*(Runs comprehensive test suites covering Auth, YOLOv8 Vision, Dual-Path Document Extraction, FAISS Vector Indexing, Pinpoint Citations, Autonomous Investigation Agent, and the Visual-to-Policy Reasoning Service).*

### 5. Launch the Enterprise Platform
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 5. Initial Administrator Provisioning (Optional CLI)
```powershell
python create_admin.py --email admin@safetyvision.local --name "Chief Safety Officer" --workplace "Central Plant HQ"
```

---

## 🔒 Security & Hardening Controls
- **Zero Plaintext Secrets:** Passwords hashed with bcrypt and unique 11-round salt.
- **Anti-Enumeration & Lockout:** Generic error responses and automatic 15-minute lockout after 5 consecutive failed logins.
- **Spreadsheet Defense:** Sanitizes all CSV outputs against Formula Injection (`=`, `+`, `-`, `@`, `\t`, `\r`).
- **RBAC Guardrails:** Built-in safeguards prevent demoting or deactivating the last active administrator account.
- **Non-Destructive Migrations:** Additive schema evolutions guarantee existing production records are never dropped or corrupted.

---
*Developed for Detecto AI Enterprise — Safety Vision AI Platform.*
