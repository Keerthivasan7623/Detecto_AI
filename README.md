# Detecto AI – Workplace Vision Control Platform

An AI-powered workplace safety monitoring platform that uses computer vision to identify Personal Protective Equipment (PPE), monitor safety violations and support workplace safety investigations.

## About the Project

Workplace safety is an important part of every industrial environment. However, manually monitoring workers, checking protective equipment and maintaining safety records can be time-consuming.

**Detecto AI** aims to make this process easier through artificial intelligence and computer vision.

The platform analyzes uploaded images and videos using a trained YOLO object detection model to identify relevant safety equipment and potential PPE violations. It also provides inspection reports, visual evidence and additional AI-powered tools to help organize safety-related information.

The idea behind this project is to bring visual safety monitoring, reporting and investigation capabilities together in one platform.

## Key Features

- **AI-Based PPE Detection:** Uses a trained YOLO model to identify personal protective equipment in visual data.
- **Image and Video Analysis:** Upload images and videos for automated safety inspection.
- **Violation Detection:** Identifies potential PPE-related safety violations from detection results.
- **Visual Evidence:** Generates annotated images and stores relevant evidence for later review.
- **Safety Reports:** Creates inspection reports to help maintain safety records.
- **Document Intelligence:** Processes uploaded documents and uses vector-based retrieval to find relevant information.
- **AI Safety Investigation:** Uses LangGraph to support structured investigation workflows and corrective and preventive action (CAPA) tracking.
- **Dashboard Analytics:** Presents safety information through KPI cards, charts and inspection summaries.
- **User Authentication:** Provides account-based access and administrator functionality.

## Technology Stack

| Component | Technologies |
|---|---|
| Programming Language | Python |
| Frontend / Dashboard | Streamlit |
| Object Detection | YOLO |
| Image Processing | OpenCV |
| Database | SQLite |
| Vector Search | FAISS |
| AI Workflow | LangGraph |
| Document Processing | PyPDF, Python-docx |
| PDF Reporting | ReportLab |
| Containerization | Docker |
| Version Control | Git, GitHub |
| Cloud Hosting | Railway |

## System Architecture

The application follows a modular architecture where different components handle visual detection, data management, document processing and AI-assisted investigations.

```text
                 USER
                   |
                   v
           STREAMLIT DASHBOARD
                   |
          +--------+--------+
          |        |        |
          v        v        v
       YOLO      Reports  Document
     Detection            Intelligence
          |        |        |
          v        v        v
       OpenCV    SQLite    FAISS
          |
          v
     PPE Detection
          |
          v
    Violation Analysis
          |
          v
    Evidence & Reports
                   |
                   v
       LANGGRAPH WORKFLOW
                   |
                   v
       SAFETY INVESTIGATION
```

## How It Works

1. The user logs into the application.
2. Images or videos are uploaded through the dashboard.
3. The trained YOLO model processes the visual data.
4. Detected objects are analyzed to identify potential safety violations.
5. Annotated outputs and supporting evidence are generated.
6. Inspection information is stored and used for reporting.
7. Additional modules support document retrieval and structured safety investigations.

## Getting Started

Follow these steps to run the project locally.

### Prerequisites

Make sure you have the following installed:

- Python 3.11 or compatible version
- Git
- Docker Desktop (optional, for containerized execution)

### 1. Clone the Repository

```bash
git clone https://github.com/Keerthivasan7623/Detecto_AI.git
```

### 2. Navigate to the Project

```bash
cd Detecto_AI/SafetyVision_AI_Milestone1
```

### 3. Create a Virtual Environment

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

Linux / macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a `.env` file if your selected integrations require API credentials.

Add the required configuration values locally. Never upload your `.env` file or expose API keys in the repository.

### 6. Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, normally:

```text
http://localhost:8501
```

## Running with Docker

Detecto AI also supports Docker-based execution.

Docker packages the application, Python environment, dependencies and model files into a container.

From the repository root, build the image:

```bash
docker build -t detecto-ai .
```

Run the container:

```bash
docker run -p 8501:8501 detecto-ai
```

Access the application at:

```text
http://localhost:8501
```

## Cloud Deployment

The application is deployed using **Railway**, with Docker as the containerization technology.

The deployment workflow is:

1. Maintain the source code in GitHub.
2. Connect the GitHub repository to Railway.
3. Railway uses the Dockerfile to build the application image.
4. The application is deployed as a running container.
5. Railway provides a public URL for accessing the dashboard.

### Persistent Storage

To retain application data across redeployments, the project uses Railway persistent storage.

The storage configuration is managed through:

```text
SAFETYVISION_DATA_DIR=/data
```

Runtime data is stored separately from the application code.

```text
/data
    |
    |-- safety.db
    |
    |-- outputs/
          |
          |-- annotated/
          |-- reports/
          |-- evidence/
          |-- documents/
          |-- investigations/
          |-- vector_store/
```

This helps preserve database records, generated reports, uploaded documents, evidence and vector-store files when application containers are replaced.

## Project Structure

```text
Detecto_AI/
|
|-- Dockerfile
|
|-- SafetyVision_AI_Milestone1/
|     |
|     |-- app.py
|     |-- database.py
|     |-- auth.py
|     |-- reporting.py
|     |-- safety_engine.py
|     |-- create_admin.py
|     |
|     |-- models/
|     |     |-- best.pt
|     |
|     |-- components/
|     |
|     |-- services/
|     |     |-- agent/
|     |     |-- document_intelligence/
|     |     |-- multimodal/
|     |
|     |-- config/
|     |
|     |-- outputs/
|     |
|     |-- requirements.txt
|     |-- README.md
|
|-- docker-compose.yml
```

*Generated runtime files and local database files are kept separate from the source repository.*

## Future Improvements

Some areas that can be explored in future development include:

- Improving PPE detection accuracy across different workplace environments.
- Expanding safety violation categories.
- Enhancing real-time video monitoring capabilities.
- Improving analytical dashboards and historical safety insights.
- Extending automated investigation and reporting workflows.
- Further strengthening application security and deployment reliability.

## Project Information

**Project Name:** Detecto AI – Workplace Vision Control Platform

**Domain:** Artificial Intelligence, Computer Vision and Workplace Safety

**Primary Objective:** AI-assisted workplace safety monitoring and violation detection.

**Developer:** Keerthivasan Govindaraj

**GitHub Repository:** https://github.com/Keerthivasan7623/Detecto_AI

**Live Application:** https://detectoai-production.up.railway.app

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Please note that separately licensed models, datasets, libraries and third-party resources remain subject to their respective licenses.

---

*Built with Python, Computer Vision and AI, with the aim of making workplace safety monitoring more accessible and efficient.*
