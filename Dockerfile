
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system libraries required by OpenCV and document processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY SafetyVision_AI_Milestone1/requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install langgraph

# Copy application source
COPY SafetyVision_AI_Milestone1/ /app/

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501"]
