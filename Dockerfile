# OLYMPUS Docker image
# Base: Python 3.11 slim
FROM python:3.11-slim

LABEL maintainer="OLYMPUS Research Team"
LABEL description="OLYMPUS Autonomous Security Intelligence System"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    iputils-ping \
    net-tools \
    nmap \
    file \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (CPU-only for Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install CPU PyTorch (smaller image; GPU users mount host CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy source
COPY . .

# Install OLYMPUS
RUN pip install --no-cache-dir -e .

# Create data directories
RUN mkdir -p data/models data/quarantine data/samples results

# Expose API and dashboard ports
EXPOSE 8000 8501

# Default: API server
CMD ["python", "-m", "uvicorn", "olympus.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
