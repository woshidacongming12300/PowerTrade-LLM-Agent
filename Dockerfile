# ============================================================
# Electricity Price Intelligence — Docker Image
# ============================================================
# Build:  docker build -t epf-agent .
# Run:    docker run -p 8000:8000 --env-file .env epf-agent
# ============================================================

FROM python:3.12-slim

LABEL maintainer="zhushuyu"
LABEL description="Multi-Agent Electricity Price Forecasting System (LSTM + Chronos-T5 + Hybrid RAG)"

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /opt/epf-agent

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (smaller, faster for inference)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Copy and install Python dependencies
COPY requirements_deploy.txt .
RUN pip install --no-cache-dir -r requirements_deploy.txt

# Copy application code
COPY . .

# Create persist directories for ChromaDB
RUN mkdir -p /opt/epf-agent/chroma_db_v2 /opt/epf-agent/data

# Pre-train LSTM model (if model files not present)
RUN if [ ! -f /opt/epf-agent/algorithms/lstm_epf_multivariate.pth ] && \
       [ ! -f /opt/epf-agent/algorithms/lstm_epf.pth ]; then \
       echo "[Docker] No trained model found — skip (mount model files or train separately)."; \
    else \
       echo "[Docker] Model file present — ready."; \
    fi

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["chainlit", "run", "chainlit_app.py", "--host", "0.0.0.0", "--port", "8000"]
