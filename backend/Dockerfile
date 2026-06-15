FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend app first (includes agents/report_agent.py)
COPY backend/ .

# Copy data-pipelines modules — agents/ merges cleanly (no filename conflicts)
COPY data-pipelines/embeddings/              ./embeddings/
COPY data-pipelines/vectorstore/             ./vectorstore/
COPY data-pipelines/config/                  ./config/
COPY data-pipelines/collector/               ./collector/
COPY data-pipelines/processing/              ./processing/
COPY data-pipelines/agents/langgraph_pipeline.py  ./agents/langgraph_pipeline.py
COPY data-pipelines/agents/dedup_agent.py         ./agents/dedup_agent.py
COPY data-pipelines/agents/relevance_agent.py     ./agents/relevance_agent.py

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]