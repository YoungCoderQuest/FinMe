# FinMe — Market Intelligence Platform

## Overview

FinMe is an AI-powered market intelligence platform that combines:

* Real-time market data
* Financial news ingestion
* Semantic search
* Retrieval-Augmented Generation (RAG)
* LLM-powered financial reporting

The goal is to provide investors with intelligent market insights by combining live market signals with historical news intelligence.

---

## Current Features

### Real-Time Market Data

Powered by Yahoo Finance.

Examples:

* TCS stock price
* Reliance stock performance
* NIFTY overview
* BANKNIFTY overview
* SENSEX overview

Returns:

* Current Price
* Daily Change
* Change %
* Day High
* Day Low
* Volume

---

### Financial News Intelligence

FinMe continuously ingests financial news through RSS feeds.

Pipeline:

RSS Sources → Article Extraction → Chunking → Deduplication → Embeddings → Qdrant

Currently stores:

* Article title
* URL
* Published date
* Summary
* Content chunks
* Metadata

---

### Semantic Search

Questions are converted into embeddings and searched against Qdrant.

Example:

"What happened with RBI this week?"

FinMe retrieves the most relevant financial news articles instead of relying on keyword matching.

---

### AI Report Generation

Retrieved articles are sent to a local LLM through Ollama.

The model generates:

* Executive summaries
* Financial analysis
* Market explanations
* Trend reports

---

### Market Intelligence Agent

Combines:

1. Real-time market data
2. Historical news intelligence
3. LLM reasoning

Example:

User:
How is TCS stock today?

Agent workflow:

* Fetches latest TCS quote from Yahoo Finance
* Retrieves related news from Qdrant
* Sends both to the LLM
* Generates a market intelligence report

---

## Tech Stack

### Backend

* FastAPI
* APScheduler

### AI

* Ollama
* Llama 3.2

### Vector Database

* Qdrant

### Embeddings

* sentence-transformers
* all-MiniLM-L6-v2

### Market Data

* Yahoo Finance (yfinance)

### Frontend

* Streamlit

### Containers

* Docker
* Docker Compose

---

## Architecture

User Query

↓

FastAPI Backend

↓

Market Intelligence Agent

↓

Yahoo Finance + Qdrant

↓

Retrieved Context

↓

Ollama (Llama 3.2)

↓

Financial Report

---

## Project Structure

backend/

* FastAPI APIs
* Report Agent
* Market Intelligence Agent

data-pipelines/

* RSS Collection
* Article Processing
* Embeddings
* Qdrant Storage
* LangGraph Pipeline

frontend/

* Streamlit UI

---

## Running Locally

### Clone

```bash
git clone <repo-url>
cd FinMe
```

### Start Services

```bash
docker compose up --build
```

### Pull Models:

```bash
docker compose exec ollama ollama pull mistral:latest
docker compose exec ollama ollama pull llama3.2:3b
```

### Running ingestion Pipeline
```bash
Invoke-RestMethod -Method POST -Uri "http://localhost:8000/ingest/run"
```
### Checking Pipeline Status:
```bash
Invoke-RestMethod -Uri "https://localhost:8000/ingest/status"       
```

### Services

Backend

```text
http://localhost:8000
```

Swagger Docs

```text
http://localhost:8000/docs
```

Qdrant Dashboard

```text
http://localhost:6333/dashboard
```

Frontend

```text
http://localhost:3000
```

---

## Example Endpoints

Health Check

```http
GET /health
```

Run Ingestion

```http
POST /ingest/run
```

Query

```http
POST /query
```

Example

```json
{
  "query": "What happened with RBI this week?",
  "top_k": 10,
  "report": true,
  "time_range": "this_week"
}
```

Market Quote

```http
GET /market/quote?symbol=TCS
```

Market Overview

```http
GET /market/overview
```

Market Intelligence

```http
POST /market/intelligence
```

Example

```json
{
  "query": "How is TCS stock today?"
}
```

---

## Current Status

Version: V1

Implemented:

* RSS ingestion pipeline
* Qdrant vector search
* Ollama integration
* Report generation
* Yahoo Finance integration
* Market Intelligence Agent
* Dockerized architecture

In Progress:

* Streamlit dashboard
* Better stock-specific retrieval
* Neo4j knowledge graph
* TimescaleDB market storage
* Agent orchestration

---

## Future Roadmap

* Multi-agent architecture
* Stock-specific memory
* Knowledge Graph reasoning
* Portfolio intelligence
* Market anomaly detection
* Personalized investor assistant
* Real-time market monitoring

---

## Author

Aishvarya S

FinMe — AI Powered Market Intelligence
