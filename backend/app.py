"""
FinMe Backend — app.py
version: 0.3.0 — date-aware search added
"""

import os
import sys
import logging
import threading
import numpy as np
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from apscheduler.schedulers.background import BackgroundScheduler

from qdrant_client import QdrantClient
from embeddings.embedder import generate_embeddings
from vectorstore.qdrant_ingestor import search_similar, create_collection
from agents.report_agent import generate_report
from agents.market_intelligence_agent import generate_market_intelligence
from market.market_service import get_quote

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# ── Config ────────────────────────────────────
QDRANT_URL      = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_HOST     = QDRANT_URL.replace("http://", "").split(":")[0]
QDRANT_PORT     = int(QDRANT_URL.replace("http://", "").split(":")[1]) if ":" in QDRANT_URL.replace("http://", "") else 6333
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "financial_news")
INGEST_INTERVAL = int(os.getenv("INGEST_INTERVAL_MINUTES", "60"))

# ── Time range → days mapping ─────────────────
TIME_RANGE_DAYS = {
    "today":      1,
    "this_week":  7,
    "this_month": 30,
    "all_time":   None,   # no filter
}

# ── Shared state ──────────────────────────────
qdrant_client: QdrantClient = None
scheduler: BackgroundScheduler = None
_ingest_lock = threading.Lock()

ingestion_status = {
    "state":              "idle",
    "last_run":           None,
    "last_success":       None,
    "last_error":         None,
    "articles_stored":    0,
    "articles_duplicate": 0,
    "articles_failed":    0,
    "next_run":           None,
}


# ── Pipeline runner ───────────────────────────

def run_ingestion_pipeline():
    global ingestion_status

    if not _ingest_lock.acquire(blocking=False):
        logger.info("⏭️  [scheduler] Ingestion already running — skipping")
        return

    try:
        logger.info("\n" + "="*60)
        logger.info("🚀 [scheduler] Starting scheduled ingestion")
        logger.info("="*60)

        ingestion_status["state"]    = "running"
        ingestion_status["last_run"] = datetime.now().isoformat()

        from agents.langgraph_pipeline import LangGraphPipeline
        from config.rss_sources import RSS_FEEDS

        pipeline = LangGraphPipeline(
            client=qdrant_client,
            collection_name=COLLECTION_NAME,
        )
        pipeline.run(RSS_FEEDS)

        ingestion_status["state"]              = "success"
        ingestion_status["last_success"]       = datetime.now().isoformat()
        ingestion_status["articles_stored"]    = pipeline.stored
        ingestion_status["articles_duplicate"] = pipeline.duplicates
        ingestion_status["articles_failed"]    = pipeline.failed

        logger.info("✅ [scheduler] Ingestion complete — stored: %d", pipeline.stored)

    except Exception as e:
        ingestion_status["state"]      = "error"
        ingestion_status["last_error"] = str(e)
        logger.error("❌ [scheduler] Ingestion failed: %s", e)

    finally:
        _ingest_lock.release()


# ── Lifespan ──────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global qdrant_client, scheduler

    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    logger.info("✅ Connected to Qdrant at %s:%d", QDRANT_HOST, QDRANT_PORT)

    create_collection(qdrant_client, COLLECTION_NAME)

    scheduler = BackgroundScheduler(timezone="Asia/Kolkata")
    scheduler.add_job(
        run_ingestion_pipeline,
        trigger="interval",
        minutes=INGEST_INTERVAL,
        id="ingestion_job",
        name="FinMe RSS Ingestion",
    )
    scheduler.start()

    next_run = scheduler.get_job("ingestion_job").next_run_time
    ingestion_status["next_run"] = next_run.isoformat() if next_run else None
    logger.info("⏰ [scheduler] Every %d min | Next: %s", INGEST_INTERVAL, ingestion_status["next_run"])

    yield

    scheduler.shutdown(wait=False)
    logger.info("👋 Shutting down FinMe backend")


# ── App ───────────────────────────────────────

app = FastAPI(
    title="FinMe API",
    description="Financial news intelligence — semantic search + LLM reports",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ────────────────────────────────────

class QueryRequest(BaseModel):
    query:      str = Field(..., min_length=3, example="What happened with RBI this week?")
    top_k:      int = Field(default=10, ge=1, le=50)
    report:     bool = Field(default=True)
    time_range: Literal["today", "this_week", "this_month", "all_time"] = Field(
        default="this_week",
        description="Filter results by publish date. 'all_time' means no filter."
    )


class ChunkResult(BaseModel):
    title:      str
    url:        str
    published:  str | None
    summary:    str | None
    chunk:      str
    score:      float
    story_type: str | None
    source_name: str | None


class QueryResponse(BaseModel):
    query:        str
    report:       str | None
    chunks:       list[ChunkResult]
    chunk_count:  int
    time_range:   str
    method:       str
    generated_at: str


class SearchResponse(BaseModel):
    query:      str
    results:    list[ChunkResult]
    count:      int
    time_range: str


class IngestResponse(BaseModel):
    message:   str
    triggered: bool
    status:    dict

class MarketIntelligenceResponse(BaseModel):
    query: str
    market_data: dict | None
    market_context: dict | None
    articles_used: list[dict]
    report: str
    articles_found: int

# ── Routes ────────────────────────────────────

@app.get("/")
def read_root():
    return {"message": "FinMe backend is running", "docs": "/docs", "version": "0.3.0"}


@app.get("/health")
def health():
    try:
        collections = qdrant_client.get_collections()
        names = [c.name for c in collections.collections]
        next_run = None
        if scheduler:
            job = scheduler.get_job("ingestion_job")
            if job and job.next_run_time:
                next_run = job.next_run_time.isoformat()
        return {
            "status":          "ok",
            "qdrant":          "connected",
            "collections":     names,
            "ingestion_state": ingestion_status["state"],
            "next_ingestion":  next_run,
            "timestamp":       datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Qdrant unreachable: {e}")


@app.get("/config")
def config():
    return {
        "qdrant_url":      os.getenv("QDRANT_URL"),
        "ollama_url":      os.getenv("OLLAMA_URL"),
        "collection":      COLLECTION_NAME,
        "ingest_interval": f"every {INGEST_INTERVAL} minutes",
        "time_ranges":     list(TIME_RANGE_DAYS.keys()),
    }


@app.get("/stats")
def stats():
    try:
        info = qdrant_client.get_collection(COLLECTION_NAME)
        return {
            "collection":    COLLECTION_NAME,
            "points_count":  info.points_count,
            "vector_size":   info.config.params.vectors.size,
            "distance":      str(info.config.params.vectors.distance),
            "last_ingestion": ingestion_status["last_success"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/run", response_model=IngestResponse)
def trigger_ingestion(background_tasks: BackgroundTasks):
    if ingestion_status["state"] == "running":
        return IngestResponse(
            message="Ingestion already running",
            triggered=False,
            status=ingestion_status,
        )
    logger.info("📥 [api] Manual ingestion triggered")
    background_tasks.add_task(run_ingestion_pipeline)
    return IngestResponse(
        message="Ingestion started in background",
        triggered=True,
        status=ingestion_status,
    )


@app.get("/ingest/status")
def ingest_status():
    next_run = None
    if scheduler:
        job = scheduler.get_job("ingestion_job")
        if job and job.next_run_time:
            next_run = job.next_run_time.isoformat()
    return {**ingestion_status, "next_run": next_run, "interval_minutes": INGEST_INTERVAL}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    logger.info(
        "📨 Query: '%s' | time_range=%s | top_k=%d | report=%s",
        request.query, request.time_range, request.top_k, request.report
    )

    days_back = TIME_RANGE_DAYS.get(request.time_range)

    if days_back is not None:
        logger.info("📅 [query] Date filter: last %d days", days_back)
    else:
        logger.info("📅 [query] No date filter (all time)")

    try:
        query_vector = np.array(generate_embeddings([request.query])[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    try:
        results = search_similar(
            qdrant_client,
            COLLECTION_NAME,
            query_vector,
            limit=request.top_k,
            days_back=days_back,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qdrant search failed: {e}")

    # If date-filtered search returns nothing, fall back to all_time
    if not results and days_back is not None:
        logger.warning(
            "⚠️  [query] No results for time_range=%s — falling back to all_time",
            request.time_range
        )
        results = search_similar(
            qdrant_client,
            COLLECTION_NAME,
            query_vector,
            limit=request.top_k,
            days_back=None,
        )

    # Deduplicate
    chunks   = []
    payloads = []
    seen_keys = set()

    for r in results:
        if not r.payload:
            continue
        url        = r.payload.get("url", "")
        chunk_text = r.payload.get("chunk", "")
        key        = f"{url}_{chunk_text[:50]}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        payloads.append(r.payload)
        chunks.append(ChunkResult(
            title=       r.payload.get("title", ""),
            url=         url,
            published=   r.payload.get("published"),
            summary=     r.payload.get("summary"),
            chunk=       chunk_text,
            score=       round(r.score, 4),
            story_type=  r.payload.get("story_type"),
            source_name= r.payload.get("source_name"),
        ))

    report_data = None
    method      = "search_only"

    if request.report and payloads:
        report_data = generate_report(request.query, payloads)
        method      = report_data.get("method", "llm")

    return QueryResponse(
        query=        request.query,
        report=       report_data["report"] if report_data else None,
        chunks=       chunks,
        chunk_count=  len(chunks),
        time_range=   request.time_range,
        method=       method,
        generated_at= datetime.now().isoformat(),
    )


@app.get("/search", response_model=SearchResponse)
def search_endpoint(
    q: str,
    top_k: int = 10,
    time_range: str = "all_time"
):
    if not q:
        raise HTTPException(status_code=400, detail="Query 'q' is required")

    days_back    = TIME_RANGE_DAYS.get(time_range)
    query_vector = np.array(generate_embeddings([q])[0])
    results      = search_similar(qdrant_client, COLLECTION_NAME, query_vector, limit=top_k, days_back=days_back)

    seen_keys = set()
    chunks    = []
    for r in results:
        if not r.payload:
            continue
        key = f"{r.payload.get('url','')}_{r.payload.get('chunk','')[:50]}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        chunks.append(ChunkResult(
            title=       r.payload.get("title", ""),
            url=         r.payload.get("url", ""),
            published=   r.payload.get("published"),
            summary=     r.payload.get("summary"),
            chunk=       r.payload.get("chunk", ""),
            score=       round(r.score, 4),
            story_type=  r.payload.get("story_type"),
            source_name= r.payload.get("source_name"),
        ))

    return SearchResponse(query=q, results=chunks, count=len(chunks), time_range=time_range)

@app.get("/market/quote")
def market_quote(symbol: str):
    return get_quote(symbol)

@app.get("/market/overview")
def market_overview():
    symbols = {
        "NIFTY": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "SENSEX": "^BSESN",
    }

    result = {}

    for name, ticker in symbols.items():
        try:
            info = get_quote(ticker)
            result[name] = info
        except:
            pass

    return result

@app.get(
    "/market/intelligence",
    response_model=MarketIntelligenceResponse
)
def market_intelligence(query: str):

    result = generate_market_intelligence(
        user_query=query,
        qdrant_client=qdrant_client,
        collection_name=COLLECTION_NAME,
    )

    return MarketIntelligenceResponse(
        query=query,
        market_data=result["market_data"],
        market_context=result["market_context"],
        articles_used=result["articles_used"],
        report=result["intelligence_report"]["report"],
        articles_found=len(result["news_chunks"]),
    )