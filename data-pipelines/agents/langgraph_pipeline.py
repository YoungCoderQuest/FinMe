"""
FinMe — LangGraph Multi-Agent Pipeline (skeleton)

This is the target architecture. Currently only the DeduplicationNode
is fully wired. RelevanceNode and downstream query agents are stubs
ready to implement.

Install:  pip install langgraph langchain-core

Graph shape (ingestion):

    fetch_articles
         │
    hash_dedup
         │
    extract_article
         │
    [RelevanceAgent]  ─── IRRELEVANT ──→ END
         │ RELEVANT
    [DeduplicAgent]   ─── DUPLICATE  ──→ END
         │ NEW_STORY / UPDATE
    chunk_and_embed
         │
    store_to_qdrant
         │
        END
"""

import logging
import uuid
from typing import TypedDict

import numpy as np
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig

from collector.rss_collector import fetch_articles
from processing.article_extractor import extract_article
from processing.chunker import chunk_text
from processing.deduplicator import deduplicate_articles
from embeddings.embedder import generate_embeddings
from vectorstore.qdrant_ingestor import insert_chunks, search_similar, url_exists
from agents.dedup_agent import classify_article
from agents.relevance_agent import classify_relevance
from qdrant_client.models import PointStruct

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────
# State schema — shared across all nodes
# ──────────────────────────────────────────────────

class ArticleState(TypedDict):
    # Input
    article:          dict

    # Set by extractor
    extracted_text:   str

    # Set by dedup node
    dedup_decision:   str     # DUPLICATE | UPDATE | NEW_STORY
    dedup_reason:     str

    # Set by relevance node (future)
    relevant:         bool

    # Set by chunker
    chunks:           list[str]
    embeddings:       list

    # Output
    stored:           bool
    skip_reason:      str


# ──────────────────────────────────────────────────
# Node functions
# ──────────────────────────────────────────────────

def extract_node(state: ArticleState, config: RunnableConfig) -> ArticleState:
    """Fetch full article text from URL."""

    article = state["article"]

    client = config["configurable"]["qdrant_client"]
    coll   = config["configurable"]["collection_name"]

    url   = article.get("link", "")
    title = article.get("title", "")

    # Fast path — skip if already stored
    if url_exists(client, coll, url):
        logger.info(
            "⏭️  [extract] Already in Qdrant, skipping: %s",
            title[:80]
        )
        return {
            **state,
            "extracted_text": "",
            "skip_reason": "already_exists"
        }

    extracted = extract_article(url)

    if not extracted or not extracted.get("text", "").strip():
        return {
            **state,
            "extracted_text": "",
            "skip_reason": "extraction_failed"
        }

    return {
        **state,
        "extracted_text": extracted["text"]
    }

def relevance_node(state: ArticleState, config: RunnableConfig) -> ArticleState:
    """Filter out noise — only financially meaningful articles pass through."""
    article = state["article"]
    title   = article.get("title", "")[:60]

    result = classify_relevance(article)

    method_indicator = {
        "pattern":   "🔍 PATTERN",
        "heuristic": "📊 HEURISTIC",
        "llm":       "🤖 LLM",
    }.get(result["method"], result["method"])

    decision_emoji = "✅" if result["decision"] == "RELEVANT" else "🚫"

    logger.info(
        "%s [relevance] %s — %s | %s",
        decision_emoji, result["decision"], result["reason"], method_indicator
    )

    return {
        **state,
        "relevant": result["decision"] == "RELEVANT",
    }


def dedup_node(state: ArticleState, config: RunnableConfig) -> ArticleState:
    """Semantic dedup at article level using title + summary embedding."""
    article = state["article"]
    client  = config["configurable"]["qdrant_client"]
    coll    = config["configurable"]["collection_name"]

    article_text   = f"{article.get('title', '')} {article.get('summary', '')}"
    article_vector = np.array(generate_embeddings([article_text])[0])

    similar = search_similar(client, coll, article_vector, limit=5)
    top_score        = similar[0].score if similar else 0.0
    similar_payloads = [r.payload for r in similar if r.payload]

    result = classify_article(
        new_article=article,
        similar_articles=similar_payloads,
        similarity_score=top_score,
    )

    logger.info(
        "[dedup] %s — %s (score %.3f, via %s)",
        result["decision"], result["reason"], top_score, result["method"]
    )

    return {
        **state,
        "dedup_decision": result["decision"],
        "dedup_reason":   result["reason"],
    }


def chunk_and_embed_node(state: ArticleState, config: RunnableConfig) -> ArticleState:
    """Chunk full article text and generate embeddings."""
    chunks     = chunk_text(state["extracted_text"])
    embeddings = generate_embeddings(chunks) if chunks else []
    return {**state, "chunks": chunks, "embeddings": embeddings}


def store_node(state: ArticleState, config: RunnableConfig) -> ArticleState:
    """Write accepted chunks to Qdrant."""
    article    = state["article"]
    client     = config["configurable"]["qdrant_client"]
    coll       = config["configurable"]["collection_name"]
    article_id = str(uuid.uuid4())
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=np.array(emb).tolist(),
            payload={
                "article_id": article_id,
                "chunk_index": idx,

                "title": article["title"],
                "url": article["link"],

                "published": article.get("published"),
                "summary": article.get("summary"),

                "source_name": article.get(
                    "source",
                    ""
                ),

                "feed_url": article.get(
                    "feed_url",
                    ""
                ),

                "chunk": chunk,

                "story_type": state["dedup_decision"],
            },
        )
        for idx, (chunk, emb) in enumerate(zip(state["chunks"], state["embeddings"]))
    ]

    if points:
        insert_chunks(client, coll, points)
        logger.info("[store] ✅ Stored %d chunks", len(points))

    return {**state, "stored": bool(points)}


# ──────────────────────────────────────────────────
# Routing functions
# ──────────────────────────────────────────────────

def route_extraction(state: ArticleState) -> str:
    if not state.get("extracted_text"):
        return END
    return "relevance"

def route_relevance(state: ArticleState) -> str:
    if not state.get("relevant", True):
        return END
    return "dedup"

def route_dedup(state: ArticleState) -> str:
    if state.get("dedup_decision") == "DUPLICATE":
        return END
    return "chunk_and_embed"


# ──────────────────────────────────────────────────
# Build graph
# ──────────────────────────────────────────────────

def build_ingestion_graph():
    g = StateGraph(ArticleState)

    g.add_node("extract",         extract_node)
    g.add_node("relevance",       relevance_node)
    g.add_node("dedup",           dedup_node)
    g.add_node("chunk_and_embed", chunk_and_embed_node)
    g.add_node("store",           store_node)

    g.set_entry_point("extract")

    g.add_conditional_edges("extract",         route_extraction, {"relevance": "relevance", END: END})
    g.add_conditional_edges("relevance",       route_relevance,  {"dedup": "dedup",         END: END})
    g.add_conditional_edges("dedup",           route_dedup,      {"chunk_and_embed": "chunk_and_embed", END: END})
    g.add_edge("chunk_and_embed", "store")
    g.add_edge("store", END)

    return g.compile()


# ──────────────────────────────────────────────────
# Pipeline runner
# ──────────────────────────────────────────────────

class LangGraphPipeline:
    """
    Drop-in replacement for IngestionPipeline using LangGraph.
    Usage is identical: pipeline.run(rss_feeds)
    """

    def __init__(self, client, collection_name):
        self.client          = client
        self.collection_name = collection_name
        self.graph           = build_ingestion_graph()

        self.stored     = 0
        self.duplicates = 0
        self.failed     = 0

    def run(self, rss_feeds):
        articles = fetch_articles(rss_feeds)
        articles = deduplicate_articles(articles)

        # Filter out liveblog/ticker URLs before processing
        from config.rss_sources import is_extractable_url
        before = len(articles)
        articles = [a for a in articles if is_extractable_url(a.get("link", ""))]
        skipped = before - len(articles)
        if skipped:
            logger.info("Skipped %d non-extractable URLs (liveblogs, tickers)", skipped)

        logger.info("After hash dedup + URL filter: %d articles", len(articles))

        config = {
            "configurable": {
                "qdrant_client":   self.client,
                "collection_name": self.collection_name,
            }
        }

        for article in articles:
            initial_state: ArticleState = {
                "article":        article,
                "extracted_text": "",
                "dedup_decision": "",
                "dedup_reason":   "",
                "relevant":       True,
                "chunks":         [],
                "embeddings":     [],
                "stored":         False,
                "skip_reason":    "",
            }

            final = self.graph.invoke(initial_state, config=config)

            if final.get("stored"):
                self.stored += 1
            elif final.get("dedup_decision") == "DUPLICATE":
                self.duplicates += 1
            elif final.get("skip_reason"):
                self.failed += 1

        logger.info(
            "Pipeline done — stored: %d  duplicates: %d  failed: %d",
            self.stored, self.duplicates, self.failed
        )
