"""
IngestionPipeline — FinMe

Dedup is now at ARTICLE level (title + summary embedding) BEFORE chunking.
Chunks are only generated and stored for NEW_STORY or UPDATE articles.

Flow:
    RSS → hash dedup → extract → article embedding → dedup agent
        → (if accepted) chunk + embed chunks → Qdrant
"""

import uuid
import logging
import numpy as np
from qdrant_client.models import PointStruct

from collector.rss_collector import fetch_articles
from processing.article_extractor import extract_article, reset_domain_tracker, get_blocked_domains
from processing.chunker import chunk_text
from processing.deduplicator import deduplicate_articles
from embeddings.embedder import generate_embeddings
from vectorstore.qdrant_ingestor import insert_chunks, search_similar
from agents.dedup_agent import classify_article

logger = logging.getLogger(__name__)


class IngestionPipeline:

    def __init__(self, client, collection_name):
        self.client          = client
        self.collection_name = collection_name

        # Similarity window for article-level dedup search
        # Anything below 0.70 is treated as NEW_STORY by the agent itself
        self.article_search_limit = 5

        # Counters
        self.total      = 0
        self.stored     = 0
        self.duplicates = 0
        self.updates    = 0
        self.failed     = 0

        self.processed_articles = []

    # ──────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────

    def run(self, rss_feeds):

        articles = fetch_articles(rss_feeds)
        logger.info("Collected %d RSS articles", len(articles))

        # Stage 1 — cheap exact dedup (URL + content hash)
        articles = deduplicate_articles(articles)
        logger.info("After hash dedup: %d articles", len(articles))

        reset_domain_tracker()
        self.total = len(articles)

        for idx, article in enumerate(articles, start=1):
            title = article.get("title", "no title")
            logger.info("[%d/%d] %s", idx, self.total, title)

            self._process_article(article)

        self._log_summary()

    # ──────────────────────────────────────────
    # Per-article processing
    # ──────────────────────────────────────────

    def _process_article(self, article: dict):

        # ── Extract full text ──
        extracted = extract_article(article["link"], rss_fallback=article)
        if not extracted or not extracted["text"].strip():
            logger.warning("  ❌ Extraction failed")
            self.failed += 1
            return

        # ── Stage 2: Article-level semantic dedup ──
        # Embed title + summary (short, fast, article-representative)
        article_text   = f"{article.get('title', '')} {article.get('summary', '')}"
        article_vector = self._embed_single(article_text)

        similar = search_similar(
            self.client,
            self.collection_name,
            article_vector,
            limit=self.article_search_limit,
        )

        top_score        = similar[0].score if similar else 0.0
        similar_payloads = [r.payload for r in similar if r.payload]

        decision = classify_article(
            new_article=article,
            similar_articles=similar_payloads,
            similarity_score=top_score,
        )

        label  = decision["decision"]
        reason = decision["reason"]
        method = decision["method"]

        logger.info(
            "  [%s via %s] score=%.3f — %s",
            label, method, top_score, reason
        )

        if label == "DUPLICATE":
            self.duplicates += 1
            return

        if label == "UPDATE":
            self.updates += 1

        # ── Stage 3: Chunk + embed full article text ──
        chunks = chunk_text(extracted["text"])
        if not chunks:
            logger.warning("  ❌ No chunks generated")
            self.failed += 1
            return

        embeddings = generate_embeddings(chunks)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=np.array(emb).tolist(),
                payload={
                    "title":     article["title"],
                    "url":       article["link"],
                    "published": article.get("published"),
                    "summary":   article.get("summary"),
                    "chunk":     chunk,
                    "story_type": label,   # tag UPDATE vs NEW_STORY for future agents
                },
            )
            for chunk, emb in zip(chunks, embeddings)
        ]

        insert_chunks(self.client, self.collection_name, points)

        self.stored += 1
        self.processed_articles.append({
            "title":      article["title"],
            "link":       article["link"],
            "published":  article.get("published"),
            "summary":    article.get("summary"),
            "story_type": label,
            "num_chunks": len(points),
            "chunks":     [p.payload["chunk"] for p in points],
        })

        logger.info("  ✅ Stored %d chunks", len(points))

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────

    def _embed_single(self, text: str) -> np.ndarray:
        """Embed a single short text for article-level comparison."""
        vectors = generate_embeddings([text])
        return np.array(vectors[0])

    def _log_summary(self):
        blocked = get_blocked_domains()
        if blocked:
            logger.warning("Blocked domains this run: %s", ", ".join(blocked))

        logger.info(
            "\n── Pipeline complete ──\n"
            "  Total:      %d\n"
            "  Stored:     %d (new + updates)\n"
            "  Updates:    %d\n"
            "  Duplicates: %d\n"
            "  Failed:     %d",
            self.total,
            self.stored,
            self.updates,
            self.duplicates,
            self.failed,
        )