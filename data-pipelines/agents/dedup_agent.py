"""
Semantic Deduplication Agent — Article Level

Classifies an incoming article against the top-N most similar articles
already in Qdrant as one of:

    DUPLICATE   — same story, skip entirely
    UPDATE      — same event, new information worth storing
    NEW_STORY   — unrelated, store as-is

LLM backend: Ollama (local). Default model: mistral or qwen2.5.
Falls back to heuristics if Ollama is unreachable.
"""
import os
import json
import logging
import requests
from enum import Enum

logger = logging.getLogger(__name__)


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/chat"
OLLAMA_MODEL = "mistral:latest"  # Switched from llama3.2:3b to mistral
OLLAMA_TIMEOUT = 30


class DedupDecision(str, Enum):
    DUPLICATE  = "DUPLICATE"
    UPDATE     = "UPDATE"
    NEW_STORY  = "NEW_STORY"


# ──────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────

def classify_article(
    new_article: dict,
    similar_articles: list[dict],
    similarity_score: float,
) -> dict:
    """
    Article-level dedup decision.

    Args:
        new_article:      {"title": ..., "summary": ..., "published": ...}
        similar_articles: list of Qdrant payloads for the top similar hits
        similarity_score: cosine similarity of the closest hit (0–1)

    Returns:
        {
            "decision": "DUPLICATE" | "UPDATE" | "NEW_STORY",
            "reason":   "...",
            "method":   "heuristic" | "llm"
        }
    """

    # Fast path — nothing similar in Qdrant yet
    if not similar_articles or similarity_score < 0.70:
        return _result(DedupDecision.NEW_STORY, "No similar articles found.", "heuristic")

    # Very high similarity — safe to call duplicate without LLM
    if similarity_score >= 0.97:
        return _result(
            DedupDecision.DUPLICATE,
            f"Cosine similarity {similarity_score:.3f} — near-identical content.",
            "heuristic",
        )

    # Middle band (0.70–0.97) → ask the LLM
    llm_result = _classify_with_llm(new_article, similar_articles, similarity_score)
    if llm_result:
        return llm_result

    # LLM unreachable — fall back to heuristics
    return _heuristic_fallback(new_article, similar_articles, similarity_score)


# ──────────────────────────────────────────────
# LLM classification
# ──────────────────────────────────────────────

def _classify_with_llm(
    new_article: dict,
    similar_articles: list[dict],
    similarity_score: float,
) -> dict | None:
    """
    Calls Ollama with a structured prompt.
    Returns a decision dict or None if the call fails.
    """

    existing_summaries = "\n".join(
        f"- [{i+1}] {a.get('title', 'No title')}: {a.get('summary', '')[:200]}"
        for i, a in enumerate(similar_articles[:3])
    )

    prompt = f"""You are a financial news deduplication agent.

Your job is to classify an incoming article relative to existing articles already stored in a database.

Classify the incoming article as exactly one of:
- DUPLICATE: Same story, no new information. Example: two wire services reporting the exact same earnings beat with no new details.
- UPDATE: Same event or company, but the new article adds meaningful new information (updated numbers, new quotes, regulatory response, market reaction). Example: breaking news vs follow-up with CEO comment.
- NEW_STORY: Unrelated topic or company. Store it.

Incoming article:
Title: {new_article.get("title", "")}
Summary: {new_article.get("summary", "")[:400]}

Most similar existing articles (cosine similarity: {similarity_score:.3f}):
{existing_summaries}

Respond with ONLY a JSON object in this exact format (no markdown, no explanation):
{{"decision": "DUPLICATE", "reason": "one sentence explanation"}}"""

    try:
        logger.debug(f"🤖 [LLM] Calling Mistral for decision (similarity: {similarity_score:.3f})")
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
        resp_data = response.json()
        
        # Try both "response" (chat API) and "message.content" (newer API)
        raw = resp_data.get("response", "").strip()
        if not raw and "message" in resp_data and "content" in resp_data["message"]:
            raw = resp_data["message"]["content"].strip()
        
        # Check for load failures
        if not raw or resp_data.get("done_reason") == "load":
            logger.warning("⚠️  [LLM] Ollama model failed to load or returned empty response")
            return None

        # Strip markdown fences if model wraps in ```json
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)
        decision = parsed.get("decision", "").upper()

        if decision not in DedupDecision._value2member_map_:
            logger.warning(f"⚠️  [LLM] Unknown decision '{decision}', falling back to heuristics")
            return None

        logger.debug(f"✅ [LLM] Mistral decided: {decision}")
        return _result(DedupDecision(decision), parsed.get("reason", ""), "llm")

    except requests.exceptions.ConnectionError:
        logger.warning("⚠️  [LLM] Ollama not reachable at %s — using heuristic fallback", OLLAMA_URL)
        return None
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"⚠️  [LLM] Response parse error: {e} — using heuristic fallback")
        return None
    except Exception as e:
        logger.warning(f"⚠️  [LLM] Call failed: {e} — using heuristic fallback")
        return None


# ──────────────────────────────────────────────
# Heuristic fallback
# ──────────────────────────────────────────────

def _heuristic_fallback(
    new_article: dict,
    similar_articles: list[dict],
    similarity_score: float,
) -> dict:
    """
    Simple title-overlap heuristic for when Ollama is unavailable.
    """
    closest = similar_articles[0] if similar_articles else {}
    new_title  = new_article.get("title", "").lower()
    old_title  = closest.get("title", "").lower()

    new_words = set(new_title.split())
    old_words = set(old_title.split())
    overlap   = len(new_words & old_words) / max(len(old_words), 1)

    if similarity_score >= 0.90 and overlap > 0.65:
        logger.debug(f"📊 [HEURISTIC] High similarity ({similarity_score:.2f}) + title overlap ({overlap:.0%}) → DUPLICATE")
        return _result(
            DedupDecision.DUPLICATE,
            f"High similarity ({similarity_score:.2f}) + title overlap ({overlap:.0%}).",
            "heuristic",
        )

    if similarity_score >= 0.80 and overlap > 0.40:
        logger.debug(f"📊 [HEURISTIC] Moderate similarity ({similarity_score:.2f}) + title overlap ({overlap:.0%}) → UPDATE")
        return _result(
            DedupDecision.UPDATE,
            f"Same topic, modified headline (overlap {overlap:.0%}).",
            "heuristic",
        )

    logger.debug(f"📊 [HEURISTIC] Low similarity ({similarity_score:.2f}) → NEW_STORY")
    return _result(DedupDecision.NEW_STORY, "Below semantic threshold.", "heuristic")


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _result(decision: DedupDecision, reason: str, method: str) -> dict:
    return {
        "decision": decision.value,
        "reason":   reason,
        "method":   method,
    }
