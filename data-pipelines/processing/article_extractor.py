"""
Article Extractor — FinMe

Improvements over base version:
- Rotating user-agent headers (looks like a real browser)
- Per-article retry with backoff (3 attempts)
- Fallback to RSS summary if full extraction fails
- Domain-level failure tracking (warns you about blocked sources)
- Extraction quality check (rejects suspiciously short articles)
"""

import time
import random
import logging
from collections import defaultdict
from newspaper import Article

logger = logging.getLogger(__name__)

# ── User agents ───────────────────────────────────────────────────
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]

# ── Config ────────────────────────────────────────────────────────
MAX_RETRIES       = 3
RETRY_BACKOFF     = 2.0      # seconds — doubles each retry
REQUEST_TIMEOUT   = 10       # seconds
MIN_ARTICLE_CHARS = 200      # reject anything shorter (likely a paywall page)
DOMAIN_FAIL_WARN  = 5        # warn if a domain fails this many times in one run

# ── Domain failure tracker (resets each pipeline run) ────────────
_domain_failures: dict = defaultdict(int)


def reset_domain_tracker():
    """Call this at the start of each pipeline run."""
    _domain_failures.clear()


def get_blocked_domains() -> list[str]:
    """Returns domains that failed >= DOMAIN_FAIL_WARN times."""
    return [d for d, n in _domain_failures.items() if n >= DOMAIN_FAIL_WARN]


# ── Main extractor ────────────────────────────────────────────────

def extract_article(url: str, rss_fallback: dict | None = None) -> dict | None:
    """
    Extract full article text from URL.

    Args:
        url:          Article URL
        rss_fallback: The original RSS article dict (title + summary).
                      If provided and extraction fails, returns summary
                      instead of None so the article isn't lost entirely.

    Returns:
        {
            "title":   str,
            "text":    str,
            "authors": list,
            "source":  "full" | "rss_summary"
        }
        or None if both full extraction and fallback are unavailable.
    """
    domain = _get_domain(url)

    # ── Attempt full extraction with retries ──
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            config = {
                "browser_user_agent": random.choice(USER_AGENTS),
                "request_timeout":    REQUEST_TIMEOUT,
            }

            article = Article(url, **config)
            article.download()
            article.parse()

            text = article.text.strip()

            if len(text) < MIN_ARTICLE_CHARS:
                raise ValueError(
                    f"Extracted text too short ({len(text)} chars) — likely paywall or bot block."
                )

            return {
                "title":   article.title,
                "text":    text,
                "authors": article.authors,
                "source":  "full",
            }

        except Exception as e:
            logger.warning(
                "Extraction attempt %d/%d failed for %s: %s",
                attempt, MAX_RETRIES, url, e
            )

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)

    # ── All retries exhausted ──
    _domain_failures[domain] += 1

    if _domain_failures[domain] == DOMAIN_FAIL_WARN:
        logger.error(
            "⚠️  Domain '%s' has failed %d times this run — consider removing it from RSS_FEEDS.",
            domain, DOMAIN_FAIL_WARN
        )

    # ── Fallback to RSS summary ──
    if rss_fallback:
        summary = rss_fallback.get("summary", "").strip()
        title   = rss_fallback.get("title", "").strip()

        if summary and len(summary) >= MIN_ARTICLE_CHARS:
            logger.info("↩️  Using RSS summary fallback for: %s", title)
            return {
                "title":   title,
                "text":    summary,
                "authors": [],
                "source":  "rss_summary",
            }

    logger.warning("❌ No usable content for: %s", url)
    return None


# ── Helpers ───────────────────────────────────────────────────────

def _get_domain(url: str) -> str:
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return "unknown"