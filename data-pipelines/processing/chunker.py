import re
import logging

logger = logging.getLogger(__name__)

# ── Boilerplate patterns to strip before chunking ─────────────────
# These are footer/navigation/subscription blobs that appear across
# ET, BS, Moneycontrol, CNBC articles and pollute chunk quality.

BOILERPLATE_PATTERNS = [
    # Economic Times footers
    r"What's moving Sensex and Nifty.*?(?=\n|$)",
    r"Track latest market news.*?(?=\n|$)",
    r"subscribe to our Telegram feeds.*?(?=\n|$)",
    r"Subscribe to ET Prime.*?(?=\n|$)",
    r"read the Economic Times ePaper.*?(?=\n|$)",
    r"Top Trending Stocks:.*?(?=\n|$)",
    r"You can now subscribe to our ETMarkets WhatsApp channel",
    r"ETMarkets\.com is now on Telegram.*?(?=\n|$)",
    r"For fastest news alerts.*?(?=\n|$)",
    r"Also, ETMarkets.*?(?=\n|$)",
    r"Disclaimer:.*?(?=\n|$)",
    r"\(Disclaimer:.*?\)",
    r"Recommendations, suggestions, views and opinions.*?(?=\n|$)",
    r"These do not represent the views of.*?(?=\n|$)",
    r"Can targeted reforms at home.*",   # ET related articles section
    r"Paisabazaar wanted to get you.*",
    r"Is Air India the cocktail.*",
    r"Long-term investing:.*",

    # Moneycontrol / BS footers
    r"First Published:.*?(?=\n|$)",
    r"Follow us on.*?(?=\n|$)",
    r"Download the.*?app.*?(?=\n|$)",
    r"Get live.*?on.*?(?=\n|$)",
    r"Click here for.*?(?=\n|$)",

    # Generic patterns
    r"Listen to this article.*?(?=\n|$)",
    r"Loading\.\.\.",
    r"Live Events",
    r"\d+\s*min read",
    r"Share this article.*?(?=\n|$)",
    r"Read more:.*?(?=\n|$)",
    r"Also read:.*?(?=\n|$)",
    r"\(You can now.*?\)",
    r"moreless\s*\d*",
    r"more less\s*\d*",
]

# Compile patterns once at module load
_COMPILED_PATTERNS = [
    re.compile(p, re.IGNORECASE | re.DOTALL)
    for p in BOILERPLATE_PATTERNS
]

# Min chunk length in words — discard tiny chunks that are pure noise
MIN_CHUNK_WORDS = 30


def clean_text(text: str) -> str:
    """Strip boilerplate from article text before chunking."""
    for pattern in _COMPILED_PATTERNS:
        text = pattern.sub(" ", text)

    # Collapse multiple whitespace/newlines
    text = re.sub(r"\s+", " ", text).strip()

    # Remove unicode garbage (Ã, â, etc. from bad encoding)
    text = re.sub(r"[^\x00-\x7F\u0900-\u097F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Clean and chunk article text.

    Args:
        text:       Raw article text
        chunk_size: Max words per chunk
        overlap:    Word overlap between consecutive chunks

    Returns:
        List of clean text chunks, noise chunks removed.
    """
    # Strip boilerplate first
    cleaned = clean_text(text)

    if not cleaned:
        logger.warning("⚠️  [chunker] Empty text after cleaning")
        return []

    words  = cleaned.split()
    chunks = []
    start  = 0

    while start < len(words):
        end   = start + chunk_size
        chunk = " ".join(words[start:end])

        # Skip chunks that are too short — likely leftover noise
        if len(words[start:end]) >= MIN_CHUNK_WORDS:
            chunks.append(chunk)
        else:
            logger.debug("⏭️  [chunker] Skipped short chunk (%d words)", len(words[start:end]))

        start += chunk_size - overlap

    logger.debug(
        "🔪 [chunker] %d chunks from %d words (size=%d, overlap=%d)",
        len(chunks), len(words), chunk_size, overlap
    )

    return chunks