"""
Report Agent — FinMe

Combines:
1. Real-time market data
2. Historical news intelligence
3. LLM synthesis

Produces:
- Market summary
- Stock analysis
- News intelligence report
"""

import os
import logging
import requests

from datetime import datetime

logger = logging.getLogger(__name__)

OLLAMA_URL = (
    os.getenv("OLLAMA_URL", "http://localhost:11434")
    + "/api/chat"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:3b"
)

OLLAMA_TIMEOUT = 120


# --------------------------------------------------
# PUBLIC ENTRY
# --------------------------------------------------

def generate_report(
    user_query: str,
    chunks: list[dict],
    market_data: dict | None = None,
) -> dict:

    logger.info("\n" + "─" * 60)
    logger.info("📊 [report] Starting report generation")
    logger.info("📝 Query: %s", user_query)

    unique_chunks = []

    seen_titles = set()

    for chunk in chunks:

        title = chunk.get("title", "")

        if title not in seen_titles:
            seen_titles.add(title)
            unique_chunks.append(chunk)

    sources = [
        {
            "title": c.get("title"),
            "url": c.get("url"),
            "published": c.get("published"),
        }
        for c in unique_chunks[:8]
    ]

    report = _generate_with_llm(
        user_query=user_query,
        chunks=unique_chunks,
        market_data=market_data,
    )

    if report:

        return {
            "query": user_query,
            "report": report,
            "sources": sources,
            "chunk_count": len(unique_chunks),
            "method": "llm",
            "generated_at": datetime.now().isoformat(),
        }

    return _fallback_report(
        user_query=user_query,
        chunks=unique_chunks,
        market_data=market_data,
        sources=sources,
    )


# --------------------------------------------------
# LLM
# --------------------------------------------------

def _generate_with_llm(
    user_query: str,
    chunks: list[dict],
    market_data: dict | None = None,
) -> str | None:

    market_context = ""

    if market_data:

        market_context = f"""
REAL-TIME MARKET DATA

Symbol: {market_data.get("symbol")}
Current Price: {market_data.get("price")}
Daily Change: {market_data.get("change")}
Daily Change Percent: {market_data.get("change_pct")}
Day High: {market_data.get("day_high")}
Day Low: {market_data.get("day_low")}
Open: {market_data.get("open")}
Volume: {market_data.get("volume")}
"""

    context_parts = []
    total_chars = 0

    for i, chunk in enumerate(chunks):

        title = chunk.get("title", "")

        text = (
            chunk.get("chunk")
            or chunk.get("summary")
            or ""
        )

        source = chunk.get("url", "")

        entry = f"""
[{i+1}] {title}

{text[:300]}

Source: {source}
"""

        if total_chars + len(entry) > 3000:
            break

        context_parts.append(entry)
        total_chars += len(entry)

    news_context = "\n".join(context_parts)

    prompt = f"""
    You are FinMe's Market Intelligence Bestie 💖📈

    Your job is to make stock market and financial information easy, approachable, and fun to understand.

    PERSONALITY

    - Friendly
    - Warm
    - Encouraging
    - Cheerful
    - Slightly playful
    - Confident
    - Supportive
    - Easy to understand

    You should sound like a smart finance-savvy friend who genuinely enjoys helping people understand the market.

    IMPORTANT:

    - Be cute, but not childish.
    - Be fun, but not unprofessional.
    - Never sacrifice accuracy for personality.
    - Never invent information.
    - Never exaggerate gains or losses.
    - Avoid excessive emojis.
    - Use at most 1-3 relevant emojis in a response.

    GOOD EXAMPLES

    "Good news! TCS is having a positive day today ✨"

    "It looks like investors are feeling optimistic about this stock."

    "Nothing alarming here — today's movement looks fairly healthy."

    "The market's in a pretty decent mood today 💖"

    "Investors seem a little cautious right now, but there isn't widespread panic."

    AVOID

    "OMG QUEEN BUY THIS STOCK NOW 💅✨"

    "THIS STOCK IS GOING TO THE MOON 🚀🚀🚀"

    Any financial advice or guarantees.

    YOU MAY RECEIVE

    1. Real-time market data from Yahoo Finance
    2. Historical and recent news articles from FinMe's intelligence database

    USER QUESTION

    {user_query}

    REAL-TIME MARKET DATA

    {market_context}

    NEWS ARTICLES

    {news_context}

    INSTRUCTIONS

    - If market data exists, ALWAYS discuss it first.
    - Mention current price.
    - Mention daily change.
    - Mention daily percentage change.
    - Mention high and low when relevant.
    - Use news articles only when relevant.
    - Ignore unrelated news.
    - Explain financial concepts in simple language.
    - Use plain English.
    - Keep answers under 300 words.
    - Never recommend buying or selling.
    - Never make future predictions with certainty.

    RESPONSE STYLE

    For stock questions:

    1. Current stock performance
    2. Relevant news
    3. Simple explanation
    4. Friendly takeaway

    For market questions:

    1. Market mood
    2. Key drivers
    3. Important news
    4. Friendly takeaway

    For beginner questions:

    - Avoid jargon
    - Explain terms naturally
    - Teach without sounding condescending

    END EVERY RESPONSE WITH A SHORT TAKEAWAY

    Examples:

    "✨ Takeaway: TCS is having a solid day, but I'd keep an eye on upcoming company news before reading too much into a single trading session."

    "💖 Takeaway: The market looks cautiously optimistic today, with investors focusing on earnings and economic developments."

    Now generate the response.
    """

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                },
            },
            timeout=OLLAMA_TIMEOUT,
        )

        response.raise_for_status()

        report = (
            response.json()
            .get("message", {})
            .get("content", "")
            .strip()
        )

        if not report:
            return None

        return report

    except Exception as e:

        logger.warning(
            "❌ Report generation failed: %s",
            e,
        )

        return None


# --------------------------------------------------
# FALLBACK
# --------------------------------------------------

def _fallback_report(
    user_query: str,
    chunks: list[dict],
    market_data: dict | None,
    sources: list[dict],
):

    lines = []

    lines.append(
        f"Query: {user_query}\n"
    )

    if market_data:

        lines.append(
            f"""
Current Market Data

Symbol: {market_data.get("symbol")}
Price: {market_data.get("price")}
Change: {market_data.get("change")}
Change %: {market_data.get("change_pct")}
High: {market_data.get("day_high")}
Low: {market_data.get("day_low")}
Volume: {market_data.get("volume")}
"""
        )

    if chunks:

        lines.append(
            "\nRelevant News:\n"
        )

        for i, chunk in enumerate(chunks[:5], start=1):

            title = chunk.get(
                "title",
                "Unknown",
            )

            summary = (
                chunk.get("summary")
                or chunk.get("chunk", "")
            )[:200]

            lines.append(
                f"{i}. {title}\n{summary}\n"
            )

    return {
        "query": user_query,
        "report": "\n".join(lines),
        "sources": sources,
        "chunk_count": len(chunks),
        "method": "fallback",
        "generated_at": datetime.now().isoformat(),
    }