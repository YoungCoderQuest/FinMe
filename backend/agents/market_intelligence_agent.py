"""
Market Intelligence Agent

Combines:
1. Real-time market data (Yahoo Finance)
2. Historical/news intelligence (Qdrant)
3. LLM synthesis (Ollama)

Output:
{
    market_data,
    news_chunks,
    intelligence_report
}
"""

import logging
import re
import numpy as np

from embeddings.embedder import generate_embeddings
from vectorstore.qdrant_ingestor import search_similar
from agents.report_agent import generate_report
from market.market_service import get_quote

logger = logging.getLogger(__name__)


# ----------------------------------------------------
# Symbol extraction
# ----------------------------------------------------

KNOWN_SYMBOLS = {
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "RELIANCE": "RELIANCE.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "SBIN": "SBIN.NS",
    "WIPRO": "WIPRO.NS",
    "LT": "LT.NS",
    "ITC": "ITC.NS",
    "AXISBANK": "AXISBANK.NS",
}


def extract_symbol(query: str):
    upper = query.upper()

    for company in KNOWN_SYMBOLS:
        if company in upper:
            return KNOWN_SYMBOLS[company]

    return None


# ----------------------------------------------------
# Agent
# ----------------------------------------------------

def generate_market_intelligence(
    user_query: str,
    qdrant_client,
    collection_name: str,
):
    """
    Main market intelligence workflow.
    """

    symbol = extract_symbol(user_query)

    market_data = None

    if symbol:
        try:
            market_data = get_quote(symbol)
        except Exception as e:
            logger.warning("Market data fetch failed: %s", e)

    # ----------------------------------------
    # News retrieval from Qdrant
    # ----------------------------------------

    query_vector = np.array(
        generate_embeddings([user_query])[0]
    )

    results = search_similar(
        qdrant_client,
        collection_name,
        query_vector,
        limit=8,
    )

    payloads = []

    for r in results:
        if r.payload:
            payloads.append(r.payload)

    # ----------------------------------------
    # Build context for LLM
    # ----------------------------------------

    context = ""

    if market_data:
        context += f"""
Current Market Data:
Symbol: {market_data.get("symbol")}
Price: {market_data.get("price")}
Change: {market_data.get("change")}
Change Percent: {market_data.get("change_pct")}
Day High: {market_data.get("day_high")}
Day Low: {market_data.get("day_low")}
Volume: {market_data.get("volume")}
"""

    for idx, article in enumerate(payloads[:5], start=1):
        context += f"""

News {idx}
Title: {article.get('title')}
Summary: {article.get('summary')}
"""

    intelligence_report = generate_report(
        user_query=user_query,
        chunks=payloads,
        market_data=market_data,
    )

    return {
        "market_data": market_data,
        "market_context": {
            "price": market_data.get("price") if market_data else None,
            "change": market_data.get("change") if market_data else None,
            "change_pct": market_data.get("change_pct") if market_data else None,
            "day_high": market_data.get("day_high") if market_data else None,
            "day_low": market_data.get("day_low") if market_data else None,
            "volume": market_data.get("volume") if market_data else None,
        },
        "articles_used": [
            {
                "title": p.get("title"),
                "url": p.get("url"),
                "published": p.get("published"),
                "source": p.get("source_name"),
            }
            for p in payloads[:5]
        ],
        "news_chunks": payloads,
        "intelligence_report": intelligence_report,
    }