"""
RSS Sources for FinMe

Reliability notes:
- MarketWatch: 401s due to paywall — replaced with WSJ free feed
- CNBC: occasional timeouts — kept, retried at extractor level
- Reuters: stable, no auth issues
- Indian sources: ET, BS, FE — stable for markets content
"""

RSS_FEEDS = [

    # ── Global Markets (reliable, no auth) ──────────────────

    # CNBC Markets
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",

    # Reuters Business
    "https://feeds.reuters.com/reuters/businessNews",

    # Reuters Markets / Wealth
    "https://feeds.reuters.com/news/wealth",

    # Investing.com (replaces MarketWatch — no paywall)
    "https://www.investing.com/rss/news.rss",

    # Yahoo Finance
    "https://finance.yahoo.com/news/rssindex",

    # ── Indian Markets ───────────────────────────────────────

    # Economic Times Markets
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",

    # Business Standard Markets
    "https://www.business-standard.com/rss/markets-106.rss",

    # Financial Express Markets
    "https://www.financialexpress.com/market/feed/",

    # Moneycontrol News (added — reliable Indian markets feed)
    "https://www.moneycontrol.com/rss/marketreports.xml",

]

# Sources known to block scrapers — kept here for reference, not in active feeds
BLOCKED_SOURCES = [
    "https://feeds.marketwatch.com/marketwatch/topstories/",  # 401 paywall
    "https://seekingalpha.com/feed.xml",                      # 197 char paywall page
]

# Add this at the bottom of config/rss_sources.py

SKIP_URL_PATTERNS = [
    "/liveblog/",
    "/live-blog/",
    "/live-updates/",
    "/livestock",
    "stock-liveblog",
]

def is_extractable_url(url: str) -> bool:
    url_lower = url.lower()
    return not any(pattern in url_lower for pattern in SKIP_URL_PATTERNS)