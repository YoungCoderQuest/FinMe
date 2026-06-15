import feedparser
import logging
from bs4 import BeautifulSoup
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from urllib.parse import urlparse

logger = logging.getLogger(__name__)
MAX_AGE_DAYS = 7
SOURCE_MAPPING = {
    "cnbc.com": "CNBC",
    "reuters.com": "Reuters",
    "finance.yahoo.com": "Yahoo Finance",
    "economictimes.indiatimes.com": "Economic Times",
    "business-standard.com": "Business Standard",
    "moneycontrol.com": "Moneycontrol",
    "investing.com": "Investing.com",
}
def clean_summary(summary):

    soup = BeautifulSoup(summary, "html.parser")

    return soup.get_text(separator=" ", strip=True)

def get_source_name(article_url):
    domain = urlparse(article_url).netloc.lower()

    for key, value in SOURCE_MAPPING.items():
        if key in domain:
            return value

    return domain

def is_fresh(published):

    try:

        article_date = parsedate_to_datetime(published)

        age = (
            datetime.now(timezone.utc)
            - article_date.astimezone(timezone.utc)
        )

        return age.days <= MAX_AGE_DAYS

    except Exception:

        return False

def to_iso_date(published):
    try:
        return (
            parsedate_to_datetime(published)
            .astimezone(timezone.utc)
            .isoformat()
        )
    except Exception:
        return None

def fetch_articles(feed_urls):
    articles = []
    for url in feed_urls:
        logger.info("📡 Fetching RSS feed: %s", url)
        try:
            feed = feedparser.parse(url)
            logger.info("   Found %d entries in feed", len(feed.entries))

            for entry in feed.entries:
                published = entry.get("published", "")
                
                if not is_fresh(published):
                    continue

                source_name = (
                    feed.feed.get("title")
                    or feed.feed.get("publisher")
                    or url
                )

                article = {
                    "title": entry.get("title", ""),
                    "link": entry.get("link", ""),
                    "published": to_iso_date(entry.get("published", "")),
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                    "summary": clean_summary(entry.get("summary", "")),
                    "source": get_source_name(entry.get("link", "")),
                    "feed_url": url,
                }

                articles.append(article)

        except Exception as e:
            logger.error("❌ Failed to parse %s: %s", url, e)

    logger.info("✅ Total articles fetched: %d", len(articles))
    return articles