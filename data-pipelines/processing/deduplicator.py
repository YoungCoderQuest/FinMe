import hashlib
import logging

logger = logging.getLogger(__name__)


def get_content_hash(article):

    text = (
        article.get("title", "") +
        article.get("summary", "")
    )

    return hashlib.md5(
        text.lower().encode("utf-8")
    ).hexdigest()

def deduplicate_articles(articles):

    seen_urls = set()
    seen_hashes = set()

    unique_articles = []
    duplicates = 0

    for article in articles:

        # URL deduplication
        url = article.get("link")

        if url in seen_urls:
            duplicates += 1
            continue

        # Content deduplication
        content_hash = get_content_hash(article)

        if content_hash in seen_hashes:
            duplicates += 1
            continue

        seen_urls.add(url)
        seen_hashes.add(content_hash)

        unique_articles.append(article)

    logger.info("🔄 Hash dedup: %d articles → %d unique (removed %d duplicates)", 
                len(articles), len(unique_articles), duplicates)
    return unique_articles