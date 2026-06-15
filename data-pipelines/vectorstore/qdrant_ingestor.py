import logging
from datetime import datetime, timezone, timedelta
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PayloadSchemaType

logger = logging.getLogger(__name__)


def search_similar(client, collection_name, vector, limit=1, days_back: int = None):
    """
    Semantic search with optional date filter.
    Date filtering is done post-search in Python for compatibility.

    Args:
        days_back: If set, only return chunks published within last N days.
                   None means no date filter (search all time).
    """
    # Fetch more results when filtering so we still get enough after date filter
    fetch_limit = limit * 3 if days_back is not None else limit

    results = client.query_points(
        collection_name=collection_name,
        query=vector,
        limit=fetch_limit,
        with_payload=True,
    ).points

    if days_back is None:
        return results[:limit]

    # Post-filter by published date in Python
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    filtered = []

    for r in results:
        published = r.payload.get("published") if r.payload else None
        if not published:
            continue
        try:
            # Parse ISO string — handle both +00:00 and Z suffixes
            pub_str = published.replace("Z", "+00:00")
            pub_dt  = datetime.fromisoformat(pub_str)
            if pub_dt >= cutoff:
                filtered.append(r)
        except Exception:
            # If we can't parse the date, include it anyway
            filtered.append(r)

    logger.debug(
        "[db] Date filter: %d/%d results within last %d days",
        len(filtered), len(results), days_back
    )

    return filtered[:limit]


def url_exists(client, collection_name, url: str) -> bool:
    """Check if an article URL is already stored in Qdrant."""
    try:
        results = client.scroll(
            collection_name=collection_name,
            scroll_filter={
                "must": [
                    {
                        "key": "url",
                        "match": {"value": url}
                    }
                ]
            },
            limit=1,
            with_payload=False,
        )
        return len(results[0]) > 0
    except Exception:
        return False


def create_collection(client, collection_name, vector_size=384):
    collections = [
        c.name for c in client.get_collections().collections
    ]

    if collection_name not in collections:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        logger.info("[db] Created collection: %s", collection_name)
    else:
        logger.info("[db] Collection already exists: %s", collection_name)

    # Create payload index on 'published' for fast date filtering
    # Safe to call repeatedly — Qdrant ignores if index already exists
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="published",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        logger.info("[db] Payload index ready: published (keyword)")
    except Exception as e:
        logger.warning("[db] Could not create published index: %s", e)


def insert_chunks(client, collection_name, points):
    if not points:
        return

    # Deduplicate by chunk text before inserting
    seen = set()
    unique_points = []
    for p in points:
        chunk_text = p.payload.get("chunk", "")
        if chunk_text not in seen:
            seen.add(chunk_text)
            unique_points.append(p)

    if len(unique_points) < len(points):
        logger.info(
            "[db] Removed %d duplicate chunks before insert",
            len(points) - len(unique_points)
        )

    client.upsert(
        collection_name=collection_name,
        points=unique_points
    )