from qdrant_client import QdrantClient
from config.rss_sources import RSS_FEEDS
# from pipeline.run_pipeline import IngestionPipeline
from agents.langgraph_pipeline import LangGraphPipeline as IngestionPipeline
from vectorstore.qdrant_ingestor import create_collection
import logging

# Configure logging with cleaner format
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

logger = logging.getLogger(__name__)


def main():
    logger.info("\n" + "="*80)
    logger.info("🏦 FinMe - Financial News Ingestion Pipeline")
    logger.info("="*80 + "\n")

    client = QdrantClient(host="localhost", port=6333)
    collection_name = "financial_news"

    logger.info("🔐 Connecting to Qdrant at localhost:6333...")
    try:
        # 🔥 MUST CREATE BEFORE PIPELINE
        create_collection(
            client=client,
            collection_name=collection_name,
            vector_size=384
        )
        logger.info("✅ Collection ready: %s\n", collection_name)
    except Exception as e:
        logger.error("❌ Failed to create collection: %s", e)
        return

    pipeline = IngestionPipeline(
        client=client,
        collection_name=collection_name
    )

    pipeline.run(RSS_FEEDS)


if __name__ == "__main__":
    main()