from vectordb import *
import logging 

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def run():
    logging.info("Creating database for Wikivoyage Documents")
    try:
        create_wikivoyage_docs_db_and_add_data()
    except Exception as e:
        logger.error(f"Error for Wikivoyage Documents: {e}")

    logging.info("Creating database for Wikivoyage Listings")
    try:
        create_wikivoyage_listings_db_and_add_data()
    except Exception as e:
        logger.error(f"Error for Wikivoyage Listings: {e}")

if __name__ == "__main__":
    run()