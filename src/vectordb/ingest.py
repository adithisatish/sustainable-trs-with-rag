from typing import Optional, Callable
import logging
import sys
sys.path.append("../")
logger = logging.getLogger(__name__)
logging.basicConfig(encoding='utf-8', level=logging.DEBUG)
from src.vectordb.helpers import read_docs, read_listings, preprocess_df
from src.vectordb.schema import WikivoyageDocuments, WikivoyageListings
from src.vectordb.helpers import set_uri
import lancedb


def _create_table_and_ingest_data(table_name: str, schema: object, data_fetcher: Callable,
                                  preprocessor: Optional[Callable] = None):
    """
    Generalized function to create a table and ingest data into the database.
    Args:
        - table_name: str, name of the table to create.
        - schema: object, schema of the table.
        - data_fetcher: Callable, function to fetch the data.
        - preprocessor: Optional[Callable], function to preprocess the data (default is None).
    """
    uri = set_uri(run_local=True)

    db = lancedb.connect(uri)
    logger.info(f"Connected to DB. Reading data for table {table_name} now...")

    df = data_fetcher()

    if preprocessor:
        df = preprocessor(df)

    logger.info(f"Finished reading data for {table_name}, attempting to create table and ingest the data...")

    db.drop_table(table_name, ignore_missing=True)
    table = db.create_table(table_name, schema=schema)

    table.add(df.to_dict('records'))
    logger.info(f"Completed ingestion for {table_name}.")


def create_wikivoyage_docs_db_and_add_data():
    """
    Creates the Wikivoyage documents table and ingests data.
    """
    _create_table_and_ingest_data(
        table_name="wikivoyage_documents",
        schema=WikivoyageDocuments,
        data_fetcher=read_docs,
        preprocessor=preprocess_df
    )


def create_wikivoyage_listings_db_and_add_data():
    """
    Creates the Wikivoyage listings table and ingests data.
    """
    _create_table_and_ingest_data(
        table_name="wikivoyage_listings",
        schema=WikivoyageListings,
        data_fetcher=read_listings
    )
