# from src import * 
from src.vectordb.helpers import *
from src.vectordb.lancedb_init import *
import logging
import os
import lancedb
from lancedb.rerankers import ColbertReranker

import sys
logger = logging.getLogger(__name__)


# db = lancedb.connect("/tmp/db")

def create_wikivoyage_docs_db_and_add_data():
    """
    
    Creates wikivoyage documents table and ingests data

    """
    uri = database_dir
    current_dir = os.path.split(os.getcwd())[1]

    if "src" or "tests" in current_dir: # hacky way to get the correct path
        uri = uri.replace("../../", "../")

    db = lancedb.connect(uri)
    logger.info("Connected to DB. Reading data now...")
    df = read_docs()
    filtered_df = preprocess_df(df)
    logger.info("Finished reading data, attempting to create table and ingest the data...")

    db.drop_table("wikivoyage_documents", ignore_missing=True)
    table = db.create_table("wikivoyage_documents", schema=WikivoyageDocuments)

    table.add(filtered_df.to_dict('records'))
    logger.info("Completed ingestion.")


def create_wikivoyage_listings_db_and_add_data():
    """
    
    Creates wikivoyage listings table and ingests data

    """
    uri = database_dir
    current_dir = os.path.split(os.getcwd())[1]

    if "src" or "tests" in current_dir: # hacky way to get the correct path
        uri = uri.replace("../../", "../")
        
    db = lancedb.connect(uri)
    logger.info("Connected to DB. Reading data now...")
    df = read_listings()
    logger.info("Finished reading data, attempting to create table and ingest the data...")
    # filtered_df = preprocess_df(df)

    db.drop_table("wikivoyage_listings", ignore_missing=True)
    table = db.create_table("wikivoyage_listings", schema=WikivoyageListings)

    table.add(df.astype('str').to_dict('records'))
    logger.info("Completed ingestion.")


def search_wikivoyage_docs(query, limit=10, reranking=0):
    """
    
    Function to search the wikivoyage database an return most relevant chunked docs. 

    Args: 
        - query: str
        - limit: number of results (default is 10)
        - reranking: bool (0 or 1), if activated, CrossEncoderReranker is used.

    """

    uri = database_dir
    current_dir = os.path.split(os.getcwd())[1]

    if "src" or "tests" in current_dir: # hacky way to get the correct path
        uri = uri.replace("../../", "../")
    # print(uri)

    db = lancedb.connect(uri)
    logger.info("Connected to Wikivoyage DB.")

    # query_embedding = embed_query(query)
    table = db.open_table("wikivoyage_documents")

    if reranking:
        try:
            reranker = ColbertReranker(column='text')
            results = table.search(query) \
                .metric('cosine') \
                .rerank(reranker=reranker) \
                .limit(limit) \
                .to_list()
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error(f"Error while getting context: {e}, {(exc_type, fname, exc_tb.tb_lineno)}")

    else:
        try:
            results = table.search(query) \
                .limit(limit) \
                .metric('cosine') \
                .to_list()
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error(f"Error while getting context: {e}, {(exc_type, fname, exc_tb.tb_lineno)}")

    logger.info("Found the most relevant documents.")
    city_lists = [{"city": r['city'], "country": r['country'], "section": r['section'], "text": r['text']} for r in
                  results]

    # context = [f"city: {r['city']}, country: {r['country']}, name: {r['title']}, description: {r['description']}" for r in results]

    return city_lists


def search_wikivoyage_listings(query, cities, limit=10, reranking=0):
    """
    
    Function to search the wikivoyage database an return most relevant listings, post-filtered by the recommended
    cities provided by wikivoyage_documents table.

    Args: 
        - query: str
        - cities: list
        - limit: number of results (default is 10)
        - reranking: bool (0 or 1), if activated, CrossEncoderReranker is used.

    """
    uri = database_dir
    current_dir = os.path.split(os.getcwd())[1]

    if "src" or "tests" in current_dir: # hacky way to get the correct path
        uri = uri.replace("../../", "../")

    db = lancedb.connect(uri)
    logger.info("Connected to Wikivoyage Listings DB.")

    table = db.open_table("wikivoyage_listings")

    cities_filter = f"city IN {tuple(cities)}"

    if reranking:
        try:
            reranker = ColbertReranker(column='description')
            results = table.search(query) \
                .where(cities_filter) \
                .metric('cosine') \
                .rerank(reranker=reranker) \
                .limit(limit) \
                .to_list()

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error(f"Error while getting context: {e}, {(exc_type, fname, exc_tb.tb_lineno)}")

    else:
        try:
            results = table.search(query) \
                .where(cities_filter) \
                .metric('cosine') \
                .limit(limit) \
                .to_list()
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error(f"Error while getting context: {e}, {(exc_type, fname, exc_tb.tb_lineno)}")

    logger.info("Found the most relevant documents.")
    city_listings = [{"city": r['city'], "country": r['country'], "type": r['type'], "title": r['title'],
                      "description": r['description']} for r in results]

    return city_listings
