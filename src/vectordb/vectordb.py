# from src import * 
from helpers import * 
from lancedb_init import *
import logging

logger = logging.getLogger(__name__)

# db = lancedb.connect("/tmp/db")

def create_wikivoyage_docs_db_and_add_data():
    """
    
    Creates wikivoyage documents table and ingests data

    """
    uri = database_dir
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
    db = lancedb.connect(uri)
    logger.info("Connected to DB. Reading data now...")
    df = read_listings()
    logger.info("Finished reading data, attempting to create table and ingest the data...")
    # filtered_df = preprocess_df(df)

    db.drop_table("wikivoyage_listings", ignore_missing=True)
    table = db.create_table("wikivoyage_listings", schema=WikivoyageListings)

    table.add(df.astype('str').to_dict('records'))
    logger.info("Completed ingestion.")

def search_wikivoyage_docs(query, limit=10, reranking = 0):
    """
    
    Function to search the wikivoyage database an return most relevant chunked docs. 

    Args: 
        - query: str
        - limit: number of results (default is 10)
        - reranking: bool (0 or 1), if activated, CrossEncoderReranker is used.

    """
    uri = database_dir
    db = lancedb.connect(uri)
    print("Connected to DB.")

    # query_embedding = embed_query(query)
    table = db.open_table("wikivoyage_documents")

    if reranking: 
        reranker = CrossEncoderReranker()
        results = table.search(query).rerank(reranker=reranker).limit(limit).to_list()

    else:
        results = table.search(query).limit(limit).to_list()
        
    print("Found the most relevant documents.")
    city_lists = [f"city: {r['city']}, country: {r['country']}, section: {r['section']}, text: {r['text']}" for r in results]

    # context = [f"city: {r['city']}, country: {r['country']}, name: {r['title']}, description: {r['description']}" for r in results]

    return city_lists

# TO DO:
# def search_wikivoyage_listings(query, limit=10, reranking = 0):
#     uri = database_dir
#     db = lancedb.connect(uri)
#     print("Connected to DB.")

#     pass