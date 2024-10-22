import logging
import os
import lancedb
from lancedb.rerankers import ColbertReranker

import sys

logger = logging.getLogger(__name__)
from typing import Optional
from src.vectordb.helpers import set_uri


# db = lancedb.connect("/tmp/db")


def search(query: str, table_name: str, filter_condition: Optional[str] = None,
           category: str = "docs", limit: int = 10, reranking: int = 0,
           run_local: Optional[bool] = False) -> list | None:
    """
    Generalized function to search a database table, with optional filters and reranking.
    Args:
        - query: str, search query.
        - table_name: str, name of the table to search.
        - filter_condition: Optional[str], optional SQL-like condition for filtering results.
        - category: str, type of category (default is 'docs').
        - limit: int, number of results (default is 10).
        - reranking: int (0 or 1), if activated, ColbertReranker is used.
        - run_local: Optional[bool], whether to run in a local environment.
    Returns:
        A list of the most relevant documents or listings based on the category.
    """
    uri = set_uri(run_local)

    try:
        db = lancedb.connect(uri)
    except Exception as e:
        logger.error(f"Error while connecting to DB: {e}")
        return None

    logger.info(f"Connected to {table_name} DB.")
    table = db.open_table(table_name)

    search_query = table.search(query).metric('cosine')

    if filter_condition:
        search_query = search_query.where(filter_condition)

    if reranking:
        try:
            column = 'description' if category == 'listings' else 'text'
            reranker = ColbertReranker(column=column)
            results = search_query.rerank(reranker=reranker).limit(limit).to_list()
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error(f"Error while reranking results: {e}, {(exc_type, fname, exc_tb.tb_lineno)}")
            return None
    else:
        try:
            results = search_query.limit(limit).to_list()
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error(f"Error while searching: {e}, {(exc_type, fname, exc_tb.tb_lineno)}")
            return None

    logger.info("Found the most relevant documents.")

    if category == "docs":
        return [{"city": r['city'], "country": r['country'], "section": r['section'], "text": r['text']} for r in
                results]
    else:
        return [{"city": r['city'], "country": r['country'], "type": r['type'], "title": r['title'],
                 "description": r['description']} for r in results]


def search_wikivoyage_docs(query: str, limit: int = 10, reranking: int = 0,
                           run_local: Optional[bool] = False) -> list | None:
    """
    Function to search documents in the Wikivoyage database.
    """
    return search(query=query, table_name="wikivoyage_documents", category="docs",
                  limit=limit, reranking=reranking, run_local=run_local)


def search_wikivoyage_listings(query: str, cities: list, limit: int = 10, reranking: int = 0,
                               run_local: Optional[bool] = False) -> list | None:
    """
    Function to search listings in the Wikivoyage database, post-filtered by cities.
    """
    cities_filter = f"city IN {tuple(cities)}"
    return search(query=query, table_name="wikivoyage_listings", filter_condition=cities_filter,
                  category="listings", limit=limit, reranking=reranking, run_local=run_local)

