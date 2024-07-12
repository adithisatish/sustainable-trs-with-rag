from src import * 
from helpers import * 
from lancedb_init import * 

# db = lancedb.connect("/tmp/db")

def create_wikivoyage_docs_db_and_add_data():
    uri = database_dir
    db = lancedb.connect(uri)
    df = read_docs()
    filtered_df = preprocess_df(df)

    db.drop_table("wikivoyage_documents", ignore_missing=True)
    table = db.create_table("wikivoyage_documents", schema=WikivoyageDocuments)

    table.add(filtered_df.to_dict('records'))

def create_wikivoyage_listings_db_and_add_date():
    uri = database_dir
    db = lancedb.connect(uri)
    df = read_listings()
    # filtered_df = preprocess_df(df)

    db.drop_table("wikivoyage_listings", ignore_missing=True)
    table = db.create_table("wikivoyage_listings", schema=WikivoyageDocuments)

    table.add(df.to_dict('records'))

def search_wikivoyage_docs(query, limit=10, reranking = 0):
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