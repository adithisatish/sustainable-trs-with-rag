from typing import Optional

import pandas as pd
import os
import re
from sentence_transformers import SentenceTransformer
import sys
sys.path.append("../")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.data_directories import *


def create_chunks(city, country, text):
    """
    
    Helper function that creates chunks given paragraph(s) of text based on implicit sections in the text. 

    Args: 
        - city: str
        - country: str
        - text: str; document that needs to be chunked

    """

    for i, line in enumerate(text):
        if line[0] == "\n":
            del text[i]

    index = 0
    chunks = []
    pattern = re.compile("==")
    ignore = re.compile("===")
    section = 'Introduction'
    for i, line in enumerate(text):
        if pattern.search(line) and not ignore.search(line):
            chunk = ''.join(text[index:i])
            chunks.append({
                'city': city,
                'country': country,
                'section': section,
                'text': chunk,
                # 'vector': f'city: {city}, country: {country}, section: {section}, text: {chunk}' 
            })
            index = i + 1
            section = re.sub(pattern, '', line).strip()

    df = pd.DataFrame(chunks)
    return df


def read_docs():
    """
    
    Helper function that reads all of the Wikivoyage documents containing information about the city. 

    """

    df = pd.DataFrame()
    cities = pd.read_csv(cities_csv)
    for file_name in os.listdir(wikivoyage_docs_dir + "cleaned/"):
        city = file_name.split(".")[0]
        # print(city)
        country = cities[cities['city'] == city]['country'].item()
        with open(wikivoyage_docs_dir + "cleaned/" + file_name) as file:
            text = file.readlines()
            chunk_df = create_chunks(city, country, text)
            df = pd.concat([df, chunk_df])

    return df


def read_listings():
    """
    
    Helper function that reads the Wikivoyage listings csv containing tabular information about 144 cities. 

    """
    df = pd.read_csv(wikivoyage_listings_dir + "wikivoyage-listings-cleaned.csv")
    cities = pd.read_csv(cities_csv)

    def find_country(city):
        return cities[cities['city'] == city]['country'].values[0]

    df['country'] = df['city'].apply(find_country)

    return df


def preprocess_df(df):
    """
    
    Helper function that preprocesses the dataframe containing chunks of text and removes hyperlinks and strips the \n from the text. 

    Args:
        - df: dataframe

    """
    section_counts = df['section'].value_counts()
    sections_to_keep = section_counts[section_counts > 150].index
    filtered_df = df[df['section'].isin(sections_to_keep)]

    def preprocess_text(s):
        s = re.sub(r'http\S+', '', s)
        s = re.sub(r'=+', '', s)
        s = s.strip()
        return s

    filtered_df['text'] = filtered_df['text'].apply(preprocess_text)

    return filtered_df


def compute_wv_docs_embeddings(df):
    """
    
    Helper function that computes embeddings for the text. The all-MiniLM-L6-v2 embedding model is used.  

    Args:
        - df: dataframe

    """
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    vector_dimension = model.get_sentence_embedding_dimension()

    print("Computing embeddings")
    embeddings = []
    for i, row in df.iterrows():
        emb = model.encode(row['combined'], show_progress_bar=True).tolist()
        embeddings.append(emb)

    print("Finished computing embeddings for wikivoyage documents.")
    df['vector'] = embeddings
    # df.to_csv(wv_embeddings + "wikivoyage-listings-embeddings.csv")
    # print("Finished saving file.")
    return df


def embed_query(query):
    """
    
    Helper function that returns the embedded query. 

    Args:
        - query: str
    
    """
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # vector_dimension = model.get_sentence_embedding_dimension()   
    embedding = model.encode(query).tolist()
    return embedding


def set_uri(run_local: Optional[bool] = False):
    if run_local:
        uri = database_dir
        current_dir = os.path.split(os.getcwd())[1]

        if "src" or "tests" in current_dir:  # hacky way to get the correct path
            uri = uri.replace("../../", "../")
    else:
        uri = os.environ["BUCKET_NAME"]
    return uri
