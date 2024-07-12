import pandas as pd 
import lancedb
import numpy as np 
import os 
import re 
from sentence_transformers import SentenceTransformer
from lancedb.rerankers import CrossEncoderReranker
from lancedb.embeddings import get_registry
from lancedb.util import attempt_import_or_raise
from lancedb.pydantic import LanceModel, Vector

data_dir = "../../european-city-data/data-sources/"
wikivoyage_docs_dir = data_dir + "wikivoyage/"
wikivoyage_listings_dir = wikivoyage_docs_dir + "listings/"
database_dir = "../../database/wikivoyage/"
seasonality_dir = "../../european-city-data/computed/seasonality/"
popularity_dir = "../../european-city-data/computed/popularity/"

model = get_registry().get("sentence-transformers").create()

class WikivoyageDocuments(LanceModel):
    city: str = model.SourceField()
    country: str = model.SourceField()
    section: str = model.SourceField()
    text: str = model.SourceField()
    vector: Vector(model.ndims()) = model.VectorField()

class WikivoyageListings(LanceModel):
    city: str = model.SourceField()
    type: str = model.SourceField()
    title: str = model.SourceField()
    description: str = model.SourceField()
    country: str = model.SourceField()
    vector: Vector(model.ndims()) = model.VectorField()
