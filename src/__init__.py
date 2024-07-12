import pandas as pd 
import lancedb
import numpy as np 
import os 
import re 
from sentence_transformers import SentenceTransformer
from lancedb.rerankers import CrossEncoderReranker
from data_directories import *
from lancedb.embeddings import get_registry
from lancedb.util import attempt_import_or_raise
from lancedb.pydantic import LanceModel, Vector