import os
import sys
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


model = get_registry().get("sentence-transformers").create()


class WikivoyageDocuments(LanceModel):
    """
    
    Schema definition for the Wikivoyage Documents table.

    """
    city: str = model.SourceField()
    country: str = model.SourceField()
    section: str = model.SourceField()
    text: str = model.SourceField()
    vector: Vector(model.ndims()) = model.VectorField()


class WikivoyageListings(LanceModel):
    """
    
    Schema definition for the Wikivoyage Listings table.

    """
    city: str = model.SourceField()
    type: str = model.SourceField()
    title: str = model.SourceField()
    description: str = model.SourceField()
    country: str = model.SourceField()
    vector: Vector(model.ndims()) = model.VectorField()
