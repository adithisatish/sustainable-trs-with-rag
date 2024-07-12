from src import * 

model = get_registry().get("sentence-transformers").create()

class WikivoyageDocuments(LanceModel):
    city: str = model.SourceField()
    country: str = model.SourceField()
    section: str = model.SourceField()
    text: str = model.SourceField()
    vector: Vector(model.ndims()) = model.VectorField()

class WikivoyageListings(LanceModel):
    city: str = model.SourceField()
    country: str = model.SourceField()
    type: str = model.SourceField()
    name: str = model.SourceField()
    description: str = model.SourceField()
    vector: Vector(model.ndims()) = model.VectorField()
