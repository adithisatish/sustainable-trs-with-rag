import os

current = os.path.dirname(os.path.realpath(''))
parent = os.path.dirname(current)

data_parent_dir = "../../european-city-data/"
data_dir = data_parent_dir + "data-sources/"
wikivoyage_docs_dir = data_dir + "wikivoyage/"
wikivoyage_listings_dir = wikivoyage_docs_dir + "listings/"
database_dir = "../../database/wikivoyage/"
seasonality_dir = data_parent_dir + "computed/seasonality/"
popularity_dir = data_parent_dir + "computed/popularity/"
cities_csv = data_parent_dir + "city_abstracts_embeddings.csv"
prompts_dir = data_parent_dir + "rag-sustainability/prompts/"
results_dir = data_parent_dir + "rag-sustainability/results/"
