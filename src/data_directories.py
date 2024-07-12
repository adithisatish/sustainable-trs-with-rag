import sys 
import os

current = os.path.dirname(os.path.realpath(''))
parent = os.path.dirname(current)



data_dir = "../../european-city-data/data-sources/"
wikivoyage_docs_dir = data_dir + "wikivoyage/"
wikivoyage_listings_dir = wikivoyage_docs_dir + "listings/"
database_dir = "../../database/wikivoyage/"
seasonality_dir = "../../european-city-data/computed/seasonality/"
popularity_dir = "../../european-city-data/computed/popularity/"