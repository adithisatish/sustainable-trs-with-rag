import sys
import os
import pandas as pd 
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from data_directories import *

def get_popularity(destination):
    """
    
    Returns the popularity score for a particular destination.

    Args:
        - destination: str
    
    """
    popularity_df = pd.read_csv(popularity_dir + "popularity_scores.csv")
    return popularity_df[popularity_df['city'] == destination]['weighted_pop_score'].item()

def get_seasonality(destination, month = None):
    """
    
    Returns the seasonality score for a particular destination for a particular month. If no month is provided then the best month, i.e. month of lowest seasonality is returned. 

    Args:
        - destination: str
        - month: str (default: None)
    
    """
    seasonality_df = pd.read_csv(seasonality_dir + "seasonality_scores.csv")
    if month:
        m = month.capitalize()[:3]
    else: 
        seasonality_df['lowest_col'] = seasonality_df.loc[:, seasonality_df.columns != 'city'].idxmin(axis="columns")
        m = seasonality_df[seasonality_df['city'] == destination]['lowest_col'].item()

    return  (m, seasonality_df[seasonality_df['city'] == destination][m].item())


def compute_sfairness_score(destination, month = None):
    """
    
    Returns the s-fairness score for a particular destination city and (optional) month.

    Args:
        - destination: str
        - month: str (default: None)
    
    """
    seasonality = get_seasonality(destination, month = None)
    month = seasonality[0]
    popularity = get_popularity(destination)
    emissions = 0

    s_fairness = 0.281 * emissions + 0.334 * popularity + 0.385 * seasonality[1]

    return {
            'month': month, 
            's-fairness': s_fairness
        }

if __name__ == "__main__":
    print(compute_sfairness_score("Paris", "Oct"))