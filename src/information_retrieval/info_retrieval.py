import sys 
import os 
import re
from vectordb import vectordb
from sustainability import s_fairness
import json

import logging 

logger = logging.getLogger(__name__)
logging.basicConfig(encoding='utf-8', level=logging.DEBUG)

def get_travel_months(query):
    """
    
    Function to parse the user's query and search if month of travel has been provided by the user.

    Args:
    - query: str

    """
    months = [
        "January", "February", "March", "April", "May", "June", 
        "July", "August", "September", "October", "November", "December"
    ]

    seasons = {
            "spring": ["March", "April", "May"],
            "summer": ["June", "July", "August"],
            "fall": ["September", "October", "November"],
            "autumn": ["September", "October", "November"],  
            "winter": ["December", "January", "February"]
        }

    months_in_query = []
    
    for month in months:
        if re.search(r'\b' + month + r'\b', query, re.IGNORECASE):
            months_in_query.append(month)
    
    # Check for seasons in the query
    for season, season_months in seasons.items():
        if re.search(r'\b' + season + r'\b', query, re.IGNORECASE):
            months_in_query += season_months
    
    # Return None if neither months nor seasons are found
    return months_in_query

def get_wikivoyage_context(query, limit = 10, reranking = 0):
    """
    
    Function to retrieve the relevant documents and listings from the wikivoyage database. Works in two steps: 
    (i) the relevant cities are returned by the wikivoyage_docs table and (ii) then passed on to the wikivoyage listings database to retrieve further information. 
    The user can pass a limit of how many results the search should return as well as whether to perform reranking (uses a CrossEncoderReranker)

    Args: 
        - query: str
        - limit: int
        - reranking: bool

    """

    # limit = params['limit']
    # reranking = params['reranking']

    docs = vectordb.search_wikivoyage_docs(query, limit, reranking)
    logger.info("Finished getting chunked wikivoyage docs.")

    results = {}
    for doc in docs: 
        results[doc['city']] = {key: value for key, value in doc.items() if key != 'city'}
        results[doc['city']]['listings'] = []

    cities = [result['city'] for result in docs] 

    listings = vectordb.search_wikivoyage_listings(query, cities, limit, reranking)
    logger.info("Finished getting wikivoyage listings.")
    # logger.info(type(docs), type(listings))

    for listing in listings: 
        # logger.info(listing['city'])
        results[listing['city']]['listings'].append({
            'type': listing['type'],
            'name': listing['title'],
            'description': listing['description']
        })

    logger.info("Returning retrieval results.")
    return results

def get_sustainability_scores(query, destinations):
    """
    
    Function to get the s-fairness scores for each destination for the given month (or the ideal month of travel if the user hasn't provided a month). 
    If multiple months are provided (or season), then the month with the minimum s-fairness score is chosen for the city.

    Args: 
        - query: str 
        - destinations: list
    
    """

    result = [] #list of dicts of the format {city: <city>, month: <month>, }
    city_scores = {}

    months = get_travel_months(query)
    logger.info("Finished parsing query for months.")

    for city in destinations:
        if city not in city_scores:
            city_scores[city] = []

        if not months: # no month(s) or seasons provided by the user
            city_scores[city].append(s_fairness.compute_sfairness_score(city))
        else:
            for month in months:
                city_scores[city].append(s_fairness.compute_sfairness_score(city, month))

    logger.info("Finished getting s-fairness scores.")

    for city, scores in city_scores.items():
        
        no_result = 0
        for score in scores:
            if not score['month']:
                no_result = 1
                result.append({
                    'city': city,
                    'month': 'No data available',
                    's-fairness': 'No data available'
                })
                break
        
        if not no_result:
            min_score = min(scores, key=lambda x: x['s-fairness'])
            result.append({
                'city': city,
                'month': min_score['month'],
                's-fairness': min_score['s-fairness']
            })

    logger.info("Returning s-fairness results.")
    return result

def get_cities(context):
    """
    Only to be used for testing! Function that returns a list of cities with their s-fairness scores, provided the retrieved context

    Args:
        - context: dict
    
    """

    recommended_cities = []

    for city, info in context: 
        city_info = {
            'city': city,
            'country': info['country']
        }

        if "sustainability" in info: 
            city_info['month'] = info['sustainability']['month']
            city_info['s-fairness'] = info['sustainability']['s-fairness']

        recommended_cities.append(city_info)
    
    return recommended_cities
        

def get_context(query, **params):
    """
    
    Function that returns all the context: from the database, as well as the respective s-fairness scores for the destinations. The default does not consider S-Fairness scores, i.e. to append sustainability scores, a non-zero parameter "sustainability" needs to be explicitly passed to params.

    Args:
        - query: str
        - params: dict; contains value of the limit and reranking (and sustainability)
    
    """

    limit = 3 
    reranking = 1

    if 'limit' in params:
        limit = params['limit'] 
    
    if 'reranking' in params: 
        reranking = params['reranking']

    wikivoyage_context = get_wikivoyage_context(query, limit, reranking)
    recommended_cities = wikivoyage_context.keys()

    if 'sustainability' in params and params['sustainability']:
        s_fairness_scores = get_sustainability_scores(query, recommended_cities)

        for score in s_fairness_scores: 
            wikivoyage_context[score['city']]['sustainability'] = {
                'month': score['month'],
                's-fairness': score['s-fairness']
            }

    return wikivoyage_context

def test():
    queries = []
    query = "Suggest some places to visit during winter. I like hiking, nature and the mountains and I enjoy skiing in winter."

    context = None

    try: 
        context = get_context(query)
    except FileNotFoundError as e:
        try: 
            vectordb.create_wikivoyage_docs_db_and_add_data()
            vectordb.create_wikivoyage_listings_db_and_add_data() 

            try:
                context = get_context(query)
            except Exception as e: 
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                logger.error(f"Error while getting context: {e}, {(exc_type, fname, exc_tb.tb_lineno)}")

        except Exception as e:
            logger.error(f"Error while creating DB: {e}")

    except Exception as e: 
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error(f"Error while getting context: {e}, {(exc_type, fname, exc_tb.tb_lineno)}")
    
    file_path = os.path.join(os.getcwd(), "test_results", "test_result.json")
    with open(file_path, 'w') as file: 
        json.dump(context, file)

    return context

if __name__ == "__main__":
    context = test()

    print(context)