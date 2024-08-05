"""
Script to run to use GPT4 

"""

import os
import sys
from openai import OpenAI
from augmentation import prompt_generation as pg
from information_retrieval import info_retrieval as ir
from text_generation.model_init import (
    Llama3,
    Mistral,
    Gemma2,
    Llama3Point1,
    Llama3Instruct,
    MistralInstruct,
    Llama3Point1Instruct,
)
from text_generation import text_generation as tg
import logging 

logger = logging.getLogger(__name__)
logging.basicConfig(encoding='utf-8', level=logging.DEBUG)

TEST_DIR = "../tests/"
MODELS = {
        'Llama3': Llama3, 
        'Mistral': Mistral, 
        'Gemma2': Gemma2, 
        'Llama3.1': Llama3Point1,
        'Llama3-Instruct': Llama3Instruct,
        'Mistral-Instruct': MistralInstruct,
        'Llama3.1-Instruct': Llama3Point1Instruct,
    }


def pipeline(query, test = 0, **params):
    """
    
    Executes the entire RAG pipeline with GPT-4o, provided the query

    Args: 
        - query: str
        - test: whether the pipeline is running a test
        - params: 
            - limit (number of results to be retained) 
            - reranking (binary, whether to rerank results using ColBERT or not)
            - sustainability

    
    """

    context_params = {
        'limit': 5,
        'reranking': 0,
        'sustainability': 0
    }

    if 'limit' in params:
        context_params['limit'] = params['limit'] 
    
    if 'reranking' in params: 
        context_params['reranking'] = params['reranking']
    
    if 'sustainability' in params: 
        context_params['sustainability'] = params['sustainability']


    logger.info("Retrieving context..")
    try:
        context = ir.get_context(query=query, **context_params)
        if test: 
            retrieved_cities = ir.get_cities(context)

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error(f"Error at line {exc_tb.tb_lineno} while trying to get context: {e}")
        return None
    
    logger.info("Retrieved context, augmenting prompt..")
    try:
        prompt = pg.augment_prompt(
            query=query, 
            context=context,
            sustainability=0,
            params=context_params
        )
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error(f"Error at line {exc_tb.tb_lineno} while trying to augment prompt: {e}")
        return None
    
    # return without_sfairness

    logger.info(f"Augmented prompt, initializing GPT-4o and generating response..")
    try:
        API_KEY = "<add OpenAI API key here>" # can also be added as an environment variable and accessed here
        client = OpenAI(api_key=API_KEY)

        completion = client.chat.completions.create(
        model="gpt-4o",
        messages=prompt,
        temperature=0.6,
        top_p=0.9,
        )

        response = completion.choices[0].message
    except Exception as e: 
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.info(f"Error at line {exc_tb.tb_lineno} while generating response: {e}")
        return None

    if test:
        return (retrieved_cities, response)

    else:
        return response
    
if __name__ == "__main__":
    query = "I'm planning a trip in the summer and I love art, history, and visiting museums. Can you suggest some European cities?"
    model_name = "Llama3"

    response = pipeline(
        query=query
    )

    print(response)