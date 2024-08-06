"""
Main file to execute the TRS Pipeline.
"""
import os
import sys
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

def pipeline(query, model_name, test = 0, **params):
    """
    
    Executes the entire RAG pipeline, provided the query and model class name.

    Args: 
        - query: str
        - model_name: string, one of the following: Llama3, Mistral, Gemma2, Llama3Point1
        - test: whether the pipeline is running a test
        - params: 
            - limit (number of results to be retained) 
            - reranking (binary, whether to rerank results using ColBERT or not)
            - sustainability

    
    """

    model = MODELS[model_name]

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

    logger.info(f"Augmented prompt, initializing {model} and generating response..")
    try:
        response = tg.generate_response(model, prompt)
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
    model_name = "Mistral-Instruct"

    response = pipeline(
        query=query,
        model_name=model_name,
    )

    print(response)