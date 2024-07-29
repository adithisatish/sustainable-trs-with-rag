from augmentation import prompt_generation as pg
from information_retrieval import info_retrieval as ir
from text_generation.model_init import (
    Llama3,
    Mistral,
    Gemma2,
    Llama3Point1,
)
import os 
import sys 
import ollama
import logging 

logger = logging.getLogger(__name__)
logging.basicConfig(encoding='utf-8', level=logging.DEBUG)

def generate_response(model, prompt):
    """
    
    Function that initializes the LLM class and calls the generate function.

    Args:
        - messages: list; contains the system and user prompt 
        - model: class; the class of the llm to be initialized
    
    """

    logger.info(f"Initializing LLM configuration for {model}")
    llm = model()

    logger.info("Generating response")
    try: 
        response = llm.generate(prompt)
    except Exception as e:
        logger.error(f"Error while generating response for {model}: {e}")
        response = 'ERROR'

    return response

def test(model):
    context_params = {
        'limit': 10,
        'reranking': 1
    }
    # model = Llama3Point1

    query = "Suggest some places to visit during winter. I like hiking, nature and the mountains and I enjoy skiing in winter."

    # without sustainability
    logger.info("Retrieving context..")
    try:
        context = ir.get_context(query=query, **context_params)
    except Exception as e:
        logger.error(f"Error while trying to get context: {e}")
        return None
    
    logger.info("Retrieved context, augmenting prompt (without sustainability)..")
    try:
        without_sfairness = pg.augment_prompt(
            query=query, 
            context=context,
            sustainability=0,
            params=context_params
        )
    except Exception as e:
        logger.error(f"Error while trying to augment prompt: {e}")
        return None
    
    return without_sfairness

    logger.info(f"Augmented prompt, initializing {model} and generating response..")
    try:
        response = generate_response(model, without_sfairness)
    except Exception as e: 
        logger.info(f"Error while generating response: {e}")
        return None

    return response


if __name__ == "__main__":
    response = test(Mistral)
    print(response)

    
