import os

from augmentation import prompt_generation as pg
from information_retrieval import info_retrieval as ir
from transformers import AutoTokenizer

from src.text_generation.models import (
    Llama3,
    Mistral,
    Gemma2,
    Llama3Point1,
    GPT4,
    Claude3Point5Sonnet,
)
import logging
import ollama
import torch
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
logger = logging.getLogger(__name__)
logging.basicConfig(encoding='utf-8', level=logging.DEBUG)


def check_backend():
    if (torch.backends.mps.is_available()) and (torch.backends.mps.is_built()):
        torch.set_default_device("mps")
        logger.info("setting default to mps")
    else:
        torch.set_default_device("cuda")
        logger.info("setting default to cuda")


def generate_response(model, prompt, use_ollama=False):
    """
    
    Function that initializes the LLM class and calls the generate function.

    Args:
        - messages: list; contains the system and user prompt 
        - model: class; the class of the llm to be initialized
    
    """

    logger.info(f"Initializing LLM configuration for {model}")

    # tokens = llm.tokenizer.tokenize(prompt)
    # print("Number of tokens in your text: ", len(tokens))
    if use_ollama:
        check_backend()
        model_name = str(model).replace("'>", "").split(".")[-1]
        print(model_name)
        try:
            response = ollama.chat(model=model_name.lower(),
                                   messages=prompt,
                                   options={"num_ctx": 1024}
                                   )
            print("Response,", response)
            return response['message']['content']
        except Exception as e:
            logger.error(f"Error while generating response for {model_name.lower()}: {e}")
            return 'ERROR'
    else:
        logger.info("Generating response")
        llm = model()
        try:
            response = llm.generate(prompt)
        except Exception as e:
            logger.error(f"Error while generating response for {model}: {e}")
            response = 'ERROR'

        return response


def test(model):
    context_params = {
        'limit': 3,
        'reranking': 0
    }
    # model = Llama3Point1

    query = "Suggest some places to visit during winter. I like hiking, nature and the mountains and I enjoy skiing " \
            "in winter. "

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

    # return without_sfairness

    logger.info(f"Augmented prompt, initializing {model} and generating response..")
    try:
        model_response = generate_response(model, without_sfairness)
    except Exception as e:
        logger.info(f"Error while generating response: {e}")
        return None

    return model_response


if __name__ == "__main__":
    resp = test(Claude3Point5Sonnet)
    print(resp)
