import sys 
import os 
import re
from information_retrieval import info_retrieval

def generate_prompt(query, context, template = None):
    """
    Function that generates the prompt given the user query and retrieved context. A specific prompt template will be used if provided, otherwise the default base_prompt template is used.

    Args: 
        - query: str
        - context: list[dict]
        - template: str
    
    """

    if template: 
        base_prompt = template
    else:
        base_prompt = """You are an AI recommendation system. Your task is to recommend cities in Europe for travel based on the user's question. You should use the provided contexts to answer the question. 
        Your answers are correct, high-quality, and written by an domain expert. If the provided context does not contain the answer, simply state, "The provided context does not have the answer."

        User question: {}

        Contexts:
        {}
        """

    prompt = f"{base_prompt.format(query, context)}"

    return prompt 

def augment_prompt(query, sustainability = 0):
    """
    Function that accepts the user query as input, obtains relevant documents and augments the prompt with the retrieved context, which can be passed to the LLM. 

    Args: 
        - query: str
        - sustainability: bool; if true, then the prompt is appended to instruct the LLM to use s-fairness scores while generating the answer
    
    """

    prompt_with_sustainability = """You are an AI recommendation system. Your task is to recommend cities in Europe for travel based on the user's question. You should use the provided contexts to answer the question. You recommend the most sustainable city to the user. 
        The context contains a sustainability score for each city, also known as the s-fairness score, along with the ideal month of travel. A lower s-fairness scores indicates that the city is a more sustainable travel destination for the month provided. 
        Your answers are correct, high-quality, and written by an domain expert. If the provided context does not contain the answer, simply state, "The provided context does not have the answer."

        User question: {}

        Contexts:
        {}
        """
    
    context = info_retrieval.get_context(query)

    # TO-DO: Some post processing for context???

    if sustainability: 
        prompt = generate_prompt(query, context, prompt_with_sustainability)
    else: 
        prompt = generate_prompt(query, context)

    return prompt