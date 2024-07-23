import sys 
import os 
import re
from information_retrieval import info_retrieval as ir

def generate_prompt(query, context, template = None):
    """
    Function that generates the prompt given the user query and retrieved context. A specific prompt template will be used if provided, otherwise the default base_prompt template is used.

    Args: 
        - query: str
        - context: list[dict]
        - template: str
    
    """

    if template: 
        SYS_PROMPT = template
    else:
        SYS_PROMPT = """You are an AI recommendation system. Your task is to recommend cities in Europe for travel based on the user's question. You should use the provided contexts to answer the question. 
        Your answers are correct, high-quality, and written by an domain expert. If the provided context does not contain the answer, simply state, "The provided context does not have the answer."
        """

    USER_PROMPT = """ Question: {}

    Context: {}

    """

    formatted_prompt = f"{USER_PROMPT.format(query, context)}"
    messages = [{"role":"system","content":SYS_PROMPT},{"role":"user","content":formatted_prompt}]

    return messages 

def augment_prompt(query, sustainability = 0, **params):
    """
    Function that accepts the user query as input, obtains relevant documents and augments the prompt with the retrieved context, which can be passed to the LLM. 

    Args: 
        - query: str
        - sustainability: bool; if true, then the prompt is appended to instruct the LLM to use s-fairness scores while generating the answer
        - params: key-value parameters to be passed to the get_context function; sets the limit of results and whether to rerank the results
    
    """

    prompt_with_sustainability = """You are an AI recommendation system. Your task is to recommend cities in Europe for travel based on the user's question. You should use the provided contexts to answer the question. You recommend the most sustainable city to the user. 
        The context contains a sustainability score for each city, also known as the s-fairness score, along with the ideal month of travel. A lower s-fairness scores indicates that the city is a more sustainable travel destination for the month provided. 
        Your answers are correct, high-quality, and written by an domain expert. If the provided context does not contain the answer, simply state, "The provided context does not have the answer."
        """
    
    context = ir.get_context(query=query, params={'sustainability': sustainability})

    # TO-DO: Some post processing for context???

    if sustainability: 
        prompt = generate_prompt(query, context, prompt_with_sustainability)
    else: 
        prompt = generate_prompt(query, context)

    return prompt


def test(): 
    context_params = {
        'limit': 5,
        'reranking': 0
    }

    query = "Suggest some places to visit during winter. I like hiking, nature and the mountains and I enjoy skiing in winter."

    without_sfairness = augment_prompt(
        query=query, 
        sustainability=0,
        params=context_params
    )

    with_sfairness = augment_prompt(
        query=query, 
        sustainability=1,
        params=context_params
    )

