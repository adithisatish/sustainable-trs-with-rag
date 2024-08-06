from information_retrieval import info_retrieval as ir
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(encoding='utf-8', level=logging.DEBUG)


def generate_prompt(query, context, template=None):
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
        SYS_PROMPT = """You are an AI recommendation system. Your task is to recommend cities in Europe for travel based on the user's question. You should use the provided contexts to suggest the city that is best suited to the user's question, as well as the month of travel. If the user has already provided the month of travel in the question, use the same month; otherwise, provide the ideal month of travel. Your answer must begin with "I recommend " followed by the city name and why you recommended it. Your answers are correct, high-quality, and written by a domain expert. If the provided context does not contain the answer, simply state, "The provided context does not have the answer." """

    USER_PROMPT = """ Question: {} Which city do you recommend and why?

    Context: Here are the options: {} 
    
    Answer:

    """

    formatted_prompt = f"{USER_PROMPT.format(query, context)}"
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": formatted_prompt}
    ]

    return messages


def format_context(context):
    """
    Function that formats the retrieved context in a way that is easy for the LLM to understand. 

    Args:
        - context: list[dict]; retrieved context 
    
    """

    formatted_context = ''

    for i, (city, info) in enumerate(context.items()):

        text = f"Option {i + 1}: {city} is a city in {info['country']}."
        info_text = f"Here is some information about the city. {info['text']}"

        attractions_text = "Here are some attractions: "
        att_flag = 0
        restaurants_text = "Here are some places to eat/drink: "
        rest_flag = 0

        hotels_text = "Here are some hotels: "
        hotel_flag = 0

        if len(info['listings']):
            for listing in info['listings']:
                if listing['type'] in ['see', 'do', 'go', 'view']:
                    att_flag = 1
                    attractions_text += f"{listing['name']} ({listing['description']}), "
                elif listing['type'] in ['eat', 'drink']:
                    rest_flag = 1
                    restaurants_text += f"{listing['name']} ({listing['description']}), "
                else:
                    hotel_flag = 1
                    hotels_text += f"{listing['name']} ({listing['description']}), "

        # If we add sustainability in the end then it could get truncated because of context window
        if "sustainability" in info:
            if info['sustainability']['month'] == 'No data available':
                sfairness_text = "This city has no sustainability (or s-fairness) score available."

            else:
                sfairness_text = f"The sustainability (or s-fairness) score for {city} in {info['sustainability']['month']} is {info['sustainability']['s-fairness']}. \n "

            text += sfairness_text

        text += info_text

        if att_flag:
            text += f"\n{attractions_text}"

        if rest_flag:
            text += f"\n{restaurants_text}"

        if hotel_flag:
            text += f"\n{hotels_text}"

        formatted_context += text + "\n\n "

    return formatted_context


def augment_prompt(query, context, sustainability=0, **params):
    """
    Function that accepts the user query as input, obtains relevant documents and augments the prompt with the retrieved context, which can be passed to the LLM. 

    Args: 
        - query: str
        - context: retrieved context, must be formatted otherwise the LLM cannot understand the nested dictionaries!
        - sustainability: bool; if true, then the prompt is appended to instruct the LLM to use s-fairness scores while generating the answer
        - params: key-value parameters to be passed to the get_context function; sets the limit of results and whether to rerank the results
    
    """

    # what about the cities without s-fairness scores? i.e. they don't have seasonality data 

    prompt_with_sustainability = """You are an AI recommendation system. Your task is to recommend cities in Europe for travel based on the user's question. You should use the provided contexts to suggest the city that is best suited to the user's question. You recommend the most sustainable city to the user, as well as the best month of travel. The context contains a sustainability score for each city, also known as the s-fairness score, along with the ideal month of travel. A lower s-fairness score indicates that the city is a more sustainable travel destination for the month provided. A city without a sustainability score should not be considered. You should only consider the s-fairness score while choosing the best city. However, your answer should not contain the numeric score itself. Your answer must begin with "I recommend " followed by the city name and why you recommended it. Your answers are correct, high-quality, and written by a domain expert. If the provided context does not contain the answer, simply state, "The provided context does not have the answer." """

    # format context
    formatted_context = format_context(context)

    if sustainability:
        prompt = generate_prompt(query, formatted_context, prompt_with_sustainability)
    else:
        prompt = generate_prompt(query, formatted_context)

    return prompt


def test():
    context_params = {
        'limit': 3,
        'reranking': 0
    }

    query = "Suggest some places to visit during winter. I like hiking, nature and the mountains and I enjoy skiing in winter."

    # without sustainability
    context = ir.get_context(query, **context_params)
    # formatted_context = format_context(context)

    without_sfairness = augment_prompt(
        query=query,
        context=context,
        sustainability=0,
        params=context_params
    )

    # with sustainability
    context_params.update({'sustainability': 1})
    s_context = ir.get_context(query, **context_params)
    # s_formatted_context = format_context(s_context)

    with_sfairness = augment_prompt(
        query=query,
        context=s_context,
        sustainability=1,
        params=context_params
    )

    return with_sfairness


if __name__ == "__main__":
    prompt = test()
    print(prompt)
