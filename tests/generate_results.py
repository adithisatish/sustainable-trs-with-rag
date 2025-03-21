"""
Test script for the pipeline
The test queries can be found under "prompts.json. These were generated by ChatGPT using the following prompt: 

"I want to test a travel recommendation system. The system requests prompts from the user about what their travel interests are, and when they plan to travel (this can either be a particular season or month(s) and is optional). The system then recommends cities in Europe based on these user prompts. Your job is to generate <n> different prompts in order to test this recommendation system. Here's an example of what a prompt looks like: 

"Suggest some places to visit during winter. I like hiking, nature and the mountains and I enjoy skiing in winter."

What are your <n> test prompts? Generate them in JSON."

Tested for n = 10 and n = 100
"""

import sys
import os
import json
import re

sys.path.append("../")
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '../src')
sys.path.insert(0, src_dir)

print("Current dir", os.getcwd())

from src.data_directories import *
from src.pipeline import pipeline
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(encoding='utf-8', level=logging.INFO)

MODEL_NAMES = ['Llama3', 'Mistral', 'Gemma2', 'Llama3.1']
INSTRUCTION_TUNED_MODELS = ['Phi3-Instruct', 'Mistral-Instruct', 'Llama3.1-Instruct', 'Llama3-Instruct']


def generate_results(models, prompt_file_name, results_dir, start_idx, end_idx, sustainability=0):
    """
    
    Test function - runs the pipeline for the prompt file and models provided and stores the results. The default settings of limit/k = 10 and reranking = 0 for the
    retrieval are used. The argument "test" in the pipeline must be set to True so that the retrieved list of cities
    are returned along with the LLM response.

    Args: 
        - models: list; of model names
        - prompt_file_name: str
        - results_dir: str
        - start_idx: int
        - end_idx: int

    """
    with open(prompt_file_name, "r") as file:
        prompts = json.load(file)

    # print(prompts)
    prompts = prompts[start_idx:end_idx]  # Only because of limited GPU space;
    print(prompts)
    for model_name in models:
        if 'Llama3.1' in model_name:
            dir_name = 'llama3point1-instruct'  # change to "llama3point1" for normal models
        else:
            dir_name = model_name.lower()

        model_results_dir = os.path.join(results_dir, dir_name)

        for item in prompts:

            logger.info(f"Prompt: {item['id']}")
            prompt_results_dir = os.path.join(model_results_dir, f"prompt_{item['id']}")

            try:
                if not sustainability:
                    logger.info(f"Running pipeline for {model_name} without sustainability..")
                    cities, context, response = pipeline(
                        query=item['prompt'],
                        model_name=model_name,
                        test=1,
                        sustainability=0,
                        limit=10,
                    )
                else:
                    logger.info(f"Running pipeline for {model_name} with sustainability..")
                    cities, context, response = pipeline(
                        query=item['prompt'],
                        model_name=model_name,
                        test=1,
                        sustainability=1,
                        limit=10,
                    )
                    print("Response in generate_results: ", len(response))

            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                logger.error(f"Error while running pipeline: {e}, {(exc_type, fname, exc_tb.tb_lineno)}")

            else:
                logger.info("Pipeline execution complete. Storing response..")
                filenames = ['response.txt', 'context.txt', 'cities.json']

                if sustainability:
                    filenames = ['response_sustainable.txt', 'context_sustainable.txt', 'cities_sustainable.json']

                if not os.path.exists(prompt_results_dir):
                    # Create the folder and any necessary intermediate directories
                    os.makedirs(prompt_results_dir)

                with open(f"{prompt_results_dir}/{filenames[0]}", "w") as f:
                    f.write(response)

                with open(f"{prompt_results_dir}/{filenames[1]}", "w") as f:
                    f.write(context)

                with open(f"{prompt_results_dir}/{filenames[2]}", 'w') as file:
                    json.dump(cities, file)

                logger.info(f"Stored response in {prompt_results_dir}")


def main(is_sustainable=True):
    prompt_file = os.path.join(prompts_dir, "prompts_combined.json")

    if is_sustainable:
        results_path = os.path.join(results_dir, 'results-combined_prompts_SAR')
    else:
        results_path = os.path.join(results_dir, 'results-combined_prompts')

    # define the indices
    start = 0
    end = 200
    # generating results for the models
    generate_results(
        # models=['Mistral-Instruct', 'Llama3.1-Instruct'],
        models=['Gemma2'],
        prompt_file_name=prompt_file,
        results_dir=results_path,
        start_idx=start,
        end_idx=end,
        sustainability=is_sustainable
    )


if __name__ == "__main__":
    main(is_sustainable=False)
