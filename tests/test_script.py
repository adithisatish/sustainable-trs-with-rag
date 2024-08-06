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

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '../src')
sys.path.insert(0, src_dir)

from pipeline import pipeline
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(encoding='utf-8', level=logging.DEBUG)

MODEL_NAMES = ['Llama3', 'Mistral', 'Gemma2', 'Llama3.1']
INSTRUCTION_TUNED_MODELS = ['Llama3-Instruct', 'Mistral-Instruct', 'Llama3.1-Instruct', ]


def test():
    """
    
    Test function - runs the pipeline twice (once without sustainability, once with sustainability enabled) for 10
    prompts for each model and stores the results. The default settings of limit/k = 5 and reranking = 0 for the
    retrieval are used. The argument "test" in the pipeline must be set to True so that the retrieved list of cities
    are returned along with the LLM response.

    """
    with open("prompts/prompts_100.json", "r") as file:
        prompts = json.load(file)

    prompts = prompts[:50] # Only because of limited GPU space; run the next 50 in the next iteration
    
    for model_name in INSTRUCTION_TUNED_MODELS: # To run the normal models, switch "INSTRUCTION_TUNED" to "MODEL_NAMES"
        if 'Llama3.1' in model_name:
            dir_name = 'llama3point1-instruct'  # change to "llama3point1" for normal models
        else:
            dir_name = model_name.lower()

        results_dir = os.path.join(os.getcwd(), "results-06.08.", dir_name)

        for i, item in enumerate(prompts):
            logger.info(f"Prompt {i}")
            try:
                logger.info(f"Running pipeline for {model_name} without sustainability..")
                cities, response = pipeline(
                    query=item['prompt'],
                    model_name=model_name,
                    test=1,
                    limit=10,
                )
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                logger.error(f"Error while running pipeline: {e}, {(exc_type, fname, exc_tb.tb_lineno)}")

            else:
                logger.info("Pipeline execution complete. Storing response..")

                prompt_results_dir = os.path.join(results_dir, f"prompt_{i + 1}")

                if not os.path.exists(prompt_results_dir):
                    # Create the folder and any necessary intermediate directories
                    os.makedirs(prompt_results_dir)

                with open(f"{prompt_results_dir}/response.txt", "w") as f:
                    f.write(response)

                with open(f"{prompt_results_dir}/cities.json", 'w') as file:
                    json.dump(cities, file)

            # with sustainability
            try:
                logger.info(f"Running pipeline for {model_name} with sustainability..")
                cities, response = pipeline(
                    query=item['prompt'],
                    model_name=model_name,
                    test=1,
                    sustainability=1
                )
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                logger.error(f"Error while running pipeline: {e}, {(exc_type, fname, exc_tb.tb_lineno)}")

            else:
                logger.info("Pipeline execution complete. Storing response..")

                if not os.path.exists(prompt_results_dir):
                    # Create the folder and any necessary intermediate directories
                    os.makedirs(prompt_results_dir)

                with open(f"{prompt_results_dir}/response_sustainable.txt", "w") as f:
                    f.write(response)

                with open(f"{prompt_results_dir}/cities_sustainable.json", 'w') as file:
                    json.dump(cities, file)

                logger.info(f"Stored response in {prompt_results_dir}")


if __name__ == "__main__":
    test()
