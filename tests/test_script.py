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

def test(prompt_file_name, results_dir, start_idx, end_idx, sustainability=0):
    """
    
    Test function - runs the pipeline twice (once without sustainability, once with sustainability enabled) for 10 prompts for each model and stores the results. 
    The default settings of limit/k = 5 and reranking = 0 for the retrieval are used. The argument "test" in the pipeline must be set to True so that the retrieved list of cities are returned along with the LLM response.

    Args: 
        - prompt_file_name: str
        - start_idx: int
        - end_idx: int

    """
    with open(f"prompts/{prompt_file_name}","r") as file:
        prompts = json.load(file)
    
    # print(prompts)
    prompts = prompts[start_idx:end_idx] # Only because of limited GPU space; 
    
    for model_name in INSTRUCTION_TUNED_MODELS: # To run the normal models, switch "INSTRUCTION_TUNED" to "MODEL_NAMES"
        if 'Llama3.1' in model_name:
                dir_name = 'llama3point1-instruct' # change to "llama3point1" for normal models
        else:
            dir_name = model_name.lower()

        results_dir = os.path.join(os.getcwd(), {results_dir}, dir_name) # change for new iteration

        for i, item in enumerate(prompts):
            
            logger.info(f"Prompt {i + start_idx}") #change this after script executes
            prompt_results_dir = os.path.join(results_dir, f"prompt_{i+1+ start_idx}") 

            try:
                if not sustainability:
                    logger.info(f"Running pipeline for {model_name} without sustainability..")
                    cities, response = pipeline(
                        query = item['prompt'],
                        model_name = model_name,
                        test = 1,
                        limit=10,
                    )
                else: 
                    logger.info(f"Running pipeline for {model_name} with sustainability..")
                    cities, response = pipeline(
                        query = item['prompt'],
                        model_name = model_name,
                        test = 1,
                        sustainability = 1,
                        limit=10, # RUN THIS AGAIN => param was not included in results-06.08. !!!
                    )

            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                logger.error(f"Error while running pipeline: {e}, {(exc_type, fname, exc_tb.tb_lineno)}")
            
            else:
                logger.info("Pipeline execution complete. Storing response..")
                filenames = ['response.txt','cities.json']
                
                if sustainability:
                    filenames = ['response_sustainable.txt','cities_sustainable.json']

                if not os.path.exists(prompt_results_dir):
                    # Create the folder and any necessary intermediate directories
                    os.makedirs(prompt_results_dir)

                with open(f"{prompt_results_dir}/{filenames[0]}", "w") as f:
                    f.write(response)

                with open(f"{prompt_results_dir}/{filenames[1]}", 'w') as file: 
                    json.dump(cities, file)

                logger.info(f"Stored response in {prompt_results_dir}")
                


if __name__ == "__main__":
    # NEEDS TO BE RUN 
    # test(
    #     prompt_file_name="prompts_100.json",
    #     results_dir="results/results-06.08.",
    #     start_idx=0,
    #     end_idx=100,
    #     sustainability=1
    # )

    test(
        prompt_file_name="prompts_100.json",
        results_dir="results/results-06.08.",
        start_idx=70,
        end_idx=100,
    )