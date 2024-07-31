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

def test():
    """
    
    Test function - runs the pipeline twice (once without sustainability, once with sustainability enabled) for 10 prompts for each model and stores the results. 
    The default settings of limit/k = 5 and reranking = 0 for the retrieval are used. The argument "test" in the pipeline must be set to True so that the retrieved list of cities are returned along with the LLM response.

    """
    with open("prompts.json","r") as file:
        prompts = json.load(file)
    
    # print(prompts)
    
    for model_name in MODEL_NAMES: 
        if model_name == 'Llama3.1':
                dir_name = 'llama3point1'
        else:
            dir_name = model_name.lower()

        results_dir = os.path.join(os.getcwd(), dir_name)

        for i, item in enumerate(prompts):
            logger.info(f"Prompt {i+1}: {item['prompt']}")
            try:
                logger.info(f"Running pipeline for {model_name} without sustainability..")
                cities, response = pipeline(
                    query = item['prompt'],
                    model_name = model_name,
                    test = 1,
                )
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                logger.error(f"Error while running pipeline: {e}, {(exc_type, fname, exc_tb.tb_lineno)}")
            
            else:
                logger.info("Pipeline execution complete. Storing response..")

                prompt_results_dir = os.path.join(results_dir, f"prompt_{i+1}")

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
                response = pipeline(
                    query = item['prompt'],
                    model_name = model_name,
                    test = 1,
                    sustainability = 1
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


if __name__ == "__main__":
    test()