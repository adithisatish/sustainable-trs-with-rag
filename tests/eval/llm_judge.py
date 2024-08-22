import pandas as pd
from text_generation.model_init import (GPT4)
from text_generation import text_generation as tg
from data_directories import results_dir, prompts_dir
import json, re

MODELS = {
    'GPT-4': GPT4,
}
JUDGE_PROMPT = """You will be given a user_question and system_answer couple. Your task is to provide a 'total 
rating' scoring how well the system_answer answers the user concerns expressed in the user_question. Give your answer 
as a float on a scale of 0 to 10, where 0 means that the system_answer is not helpful at all, and 10 means that the 
answer completely and helpfully addresses the question. 

Provide your feedback as follows:

Feedback:::
Total rating: (your rating, as a float between 0 and 10)

Now here are the question and answer.

Question: {question}
Answer: {answer}

Feedback:::
Total rating: """


def read_data(file_path):
    return pd.read_csv(file_path)


def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def get_prompts(prompt_ids: list):
    prompts_file_path = prompts_dir + "prompts_combined.json"
    prompts = read_json(prompts_file_path)
    print("Results file read successfully.")
    # Filter the JSON data to include only the entries with IDs in prompt_ids
    filtered_prompts = [entry for entry in prompts if entry["id"] in prompt_ids]
    # Convert the filtered data into a Pandas DataFrame
    df = pd.DataFrame(filtered_prompts)
    df.rename({"id": "prompt_id"}, inplace=True, axis=1)
    return df[["prompt_id", "prompt"]]


def prepare_data():
    results_file_path = results_dir + "results-combined_prompts/recommended_cities.csv"
    results = read_data(results_file_path)
    prompt_ids = list(results.prompt_id.unique())
    prompt_ids = [prompt_id.replace("prompt_", "") for prompt_id in prompt_ids]

    results['new_prompt_id'] = results['prompt_id'].apply(lambda x: x.replace("prompt_", ""))
    results['prompt_id'] = results['new_prompt_id']
    results.drop('new_prompt_id', axis=1, inplace=True)
    prompts = get_prompts(prompt_ids)
    merged_df = pd.merge(prompts, results, on="prompt_id")
    merged_df = merged_df[["prompt_id", "model", "prompt", "response", "response_sustainable"]]
    print("Data merged successfully.")
    return merged_df


def generate(data: pd.DataFrame, model, col_to_judge: str) -> pd.DataFrame:
    data[f"llm_judge_{col_to_judge}"] = data.apply(
        lambda x: tg.generate_response(
            model=model,
            prompt=[{"role": "system", "content": JUDGE_PROMPT.format(question=x["prompt"], answer=x[col_to_judge])}],
        ),
        axis=1,
    )
    print("Judged successfully.")
    return data


def judge(model_name):
    model = MODELS[model_name]
    data = prepare_data()
    print("Data prepared successfully.")
    data = generate(data, model, "response")
    print("Data generated successfully for vanilla.")
    data = generate(data, model, "response_sustainable")
    print("Data generated successfully for sustainability.")
    data.to_csv(results_dir + f"results-combined_prompts/judged_cities_{model_name}.csv", index=False)


def extract_judge_score(answer: str, split_str: str = "Total rating:") -> int:
    try:
        if split_str in answer:
            rating = answer.split(split_str)[1]
        else:
            rating = answer
        digit_groups = [el.strip() for el in re.findall(r"\d+(?:\.\d+)?", rating)]
        return float(digit_groups[0])
    except Exception as e:
        print(e)
        return None


def main():
    # judge('GPT-4')
    data = read_data(results_dir + "results-combined_prompts/judged_cities_GPT-4.csv")
    data['total_rating'] = data['llm_judge_response'].apply(lambda x: extract_judge_score(x))
    data['total_rating_sustainable'] = data['llm_judge_response_sustainable'].apply(lambda x: extract_judge_score(x))
    data.to_csv(results_dir + "results-combined_prompts/judged_cities_GPT-4.csv", index=False)


if __name__ == '__main__':
    #
    main()
