import json
import os
import sys

sys.path.append("../")
from src.data_directories import prompts_dir


def load_prompts(prompt_file_name: str):
    with open(prompt_file_name, "r") as file:
        prompts = json.load(file)
    return prompts


def modify_prompts(prompts: list, model_name: str):
    for idx, prompt in enumerate(prompts):
        prompt["id"] = str(idx) + "_" + model_name
        prompt["model"] = model_name
    return prompts


def save_file(file_name: str, data: list):
    with open(file_name, 'w') as file:
        json.dump(data, file, indent=4)


def main():
    prompt_files = os.listdir(prompts_dir)
    prompt_list = []
    file_to_save = "prompts_combined.json"
    if file_to_save in prompt_files: # remove the file to save if it already exists
        prompt_files.remove(file_to_save)
    for prompt_file in prompt_files:
        prompts = load_prompts(prompts_dir + prompt_file)
        model_name = prompt_file[:-5]
        modified_prompts = modify_prompts(prompts, model_name)
        prompt_list.extend(modified_prompts)
    save_file(prompts_dir + file_to_save, prompt_list)
    print(f"Prompts saved to {file_to_save}")


if __name__ == "__main__":
    main()
