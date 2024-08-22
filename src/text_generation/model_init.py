import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv
from anthropic import AnthropicVertex
import os
from openai import OpenAI
from src.text_generation.vertexai_setup import initialize_vertexai_params

load_dotenv()
if "OPENAI_API_KEY" in os.environ:
    OAI_API_KEY = os.environ["OPENAI_API_KEY"]
if "VERTEXAI_PROJECTID" in os.environ:
    VERTEXAI_PROJECT = os.environ["VERTEXAI_PROJECTID"]


class LLMBaseClass:
    """
    Base Class for text generation - user needs to provide the HF model ID while instantiating the class after which
    the generate method can be called to generate responses
    
    """

    def __init__(self, model_id) -> None:
        # Initialize quantization to use less GPU

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        match (model_id[0].lower()):
            case "gpt-4o-mini":  # for open AI models
                self.api_key = OAI_API_KEY
                self.model = OpenAI(api_key=self.api_key)
            case "claude-3-5-sonnet@20240620":  # for Claude through vertexAI
                self.api_key = None
                self.model = AnthropicVertex(region="europe-west1", project_id=VERTEXAI_PROJECT)
            case _:  # for HF models
                self.api_key = None
                self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                self.tokenizer.pad_token = self.tokenizer.eos_token

                self.tokenizer.chat_template = "{%- for message in messages %}{%- if message['role'] == 'user' %}{{- " \
                                               "bos_token + '[INST] ' + message['content'].strip() + ' [/INST]' }}{%- " \
                                               "elif " \
                                               "message['role'] == 'system' %}{{- '<<SYS>>\\n' + message[" \
                                               "'content'].strip() + " \
                                               "'\\n<</SYS>>\\n\\n' }}{%- elif message['role'] == 'assistant' %}{{- '[" \
                                               "ASST] ' " \
                                               "+ message['content'] + ' [/ASST]' + eos_token }}{%- endif %}{%- " \
                                               "endfor %} "
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    quantization_config=bnb_config,
                )

                self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

                self.terminators = [
                    self.tokenizer.eos_token_id,
                    self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]

    def generate(self, messages):
        match (self.model_id[0].lower()):
            case "gpt-4o-mini":
                completion = self.model.chat.completions.create(
                    model=self.model_id[0],
                    messages=messages,
                    temperature=0.6,
                    top_p=0.9,
                )
                # Return the generated content from the API response
                return completion.choices[0].message.content
            case "claude-3-5-sonnet@20240620":
                initialize_vertexai_params()
                message = self.model.messages.create(
                    max_tokens=1024,
                    model=self.model_id[0],
                    messages=[
                        {
                          "role": "user",
                          "content": messages[0]["content"],
                        }
                      ],
                )
                return message.content[0].text
            case _:
                input_ids = self.tokenizer.apply_chat_template(
                    conversation=messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    padding=True
                ).to(self.model.device)

                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=1024,
                    # eos_token_id=self.terminators,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                )
                response = outputs[0][input_ids.shape[-1]:]

                return self.tokenizer.decode(response, skip_special_tokens=True)


