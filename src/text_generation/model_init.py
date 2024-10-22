import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv
from anthropic import AnthropicVertex
import os
from openai import OpenAI
from src.text_generation.vertexai_setup import initialize_vertexai_params

load_dotenv()
OAI_API_KEY = os.getenv("OPENAI_API_KEY")
VERTEXAI_PROJECT = os.getenv("VERTEXAI_PROJECTID")


def get_chat_template():
    """
    Returns the chat template format for Hugging Face models.
    """
    return "{%- for message in messages %}{%- if message['role'] == 'user' %}{{- bos_token + '[INST] ' + message[" \
           "'content'].strip() + ' [/INST]' }}{%- elif message['role'] == 'system' %}{{- '<<SYS>>\\n' + message[" \
           "'content'].strip() + '\\n<</SYS>>\\n\\n' }}{%- elif message['role'] == 'assistant' %}{{- '[ASST] ' + " \
           "message['content'] + ' [/ASST]' + eos_token }}{%- endif %}{%- endfor %} "


def initialize_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )


class LLMBaseClass:
    """
    Base Class for text generation - user needs to provide the HF model ID while instantiating the class after which
    the generate method can be called to generate responses.
    """

    def __init__(self, model_id) -> None:
        self.model_id = model_id
        self.bnb_config = initialize_bnb_config()
        self.model = self._initialize_model()
        self.terminators = None
        self.tokenizer = None

    def _initialize_model(self):
        """
        Initialize the model based on the model_id.
        """
        model_type = self.model_id[0].lower()

        if model_type == "gpt-4o-mini":
            return OpenAI(api_key=OAI_API_KEY)

        elif model_type == "claude-3-5-sonnet@20240620":
            return AnthropicVertex(region="europe-west1", project_id=VERTEXAI_PROJECT)

        else:  # Assume Hugging Face model
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.chat_template = get_chat_template()

            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                quantization_config=self.bnb_config,
            )
            model.generation_config.pad_token_id = tokenizer.pad_token_id

            self.tokenizer = tokenizer
            self.terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

            return model

    def _generate_openai(self, messages):
        """
        Generates a response using OpenAI's GPT model.
        """
        completion = self.model.chat.completions.create(
            model=self.model_id[0],
            messages=messages,
            temperature=0.6,
            top_p=0.9,
        )
        return completion.choices[0].message.content

    def _generate_claude(self, messages):
        """
        Generates a response using Claude via VertexAI.
        """
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

    def _generate_huggingface(self, messages):
        """
        Generates a response using Hugging Face models.
        """
        input_ids = self.tokenizer.apply_chat_template(
            conversation=messages,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=1024,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True)

    def generate(self, messages):
        """
        Generates a response based on the model type.
        """
        model_type = self.model_id[0].lower()

        if model_type == "gpt-4o-mini":
            return self._generate_openai(messages)

        elif model_type == "claude-3-5-sonnet@20240620":
            return self._generate_claude(messages)

        else:  # Hugging Face models
            return self._generate_huggingface(messages)
