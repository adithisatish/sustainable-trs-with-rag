import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv

load_dotenv()
import os
from openai import OpenAI

OAI_API_KEY = os.environ["OPENAI_API_KEY"]


class LLMBaseClass():
    """
    Base Class for text generation - user needs to provide the HF model ID while instantiating the class after which
    the generate method can be called to generate responses
    
    """

    def __init__(self, model_id) -> None:
        # Initialize quantization to use less GPU
        self.api_key = OAI_API_KEY
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        if "gpt" in model_id[0].lower():
            self.model = OpenAI(api_key=self.api_key)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            self.tokenizer.chat_template = "{%- for message in messages %}{%- if message['role'] == 'user' %}{{- " \
                                           "bos_token + '[INST] ' + message['content'].strip() + ' [/INST]' }}{%- elif " \
                                           "message['role'] == 'system' %}{{- '<<SYS>>\\n' + message[" \
                                           "'content'].strip() + " \
                                           "'\\n<</SYS>>\\n\\n' }}{%- elif message['role'] == 'assistant' %}{{- '[" \
                                           "ASST] ' " \
                                           " + message['content'] + ' [/ASST]' + eos_token }}{%- endif %}{%- endfor %} "

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
        if "gpt" in self.model_id[0].lower():
            completion = self.model.chat.completions.create(
                model=self.model_id[0],
                messages=messages,
                temperature=0.6,
                top_p=0.9,
            )
            # Return the generated content from the API response
            return completion.choices[0].message.content
        # print(messages[0])
        else:
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


class Llama3(LLMBaseClass):
    """
    Initializes a Llama3 model object
    """

    def __init__(self) -> None:
        self.model_id = "meta-llama/Meta-Llama-3-8B"

        super().__init__(self.model_id)


class Mistral(LLMBaseClass):
    """
    Initializes a Mistral model object
    """

    def __init__(self) -> None:
        self.model_id = "mistralai/Mistral-7B-v0.3"
        super().__init__(self.model_id)


class Gemma2(LLMBaseClass):
    """
    Initializes a Gemma2 model object
    """

    def __init__(self) -> None:
        self.model_id = "google/gemma-2-9b"
        super().__init__(self.model_id)


class Llama3Point1(LLMBaseClass):
    """
    Initializes a Llama 3.1 object 
    """

    def __init__(self) -> None:
        self.model_id = "meta-llama/Meta-Llama-3.1-8B"
        super().__init__(self.model_id)


class Llama3Instruct(LLMBaseClass):
    """
    Initializes a Llama 3 Instruct object
    """

    def __init__(self) -> None:
        self.model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        super().__init__(self.model_id)


class MistralInstruct(LLMBaseClass):
    """
    Initializes a Mistral Instruct object
    """

    def __init__(self) -> None:
        self.model_id = "mistralai/Mistral-7B-Instruct-v0.1"
        super().__init__(self.model_id)


class Llama3Point1Instruct(LLMBaseClass):
    """
    Initializes a Llama 3.1 Instruct object
    """

    def __init__(self) -> None:
        self.model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        super().__init__(self.model_id)


class GPT4(LLMBaseClass):
    """
    Initializes a GPT-4 Instruct object
    """

    def __init__(self) -> None:
        self.model_id = "gpt-4o-mini",
        super().__init__(self.model_id)

    # def generate(self, messages):
    #     client = OpenAI(api_key=self.api_key)
    #
    #     completion = client.chat.completions.create(
    #         model="gpt-4o",
    #         messages=messages,
    #         temperature=0.6,
    #         top_p=0.9,
    #     )
    #
    #     # Return the generated content from the API response
    #     return completion['choices'][0]['message']['content']
