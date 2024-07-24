import os 
import sys 
import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from augmentation import prompt_generation as pg


class LLMBaseClass():
    """
    Base Class for text generation - user needs to provide the HF model ID while instantiating the class after which the generate method can be called to generate responses 
    
    """
    def __init__(self, model_id) -> None:

        # Initialize quantization to use less GPU 
        bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=bnb_config
        )

        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    def generate(self, messages):

        # print(messages[0])
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=1024,
            eos_token_id=self.terminators,
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
        