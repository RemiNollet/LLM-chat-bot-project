"""
This module is responsible for loading the LLM (Mistral 7B Instruct)
and returning a text-generation pipeline that other parts of the app can reuse.

We keep this isolated so:
- We only load the model once.
- Other modules don't need to worry about tokenizer / config.
"""

import transformers
import torch
from config import MODEL_ID, HF_TOKEN, MAX_NEW_TOKENS


def load_llm():
    """
    Load the Mistral model in 8-bit precision. + tokenizer as a text-generation pipeline.
    Returns a Hugging Face transformers pipeline object.

    The pipeline will be reused across classifier / parameter extractor / answer generator.
    """

    if torch.backends.mps.is_available():
        # On Apple Silicon, MPS = Metal Performance Shaders (Apple GPU backend)
        Device_map = "mps"
        Torch_dtype = torch.bfloat16  # bfloat16 works better than float16 on MPS
    else:
        Device_map = "cpu"
        Torch_dtype = torch.float32   # safe default on CPU

    # Load model configuration
    model_config = transformers.AutoConfig.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN
    )

    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        use_fast=False
    )

    # Load model weights
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,   # allows custom code from model repo (needed for some instruct models)
        config=model_config,
        #device_map="auto",        # automatically put layers on GPU if available
        #load_in_8bit=True, 
        device_map=Device_map,     # "mps" on Apple GPU, otherwise "cpu"
        torch_dtype=Torch_dtype,   # lowers memory usage a bit on MPS
        #quantization_config=bnb_config,
        token=HF_TOKEN
    )


    # Build the generation pipeline
    pipe = transformers.pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.2,    # low temperature = more deterministic
        do_sample=False     # no sampling = stable, predictable output
    )

    return pipe