"""
This module is responsible for loading the LLM
and returning a text-generation pipeline that other parts of the app can reuse.

We keep this isolated so:
- We only load the model once.
- Other modules don't need to worry about tokenizer / config.

=> Uncomment the section you want to use depending of your configuration
"""

import transformers
import torch
from config import MODEL_ID, HF_TOKEN, MAX_NEW_TOKENS


def load_llm(logger):

    print("DEBUG HF_TOKEN present:", HF_TOKEN is not None)
    logger.info("Dont forget to install database following instructions in src/db/connection.py !!")
    logger.info("Dont forget to create .env file with your HF_TOKEN !!")

    """ ################# Cloud config: #################
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

    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,   # allows custom code from model repo (needed for some instruct models)
        config=model_config,
        device_map="auto",        # automatically put layers on GPU if available
        quantization_config=bnb_config,
        token=HF_TOKEN
    )
    ###################################################
    """ 

    ################# MAC LOCAL CONFIG: #################
    if torch.backends.mps.is_available():
        # On Apple Silicon, MPS = Metal Performance Shaders (Apple GPU backend)
        Device_map = "mps"
        dtype = torch.bfloat16  # bfloat16 works better than float16 on MPS
    else:
        Device_map = "cpu"
        dtype = torch.float32   # safe default on CPU

    # Load model configuration
    model_config = transformers.AutoConfig.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN
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
        device_map=Device_map,     # "mps" on Apple GPU, otherwise "cpu"
        dtype=dtype,   # lowers memory usage a bit on MPS
        token=HF_TOKEN
    )

    # Apple-friendly settings
    model.generation_config.use_cache = False
    try:
        model.config.attn_implementation = "eager"
    except Exception:
        pass
    model.eval()
    ###################################################

    # Build the generation pipeline
    pipe = transformers.pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        return_full_text=False,
        temperature=0.2,    # low temperature = more deterministic
        do_sample=False     # no sampling = stable, predictable output
    )
    pipe.SMALL_KW = dict(max_new_tokens=8,  do_sample=False, use_cache=False)
    pipe.MED_KW   = dict(max_new_tokens=32, do_sample=False, use_cache=False)
    pipe.LONG_KW  = dict(max_new_tokens=128, do_sample=False, use_cache=False)

    return pipe