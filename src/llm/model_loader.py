"""
This module is responsible for loading the LLM (Mistral 7B Instruct)
and returning a text-generation pipeline that other parts of the app can reuse.

We keep this isolated so:
- We only load the model once.
- Other modules don't need to worry about tokenizer / config.
"""

import transformers
from config import MODEL_ID, HF_TOKEN, MAX_NEW_TOKENS


def load_llm():
    """
    Load the Mistral model + tokenizer as a text-generation pipeline.
    Returns a Hugging Face transformers pipeline object.

    The pipeline will be reused across classifier / parameter extractor / answer generator.
    """

    # Load model configuration
    model_config = transformers.AutoConfig.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN
    )

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN
    )

    # Load model weights
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,   # allows custom code from model repo (needed for some instruct models)
        config=model_config,
        device_map="auto",        # automatically put layers on GPU if available
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