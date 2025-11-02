import os
from dotenv import load_dotenv
import transformers
import torch

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
# MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

print("HF_TOKEN present:", HF_TOKEN is not None)

# pick device for test (cpu for now to avoid mps weirdness while just downloading)
device_map = "cpu"

print("Loading tokenizer...")
tokenizer = transformers.AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    use_fast=False,
)
print("Tokenizer OK")

print("Loading model (this will download weights if not cached)...")
model = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    token=HF_TOKEN,
    device_map=device_map,
    torch_dtype=torch.float32,
)
print("Model OK")

print("All good ✅")