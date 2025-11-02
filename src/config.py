import os
from dotenv import load_dotenv

# Load environment variables from .env file into the process environment
load_dotenv()

# ID of the LLM you want to use from Hugging Face
#MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

# Hugging Face token (must be in .env as HF_TOKEN=hf_xxx)
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise RuntimeError(
        "Missing HF_TOKEN environment variable. "
        "Create a .env file with HF_TOKEN=your_token and DO NOT commit it."
    )

# Database path (SQLite file)
DB_PATH = "data/orders.db"

# Optional: default max new tokens for generation
MAX_NEW_TOKENS = 96