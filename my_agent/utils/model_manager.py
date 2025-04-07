# my_agent/utils/model_manager.py
from functools import lru_cache
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

@lru_cache(maxsize=4)
def get_model(model_name: str = "openai") -> ChatOpenAI:
    """Initializes and caches the model."""
    api_key = os.getenv("OPENAI_API_KEY")  # Get API key from environment
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Please set it in your environment.")
    if model_name == "openai":
        return ChatOpenAI(temperature=0, model_name="gpt-4o-mini", openai_api_key=api_key)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
