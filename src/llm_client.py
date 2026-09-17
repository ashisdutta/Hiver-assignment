"""Shared Groq (OpenAI-compatible) client setup for LLM-backed pipeline stages."""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Copy .env.example to .env and fill in your key."
            )
        _client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    return _client
