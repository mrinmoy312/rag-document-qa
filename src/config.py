
import os
import streamlit as st

from dotenv import load_dotenv

# Load environment variables from local .env file
load_dotenv()


def get_config_value(name, default=None):
    """
    Read configuration from Streamlit Cloud Secrets first.
    If unavailable, fall back to local environment variables.
    """

    try:
        value = st.secrets.get(name, None)

        if value is not None:
            return value

    except Exception:
        pass

    return os.getenv(name, default)


# LLM Configuration
LLM_PROVIDER = get_config_value(
    "LLM_PROVIDER",
    "groq"
).lower()

GROQ_API_KEY = get_config_value(
    "GROQ_API_KEY"
)

GROQ_CHAT_MODEL = get_config_value(
    "GROQ_CHAT_MODEL",
    "openai/gpt-oss-120b"
)


# Hugging Face Embedding Configuration
HF_EMBEDDING_MODEL = get_config_value(
    "HF_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)


# Document Processing Configuration
CHUNK_SIZE = int(
    get_config_value("CHUNK_SIZE", "1000")
)

CHUNK_OVERLAP = int(
    get_config_value("CHUNK_OVERLAP", "200")
)

TOP_K = int(
    get_config_value("TOP_K", "4")
)


# Vector Store Configuration
VECTOR_STORE_DIR = get_config_value(
    "VECTOR_STORE_DIR",
    "vector_store"
)
