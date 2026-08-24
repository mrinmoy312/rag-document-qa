from src.config import (
    LLM_PROVIDER,
    GROQ_API_KEY,
    GROQ_CHAT_MODEL,
    HF_EMBEDDING_MODEL,
)


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=HF_EMBEDDING_MODEL
    )


def get_llm():
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq

        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to your .env file."
            )

        return ChatGroq(
            model=GROQ_CHAT_MODEL,
            groq_api_key=GROQ_API_KEY,
            temperature=0,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER: {LLM_PROVIDER!r}. "
        "Use 'groq'."
    )