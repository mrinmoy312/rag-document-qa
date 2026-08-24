import os
import warnings

warnings.filterwarnings(
    "ignore",
    message="`langchain-community` is being sunset"
)

from langchain_community.vectorstores import FAISS


def build_vector_store(chunks, embeddings):
    return FAISS.from_documents(chunks, embeddings)


def save_vector_store(db, path: str) -> None:
    db.save_local(path)


def load_vector_store(embeddings, path: str):
    if not os.path.isdir(path):
        return None

    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )