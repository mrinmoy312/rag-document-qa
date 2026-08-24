import warnings

warnings.filterwarnings(
    "ignore",
    message="`langchain-community` is being sunset"
)

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def load_document(file_path: str):
    lower = file_path.lower()

    if lower.endswith(".pdf"):
        loader = PyPDFLoader(file_path)

    elif lower.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")

    else:
        raise ValueError(
            f"Unsupported file type: {file_path}. Use .pdf or .txt"
        )

    return loader.load()


def split_documents(
    documents,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return splitter.split_documents(documents)