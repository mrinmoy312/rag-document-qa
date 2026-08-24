import os
import sys

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.document_loader import split_documents
from src.vector_store import (
    build_vector_store,
    save_vector_store,
    load_vector_store
)
from src.rag_pipeline import build_rag_chain, format_docs


class DeterministicFakeEmbeddings(Embeddings):
    DIM = 64

    def _vec(self, text):
        vec = [0.0] * self.DIM

        for word in text.lower().split():
            vec[hash(word) % self.DIM] += 1.0

        return vec

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def test_split_documents_creates_multiple_chunks():
    doc = Document(
        page_content="Sentence one. " * 100,
        metadata={"page": 0}
    )

    chunks = split_documents(
        [doc],
        chunk_size=200,
        chunk_overlap=20
    )

    assert len(chunks) > 1


def test_vector_store_round_trip(tmp_path):
    docs = [
        Document(
            page_content="FAISS is a library for vector similarity search."
        )
    ]

    embeddings = DeterministicFakeEmbeddings()

    db = build_vector_store(
        docs,
        embeddings
    )

    save_vector_store(
        db,
        str(tmp_path / "vs")
    )

    reloaded = load_vector_store(
        embeddings,
        str(tmp_path / "vs")
    )

    results = reloaded.similarity_search(
        "What is FAISS?",
        k=1
    )

    assert "FAISS" in results[0].page_content


def test_rag_chain_returns_answer_and_context():
    docs = [
        Document(
            page_content="The sky is blue because of Rayleigh scattering."
        )
    ]

    db = build_vector_store(
        docs,
        DeterministicFakeEmbeddings()
    )

    fake_llm = FakeListChatModel(
        responses=["Because of Rayleigh scattering."]
    )

    result = build_rag_chain(
        db,
        fake_llm,
        k=1
    ).invoke("Why is the sky blue?")

    assert "answer" in result
    assert "context" in result