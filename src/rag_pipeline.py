from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from src.config import TOP_K


RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant answering questions about a user-provided document.
Use ONLY the context below to answer. If the answer isn't contained in the context,
say you don't know rather than guessing.

Context:
{context}

Question: {question}

Answer:"""
)


def format_docs(docs) -> str:
    return "\n\n".join(
        f"[Chunk {i + 1} | page {d.metadata.get('page', 'N/A')}]\n{d.page_content}"
        for i, d in enumerate(docs)
    )


def build_rag_chain(vector_db, llm, k: int = TOP_K):
    retriever = vector_db.as_retriever(
        search_kwargs={"k": k}
    )

    answer_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(x["context"])
        )
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return RunnableParallel(
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
    ).assign(answer=answer_chain)