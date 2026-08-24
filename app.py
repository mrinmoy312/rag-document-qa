import os
import tempfile

import streamlit as st

from src.document_loader import load_document, split_documents
from src.llm_provider import get_embeddings, get_llm
from src.vector_store import build_vector_store, save_vector_store
from src.rag_pipeline import build_rag_chain
from src.config import VECTOR_STORE_DIR, LLM_PROVIDER


st.set_page_config(
    page_title="RAG-Powered Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: Inter, -apple-system, BlinkMacSystemFont,
                 "Segoe UI", sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 15% 10%,
            rgba(59, 130, 246, 0.08),
            transparent 28%
        ),
        radial-gradient(
            circle at 85% 20%,
            rgba(139, 92, 246, 0.07),
            transparent 30%
        ),
        #0b0f19;
    color: #e5e7eb;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

#MainMenu,
footer,
header {
    visibility: hidden;
}

.hero {
    background:
        linear-gradient(
            135deg,
            rgba(37, 99, 235, 0.20),
            rgba(124, 58, 237, 0.18)
        );
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 20px;
    padding: 2.2rem 2.4rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.25);
}

.hero h1 {
    color: #f8fafc;
    font-size: 2.5rem;
    font-weight: 750;
    margin-bottom: 0.45rem;
}

.hero p {
    color: #94a3b8;
    font-size: 1.05rem;
    margin-bottom: 0;
}

.info-card {
    background: rgba(17, 24, 39, 0.75);
    border: 1px solid rgba(148, 163, 184, 0.13);
    border-radius: 15px;
    padding: 1.15rem;
    margin-bottom: 1rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.18);
}

.info-title {
    color: #64748b;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.info-value {
    color: #f8fafc;
    font-size: 1rem;
    font-weight: 650;
    margin-top: 0.35rem;
}

section[data-testid="stSidebar"] {
    background: #0f1420;
    border-right: 1px solid rgba(148, 163, 184, 0.10);
}

section[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #f8fafc;
}

[data-testid="stFileUploader"] {
    background: #111827;
    border: 1px dashed #334155;
    border-radius: 12px;
    padding: 0.4rem;
}

[data-testid="stFileUploader"] section {
    background: transparent;
}

.stButton > button {
    width: 100%;
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1rem;
    font-weight: 650;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: #3b82f6;
    border: none;
    transform: translateY(-1px);
}

[data-testid="stChatMessage"] {
    background: rgba(17, 24, 39, 0.70);
    border: 1px solid rgba(148, 163, 184, 0.10);
    border-radius: 15px;
    padding: 1rem;
    margin-bottom: 0.8rem;
}

[data-testid="stChatInput"] {
    background: #111827;
    border: 1px solid #334155;
    border-radius: 14px;
}

[data-testid="stChatInput"] textarea {
    color: #f8fafc !important;
}

.source-box {
    background: #111827;
    border: 1px solid #273449;
    border-radius: 10px;
    padding: 0.9rem;
    margin-bottom: 0.7rem;
}

.source-title {
    color: #cbd5e1;
    font-weight: 650;
}

.source-text {
    color: #94a3b8;
    font-size: 0.88rem;
    margin-top: 0.4rem;
    line-height: 1.6;
}

.status {
    background: rgba(37, 99, 235, 0.08);
    border: 1px solid rgba(59, 130, 246, 0.18);
    color: #93c5fd;
    padding: 0.85rem 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}

[data-testid="stExpander"] {
    background: #0f172a;
    border: 1px solid #273449;
    border-radius: 12px;
}

hr {
    border-color: rgba(148, 163, 184, 0.12);
}

[data-testid="stAlert"] {
    border-radius: 10px;
}

.custom-footer {
    text-align: center;
    color: #64748b;
    font-size: 0.8rem;
    padding-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>📚 RAG-Powered Document Q&A</h1>
    <p>
        Ask intelligent questions about your documents
        using Retrieval-Augmented Generation.
    </p>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="info-card">
        <div class="info-title">LLM Provider</div>
        <div class="info-value">⚡ Groq</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    document_status = (
        st.session_state.document_name
        if st.session_state.document_name
        else "No document"
    )

    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-title">Document</div>
            <div class="info-value">📄 {document_status}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with st.sidebar:
    st.markdown("## 📄 Document")
    st.caption("Upload a PDF or TXT document to begin.")

    uploaded_file = st.file_uploader(
        "Choose a document",
        type=["pdf", "txt"],
        help="Supported formats: PDF and TXT"
    )

    if uploaded_file:
        st.info(f"Selected: **{uploaded_file.name}**")

        if st.button("🚀 Process Document", type="primary"):
            tmp_path = None

            try:
                with st.spinner(
                    "Reading, chunking and embedding..."
                ):
                    suffix = os.path.splitext(
                        uploaded_file.name
                    )[1]

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=suffix
                    ) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    docs = load_document(tmp_path)
                    chunks = split_documents(docs)
                    embeddings = get_embeddings()

                    db = build_vector_store(
                        chunks,
                        embeddings
                    )

                    save_vector_store(
                        db,
                        VECTOR_STORE_DIR
                    )

                    st.session_state.vector_db = db
                    st.session_state.document_name = (
                        uploaded_file.name
                    )
                    st.session_state.chunk_count = len(chunks)
                    st.session_state.messages = []

                st.success(
                    f"✓ Processed {len(chunks)} chunks"
                )

            except Exception as e:
                st.error(f"Processing failed: {e}")

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    st.divider()

    st.markdown("### ⚙️ Configuration")
    st.write(f"**Provider:** `{LLM_PROVIDER}`")

    if st.session_state.chunk_count:
        st.write(
            f"**Chunks:** `{st.session_state.chunk_count}`"
        )

    st.divider()

    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

st.markdown("## 💬 Ask Questions")

if st.session_state.vector_db is None:
    st.markdown("""
    <div class="status">
        📄 Upload and process a document from the
        sidebar to start asking questions.
    </div>
    """, unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input(
    "Ask something about your document..."
)

if question:
    if st.session_state.vector_db is None:
        st.warning(
            "Please upload and process a document first."
        )
    else:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner(
                    "Searching document and thinking..."
                ):
                    llm = get_llm()

                    chain = build_rag_chain(
                        st.session_state.vector_db,
                        llm
                    )

                    result = chain.invoke(question)

                st.write(result["answer"])

                with st.expander(
                    f"📚 Sources • "
                    f"{len(result['context'])} chunks"
                ):
                    for i, doc in enumerate(
                        result["context"]
                    ):
                        page = doc.metadata.get(
                            "page",
                            "N/A"
                        )

                        st.markdown(
                            f"""
                            <div class="source-box">
                                <div class="source-title">
                                    📄 Chunk {i + 1}
                                    &nbsp; • &nbsp;
                                    Page {page}
                                </div>
                                <div class="source-text">
                                    {doc.page_content[:500]}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"]
                    }
                )

            except Exception as e:
                st.error(
                    f"Unable to generate answer: {e}"
                )

st.markdown("""
<div class="custom-footer">
    RAG-Powered Document Q&A
    &nbsp; • &nbsp;
    Groq
    &nbsp; • &nbsp;
    Hugging Face
    &nbsp; • &nbsp;
    FAISS
    &nbsp; • &nbsp;
    LangChain
</div>
""", unsafe_allow_html=True)