# 📚 RAG-Powered Document Q&A

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF or TXT documents and ask questions about their content.

The application uses LangChain for the RAG pipeline, Hugging Face Sentence Transformers for document embeddings, FAISS for vector similarity search, and Groq for fast LLM inference.

## 🚀 Features

- 📄 Upload PDF and TXT documents
- ✂️ Automatically split documents into smaller chunks
- 🔎 Perform semantic similarity search using FAISS
- 🧠 Generate answers using Groq LLM
- 📚 Display the source chunks used to generate answers
- 💬 Interactive conversational interface
- 🌙 Modern dark-themed Streamlit UI
- ⚡ Fast inference using Groq
- 🧪 Automated tests for the core RAG pipeline
- 🔐 API keys managed through environment variables

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application UI |
| LangChain | RAG pipeline and document processing |
| Groq | Large Language Model inference |
| Hugging Face | Document embeddings |
| FAISS | Vector similarity search |
| PyPDF | PDF document loading |
| pytest | Automated testing |

## 🧠 Architecture

```text
                    PDF / TXT
                        │
                        ▼
               Document Loader
                        │
                        ▼
                 Text Splitting
                        │
                        ▼
            Hugging Face Embeddings
                        │
                        ▼
                      FAISS
                 Vector Database
                        │
                        ▼
                   Retriever
                        │
                        ▼
                  RAG Prompt
                        │
                        ▼
                 Groq LLM
                        │
                        ▼
                   Final Answer
                        │
                        ▼
                  Source Chunks
