# 📚 RAG-Powered Document Q&A

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF or TXT documents and ask questions about their content.

The application uses LangChain for the RAG pipeline, Hugging Face Sentence Transformers for document embeddings, FAISS for vector similarity search, and Groq for fast LLM inference.

## 🚀 Features

- 📄 Upload PDF and TXT documents
- ✂️ Automatically split documents into smaller chunks
- 🔎 Semantic similarity search using FAISS
- 🧠 Generate answers using Groq LLM
- 📚 Display source chunks used to generate answers
- 💬 Interactive conversational interface
- 🌙 Modern dark-themed Streamlit UI
- ⚡ Fast inference using Groq
- 🧪 Automated tests for the RAG pipeline
- 🔐 API keys managed through environment variables

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application UI |
| LangChain | RAG pipeline and document processing |
| Groq | LLM inference |
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
```

## 📂 Project Structure

```text
rag-document-qa/
│
├── app.py
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
│
├── sample_docs/
│   └── sample.txt
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── document_loader.py
│   ├── llm_provider.py
│   ├── vector_store.py
│   └── rag_pipeline.py
│
└── tests/
    └── test_pipeline.py
```

## ⚙️ How It Works

### 1. Document Upload

The user uploads a PDF or TXT document through the Streamlit interface.

### 2. Document Loading

The application loads the document using LangChain document loaders.

### 3. Text Splitting

Large documents are divided into smaller chunks using `RecursiveCharacterTextSplitter`.

Default configuration:

```text
Chunk Size: 1000
Chunk Overlap: 200
```

### 4. Embeddings

Each document chunk is converted into a vector representation using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

### 5. Vector Storage

The generated embeddings are stored in a FAISS vector database.

### 6. Retrieval

When the user asks a question, FAISS searches for the most relevant document chunks.

### 7. Generation

The retrieved context is passed to the Groq LLM together with the user's question.

The model generates an answer based on the retrieved document context.

### 8. Sources

The application displays the retrieved chunks so the user can inspect the context used to generate the answer.

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/rag-document-qa.git
cd rag-document-qa
```

### 2. Create a Virtual Environment

Windows:

```powershell
python -m venv .venv
```

### 3. Activate the Virtual Environment

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, you can install dependencies directly using:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

## 🔑 Environment Variables

Create a `.env` file in the project root.

Use the following structure:

```env
LLM_PROVIDER=groq

GROQ_API_KEY=your_groq_api_key_here

GROQ_CHAT_MODEL=openai/gpt-oss-120b

HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=4

VECTOR_STORE_DIR=vector_store
```

Never commit your `.env` file to GitHub.

The repository includes `.env.example` as a template.

## ▶️ Run the Application

Start the Streamlit application:

```powershell
python -m streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

## 💬 Example Usage

1. Open the application.
2. Upload a PDF or TXT document.
3. Click **Process Document**.
4. Wait for the document to be processed.
5. Enter a question about the document.
6. View the generated answer.
7. Expand the **Sources** section to inspect the retrieved chunks.

Example questions:

```text
What is the main idea of this document?

Summarize the document.

What are the key points discussed?

Explain the concept mentioned in the document.

What does the document say about [topic]?
```

## 🧪 Running Tests

The project includes automated tests for:

- Document splitting
- FAISS vector store
- RAG chain

Run:

```powershell
python -m pytest tests/ -v
```

Expected result:

```text
3 passed
```

The tests use deterministic and fake components where appropriate, so they don't require real LLM API calls.

## 🔒 Security

The Groq API key is loaded through environment variables.

Do not upload:

```text
.env
.venv/
vector_store/
__pycache__/
.pytest_cache/
```

These files are excluded through `.gitignore`.

## 🌐 Deployment

The application can be deployed using Streamlit Community Cloud.

Deployment flow:

```text
GitHub Repository
        ↓
Streamlit Community Cloud
        ↓
Configure Secrets
        ↓
Deploy
        ↓
Public Streamlit Application
```

For deployment, add your API key through the platform's secret management instead of committing `.env` to GitHub.

## 📈 Future Improvements

- Support for DOCX files
- Multiple document support
- Persistent vector databases
- Conversation memory
- Streaming LLM responses
- Document management
- Improved source citations
- Authentication
- User-specific document collections
- Cloud-based vector storage
- Improved document metadata handling

## 👨‍💻 Author

**Mrinmoy Jana**

B.Tech Computer Science & Engineering (Data Science)

## ⭐ Project Goal

This project demonstrates the implementation of a complete Retrieval-Augmented Generation pipeline by combining document processing, semantic search, vector databases, LLM inference, and an interactive web interface.
