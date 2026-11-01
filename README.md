
# RAG-Powered Document Q&A

Ever had a long PDF and struggled to find a specific piece of information in it?

That's exactly why I built this project.

**RAG-Powered Document Q&A** is an application that lets you upload a PDF or text file and ask questions about its content in plain English. Instead of searching through pages manually, you can simply ask a question and get an answer based on the uploaded document.

Every answer also includes the source chunks used to generate it, making it easier to verify the information.

## How It Works

The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant information before generating an answer.

```text
Document (PDF/TXT)
       |
       v
 Text Extraction
       |
       v
  Text Chunking
       |
       v
  Generate Embeddings
       |
       v
  FAISS Vector Store
       |
       |
User Question
       |
       v
 Retrieve Relevant Chunks
       |
       v
   Language Model
       |
       v
 Grounded Answer + Sources
```

Here's what happens behind the scenes:

1. **Upload a document:** Upload a PDF or TXT file through the Streamlit interface.
2. **Split into chunks:** The document is divided into smaller pieces of text.
3. **Generate embeddings:** Each chunk is converted into a numerical representation using an embedding model.
4. **Store in FAISS:** The embeddings are saved in a FAISS vector database.
5. **Ask a question:** Enter your question in the chat interface.
6. **Retrieve relevant information:** FAISS finds the most relevant chunks from the uploaded document.
7. **Generate an answer:** The LLM uses the retrieved content to generate an answer along with its source references.

The main idea is to make the model answer using the uploaded document rather than relying on its general knowledge.

## Features

- Upload PDF and TXT files directly through the interface.
- Ask questions about documents using natural language.
- Get answers based on the uploaded content.
- View the source chunks used to generate each answer.
- See page references for PDF documents.
- Maintain conversation history while asking follow-up questions.
- Switch between different LLM providers without changing the main RAG pipeline.
- Save FAISS indexes locally and reuse them after restarting the application.
- Run automated tests without requiring API keys or an internet connection.

## Technologies Used

| Component | Technology |
|---|---|
| Programming Language | Python |
| LLM Orchestration | LangChain (LCEL) |
| Vector Database | FAISS |
| Embeddings | OpenAI Embeddings / Sentence Transformers |
| Language Models | OpenAI / Hugging Face / Grok |
| Frontend | Streamlit |
| Testing | Pytest |
| Configuration | Python dotenv 

## Deploying a live demo

live: https://rag-document-q.streamlit.app/



## License

MIT — see [LICENSE](LICENSE).
