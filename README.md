RAG-Powered Document Q&A

Ever had a long PDF and struggled to find a specific piece of information in it?

That's exactly why I built this project.

RAG-Powered Document Q&A is a simple application that lets you upload a PDF or text file and ask questions about its content in plain English. Instead of searching through pages manually, you can simply ask a question and get an answer based on the uploaded document.

The best part? Every answer comes with the source text used to generate it, so you can check where the information came from.

How It Works

The application uses Retrieval-Augmented Generation (RAG) to find relevant information before generating an answer.

Here's what happens behind the scenes:

Upload a document: Upload a PDF or TXT file through the Streamlit interface.

Split into chunks: The document is divided into smaller pieces of text.

Generate embeddings: Each chunk is converted into a numerical representation using an embedding model.

Store in FAISS: The embeddings are saved in a FAISS vector database for faster retrieval.

Ask a question: Enter your question in the chat interface.

Retrieve relevant information: FAISS finds the most relevant chunks from the uploaded document.

Generate an answer: The LLM uses the retrieved content to generate an answer, along with its source references.

The idea is simple: the model should answer based on the document rather than relying on its general knowledge.

Features

Upload PDF and TXT files directly through the interface.

Ask questions about your documents using natural language.

Get answers based on the uploaded content.

View the source chunks used to generate each answer.

See page references for PDF documents.

Maintain conversation history while asking follow-up questions.

Switch between different LLM providers without changing the main RAG pipeline.

Save FAISS indexes locally and reuse them after restarting the application.

Run automated tests without requiring API keys or an internet connection.

Technologies Used

Component

Technology

Programming Language

Python

LLM Orchestration

LangChain (LCEL)

Vector Database

FAISS

Embeddings

OpenAI Embeddings / Sentence Transformers

Language Models

OpenAI / Hugging Face / Grok

Frontend

Streamlit

Testing

Pytest

Configuration

Python dotenv

## Deploying a live demo (free)

live: https://rag-document-q.streamlit.app/



## License

MIT — see [LICENSE](LICENSE).
