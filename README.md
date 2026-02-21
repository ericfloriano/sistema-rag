# ReCARE AI Assistant - "First-Class" RAG System

A production-ready Retrieval-Augmented Generation (RAG) system built to answer questions accurately based on internal documents and equipment manuals.

## 🌟 Key Features

- **Hybrid Native Search:** Combines semantic searches (Dense Vectors) with exact keyword searches (BM25 Sparse Vectors) using Qdrant.
- **Contextual Document Compression:** Employs a HuggingFace Cross-Encoder (BGE-Reranker) to evaluate and prune irrelevant documents before passing context to the LLM, reducing hallucinations and inference time.
- **LLM-Ops / Observability:** Integrated with LangSmith for real-time monitoring of traces, token usage, latency, and costs.
- **Multi-Interface:** 
  - 🤖 **Telegram Bot**: Asynchronous and fast conversational interface.
  - 🌐 **Streamlit Web App**: Clean, professional web UI with streaming responses.
- **Containerized:** Ready for zero-cost cloud deployments (Render, Koyeb, Streamlit Cloud) via `Dockerfile`.

## 🛠️ Technology Stack

- **Vector Database:** Qdrant (Local / Cloud)
- **Primary LLM:** Google Gemini 2.0 Flash (`gemini-2.0-flash`)
- **Embedding Model:** Google Generative AI Embeddings
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (Optimized for fast CPU inference)
- **Framework:** LangChain 

## 🚀 Getting Started

### 1. Requirements
- Python 3.11+
- [Git](https://git-scm.com/)

### 2. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root of the project with your API keys:
```env
# Google AI Studio (Required for LLM and Embeddings)
GOOGLE_API_KEY=your_google_api_key_here

# Telegram Bot (Required only if running telegram_bot.py)
TELEGRAM_BOT_TOKEN=your_telegram_token_here

# LangSmith Observability (Optional but recommended)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=ReCARE-RAG
```
> ⚠️ **Security Note:** Never commit your `.env` file to GitHub. It is already safely included in the `.gitignore`.

### 4. Build the Knowledge Base
1. Place your `.pdf`, `.txt`, or `.docx` training files into the `documentos_fonte/` directory.
2. Run the ingestion script to vectorize the documents into the local Qdrant database:
```bash
python ingest.py
```
*Note: The first run will download embedding tools and the Reranker model locally.*

### 5. Run the Application
You can run either the web interface or the Telegram bot:

**Web UI (Streamlit):**
```bash
streamlit run streamlit_app.py
```

**Telegram Assistant:**
```bash
python telegram_bot.py
```

## ☁️ Cloud Deployment (Zero Cost)

This project includes a `Dockerfile` pre-configured to run the Streamlit application in containerized environments.

1. Create a free **Qdrant Cloud** cluster.
2. Update `config.py` to point to your new cloud URL and API Key instead of the local path.
3. Push this repository to GitHub.
4. Connect the repository to **Render.com** (Web Service free tier) or **Streamlit Community Cloud**.
5. Add your `.env` secrets directly into the Cloud Provider's environment variables dashboard.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
