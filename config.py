import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Base directories configuration
BASE_DIR = Path(__file__).parent
SOURCE_DIRECTORY = BASE_DIR / "documentos_fonte"
PERSIST_DIRECTORY = BASE_DIR / "qdrant_db"

# File paths & Qdrant Settings
SUCCESS_FLAG_FILE = PERSIST_DIRECTORY / "ingest_success.flag"
COLLECTION_NAME = "recare_knowledge_base"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# LLM & Embeddings Configuration
LLM_PROVIDER = "gemini"
GEMINI_MODEL_NAME = "gemini-2.0-flash"
EMBEDDINGS_MODEL_NAME = "models/gemini-embedding-001"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Observability (LLM-Ops) via LangSmith
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "ReCARE-RAG")

# Reranker Configuration (Optimized for Free-Tier CPU Servers)
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# RAG Hyperparameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
VECTOR_SEARCH_K = 10 # Retrieves top 10 from Qdrant Hybrid Search to prevent Reranker overload
RERANKER_TOP_K = 4   # Retains the top 4 best chunks post-compression to feed the LLM
INGEST_BATCH_SIZE = 50
