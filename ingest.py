import os
import time
import logging
import uuid
import shutil
from pathlib import Path
from tenacity import retry, wait_exponential, stop_after_attempt

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)
from langchain_core.documents import Document

# --- QDRANT IMPORTS ---
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from config import (
    SOURCE_DIRECTORY,
    PERSIST_DIRECTORY,
    SUCCESS_FLAG_FILE,
    COLLECTION_NAME,
    EMBEDDINGS_MODEL_NAME,
    GOOGLE_API_KEY,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    INGEST_BATCH_SIZE,
    QDRANT_URL,
    QDRANT_API_KEY
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configure tenacity retry strategy to handle API ratelimits
@retry(
    wait=wait_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(5),
    reraise=True
)
def add_documents_with_retry(db, batch_chunks):
    """Adds documents with exponential backoff on failure."""
    db.add_documents(documents=batch_chunks)

def run_ingestion():
    """
    Executes the document ingestion process into Qdrant Hybrid Cloud/Local.
    Returns (True, "success message") or (False, "error message").
    """
    try:
        logger.info(f"Starting ingestion... Reading from: {SOURCE_DIRECTORY}")
        
        if not SOURCE_DIRECTORY.exists():
            msg = f"ERROR: Source directory '{SOURCE_DIRECTORY}' not found."
            logger.error(msg)
            return False, msg

        logger.info("Scanning for files in source directory...")
        filepaths = list(SOURCE_DIRECTORY.rglob("*"))
        
        all_documents = []
        for filepath in filepaths:
            if filepath.is_file() and filepath.suffix.lower() in [".pdf", ".docx", ".txt"]:
                logger.info(f"Processing file: {filepath}")
                if filepath.suffix.lower() == ".pdf":
                    loader = PyPDFLoader(str(filepath))
                elif filepath.suffix.lower() == ".docx":
                    loader = Docx2txtLoader(str(filepath))
                else: # .txt
                    loader = TextLoader(str(filepath), encoding="utf-8")
                
                all_documents.extend(loader.load())
            elif filepath.is_file():
                logger.warning(f"Warning: Skipping unsupported file format: {filepath}")

        if not all_documents:
            msg = "ERROR: No valid documents (.pdf, .docx, .txt) were loaded."
            logger.error(msg)
            return False, msg

        logger.info(f"Success: {len(all_documents)} raw document pages loaded.")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = text_splitter.split_documents(all_documents)
        
        # Inject unique chunk IDs for tracking
        for chunk in chunks:
            chunk.metadata["chunk_id"] = str(uuid.uuid4())
            
        logger.info(f"Success: {len(chunks)} contextual chunks generated.")

        logger.info("Loading dense embedding model (Google API) and sparse model (FastEmbed)...")
        if not GOOGLE_API_KEY:
            msg = "ERROR: GOOGLE_API_KEY not found. Please check your .env variables."
            logger.error(msg)
            return False, msg
            
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model=EMBEDDINGS_MODEL_NAME,
            google_api_key=GOOGLE_API_KEY
        )
        
        # FastEmbed Sparse creates the "BM25-like" sparse vectors for native keyword search
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

        logger.info(f"Connecting to Qdrant HYBRID DB at '{PERSIST_DIRECTORY}'...")
        
        # Remove old local DB directory to prevent schema conflicts if running locally
        if PERSIST_DIRECTORY.exists():
            logger.info("Cleaning up existing Qdrant collection to ensure a pristine slate...")
            shutil.rmtree(PERSIST_DIRECTORY)

        # 1. Instantiate a LangChain Qdrant Client (Hybrid setup via from_documents)
        if QDRANT_URL and QDRANT_API_KEY:
            logger.info("Connecting to Qdrant Cloud Cluster...")
            db = QdrantVectorStore.from_documents(
                [],
                embedding=embeddings_model,
                sparse_embedding=sparse_embeddings,
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                collection_name=COLLECTION_NAME,
                retrieval_mode="hybrid"
            )
        else:
            logger.info(f"Connecting to Local Qdrant DB at '{PERSIST_DIRECTORY}'...")
            db = QdrantVectorStore.from_documents(
                [],
                embedding=embeddings_model,
                sparse_embedding=sparse_embeddings,
                path=str(PERSIST_DIRECTORY),
                collection_name=COLLECTION_NAME,
                retrieval_mode="hybrid"
            )
        
        # 2. Add documents in chunks / batches to circumvent API ratelimits
        total_batches = (len(chunks) // INGEST_BATCH_SIZE) + 1
        for i in range(0, len(chunks), INGEST_BATCH_SIZE):
            batch_chunks = chunks[i:i + INGEST_BATCH_SIZE]
            logger.info(f"Processing batch {i//INGEST_BATCH_SIZE + 1} of {total_batches}...")
            
            add_documents_with_retry(db, batch_chunks)
            time.sleep(1) # Extra slight pause to respect Google/Qdrant Cloud limits
        
        # 3. Create the "success flag" file
        PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)
        with open(SUCCESS_FLAG_FILE, "w") as f:
            f.write("ok")
        
        msg = f"Success! {len(all_documents)} documents ingested via Hybrid Pipeline, {len(chunks)} chunks successfully pushed to Qdrant."
        logger.info(msg)

        return True, msg

    except Exception as e:
        msg = f"UNEXPECTED ERROR in run_ingestion: {e}"
        logger.error(msg, exc_info=True)
        return False, msg

if __name__ == "__main__":
    success, message = run_ingestion()
    if success:
        logger.info(f"✅ Qdrant Ingestion completed successfully! {message}")
    else:
        logger.error(f"❌ Ingestion failed: {message}")