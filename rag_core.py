import os
from dotenv import load_dotenv
load_dotenv()

import warnings
import logging
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# --- QDRANT & RERANKER IMPORTS ---
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse
from qdrant_client import QdrantClient

from config import (
    PERSIST_DIRECTORY,
    COLLECTION_NAME,
    EMBEDDINGS_MODEL_NAME,
    GEMINI_MODEL_NAME,
    GOOGLE_API_KEY,
    LLM_PROVIDER,
    RERANKER_MODEL_NAME,
    VECTOR_SEARCH_K,
    RERANKER_TOP_K,
    QDRANT_URL,
    QDRANT_API_KEY
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning, module="langchain")

RAG_PROMPT_TEMPLATE = """
ATTENTION: You are a polite and helpful AI assistant focused on answering questions about the company's internal processes and products.
CONTEXT:
{context}
QUESTION:
{question}
INSTRUCTIONS:
1. First, analyze the QUESTION: Is it PURELY conversational (e.g., just greetings like "olá", "bom dia", "tudo bem?") with no specific subject? If YES, respond politely and naturally in a friendly tone without using the CONTEXT. Briefly introduce yourself as the ReCARE AI Assistant and ask how you can help.
2. If the QUESTION mentions ANY product (like ReCARE, RePAD), process, or asks for specific information, it is a FACTUAL question. For factual questions, you MUST answer **strictly** based on the provided CONTEXT.
3. CRITICAL RULE: If the QUESTION is factual (not a simple greeting) and the CONTEXT does not contain the answer, you are FORBIDDEN to use the polite greeting fallback. You MUST say **exactly**: "Desculpe, não tenho informações sobre isso no meu banco de dados."
4. Do not invent information, make assumptions, or use external knowledge for factual questions.
5. Answer in Brazilian Portuguese, clearly and objectively.
"""

def get_llm(provider: str):
    if provider == "gemini":
        logger.info(f"Loading LLM: Google Gemini ({GEMINI_MODEL_NAME})")
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_NAME,
            temperature=0.0,
            google_api_key=GOOGLE_API_KEY
        )
    else:
        raise ValueError(f"Unknown LLM Provider: {provider}.")

def get_rag_chain():
    """
    Function that BUILDS and RETURNS the HYBRID RAG + RERANKER chain.
    """
    llm = get_llm(LLM_PROVIDER)

    logger.info("Loading dense and sparse embedding models...")
    if not GOOGLE_API_KEY:
         raise ValueError("GOOGLE_API_KEY not found.")

    # 1. Base Embeddings (Gemini Dense + FastEmbed Sparse)
    dense_embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDINGS_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY
    )
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    logger.info(f"Loading Qdrant Database: {PERSIST_DIRECTORY}")
    
    if QDRANT_URL and QDRANT_API_KEY:
        logger.info("Connecting to Qdrant Cloud...")
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    else:
        logger.info("Connecting to Local Qdrant DB...")
        if not PERSIST_DIRECTORY.exists():
             raise FileNotFoundError(f"Qdrant database not found at {PERSIST_DIRECTORY}. Run ingest.py first.")
        qdrant_client = QdrantClient(path=str(PERSIST_DIRECTORY))
    
    # Validates if the collection exists before loading
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        raise ValueError(f"Collection '{COLLECTION_NAME}' does not exist in the Qdrant DB.")
        
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode="hybrid"
    )

    USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"
    
    # 2. Base Retriever (Retrieves K documents via hybrid search: Dense + BM25)
    logger.info("Setting up Base Retriever (Qdrant Hybrid)...")
    base_retriever = vector_store.as_retriever(
        search_kwargs={"k": VECTOR_SEARCH_K if USE_RERANKER else 4} # Fetch a large K for the Reranker, or standard K if disabled
    )

    if USE_RERANKER:
        # 3. Reranker Pipeline (CrossEncoder)
        logger.info(f"Loading Local Reranker ({RERANKER_MODEL_NAME}). This might download weights on first use...")
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
        from langchain_classic.retrievers import ContextualCompressionRetriever
        
        cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL_NAME)
        
        compressor = CrossEncoderReranker(
            model=cross_encoder, 
            top_n=RERANKER_TOP_K # From the original K, only pass the top N to the LLM
        )

        # 4. Final Hybrid Compressor (Combines Search + Re-Ordering from BGE)
        final_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=base_retriever
        )
    else:
        logger.info("Reranker is DISABLED via environment variables. Using standard Hybrid Search to save RAM.")
        final_retriever = base_retriever

    rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    def format_docs(docs: List[Document]):
        if not docs:
            return "Nenhum contexto relevante encontrado."
        return "\n\n".join(doc.page_content for doc in docs)

    logger.info("Assembling Hybrid LCEL Chain...")
    rag_chain = (
        {"context": final_retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    
    logger.info(f"RAG Chain (Qdrant Hybrid={'Reranked' if USE_RERANKER else 'Base'}) assembled successfully!")
    return rag_chain

# --- Response Functions ---

def get_rag_response(chain, question_text: str) -> str:
    if chain is None:
        raise ValueError("The RAG chain was not initialized correctly.")
    if not question_text:
        return "Por favor, faça uma pergunta."
    
    response = chain.invoke(question_text)
    return response

async def get_rag_response_async(chain, question_text: str) -> str:
    if chain is None:
        raise ValueError("The RAG chain was not initialized correctly.")
    if not question_text:
        return "Por favor, faça uma pergunta."
    
    response = await chain.ainvoke(question_text)
    return response