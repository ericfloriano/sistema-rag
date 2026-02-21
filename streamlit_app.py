import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import time
import logging
from rag_core import get_rag_response, get_rag_chain
from config import GEMINI_MODEL_NAME

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- THE RAG BRAIN IS LOADED HERE ONCE ---
@st.cache_resource # Caches the brain across sessions
def load_rag_chain_cached():
    """Loads and caches the LLM RAG Chain."""
    logger.info("Loading RAG chain for Streamlit UI...")
    try:
        chain = get_rag_chain()
        return chain
    except Exception as e:
        logger.error(f"Failed to load RAG chain: {e}", exc_info=True)
        return None

# Load chain on startup
chain = load_rag_chain_cached()

# --- Streamlit Front-End Interface ---
st.title("🧠 ReCARE FastLearn")
st.subheader("🤖 Agent Assist")
st.caption("⚙️ Este assistente de IA focado no equipamento ReCARE está em constante evolução. Respostas são geradas com base estrita nos manuais. Algumas questões podem estar fora do escopo coberto.")

if chain is None:
    st.error("Falha fatal ao carregar a Inteligência Artificial. A equipe técnica foi notificada do erro sistêmico. Verifique logs do servidor.")
    st.stop()

# Fallback to get LLM name
llm_model_name = GEMINI_MODEL_NAME

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Olá! Bem-vindo ao portal de conhecimento especializado do equipamento ReCARE. Como eu posso te ajudar na sua dúvida hoje?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ex: 'Quantos canais o equipamento possui?'..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Buscando no banco de conhecimento oficial... ⏳")
        try:
            full_response = get_rag_response(chain, prompt)
            
            # Simple streaming simulation for a smooth UX
            response_chunks = full_response.split()
            streamed_response = ""
            if response_chunks:
                for chunk in response_chunks:
                    streamed_response += chunk + " "
                    message_placeholder.markdown(streamed_response + "▌")
                    time.sleep(0.05)
                message_placeholder.markdown(full_response)
            else:
                 message_placeholder.markdown(full_response)
        
        except Exception as e:
            logger.error(f"Request processing error in Streamlit: {e}", exc_info=True)
            full_response = "Ocorreu um erro técnico na comunicação com o LLM. Por favor, tente novamente mais tarde."
            message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})