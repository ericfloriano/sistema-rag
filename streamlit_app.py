import streamlit as st
import time
import os
from rag_core import get_rag_response, get_rag_chain # Importa o CONSTRUTOR da chain

# --- O 'CÉREBRO' É CARREGADO AQUI, UMA VEZ ---
@st.cache_resource # Armazena o "cérebro" em cache
def load_rag_chain_cached():
    """Carrega a cadeia RAG (que agora inclui o DB e o BM25 pré-carregados)."""
    print("Carregando a cadeia RAG para o Streamlit...")
    chain = get_rag_chain()
    if chain is None:
        print("Falha ao carregar a cadeia RAG.")
    return chain

# Carrega a cadeia na inicialização
chain = load_rag_chain_cached()

# --- Interface do Streamlit ---
st.title("🧠 ReCARE FastLearn")
st.subheader("🤖 Agent Assist")
st.caption("⚙️ Este assistente está em constante evolução. Algumas respostas ainda podem estar fora do meu escopo de resposta.")

if chain is None:
    st.error("Falha fatal ao carregar a cadeia RAG. Verifique os logs do servidor.")
    st.stop()

# Pega o nome do modelo de dentro da 'chain'
try:
    llm_model_name = chain.middle[1].model
except Exception:
    llm_model_name = "gemini-2.0-flash" # Fallback

# st.caption(f"Utilizando o modelo: {llm_model_name}")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Olá! Como posso ajudar você hoje?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Faça sua pergunta sobre processos ou produtos..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Buscando nos documentos... ⏳")
        try:
            full_response = get_rag_response(chain, prompt)
            
            # Simulação de streaming
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
            full_response = f"Ocorreu um erro: {e}"
            message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})