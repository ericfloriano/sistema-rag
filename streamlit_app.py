import streamlit as st
import time
import os
from rag_core import get_rag_response, MODEL_NAME_FOR_DISPLAY
from ingest import run_ingestion  # Importa nossa função de ingestão

PERSIST_DIRECTORY = "/var/data/chroma_db"

@st.cache_resource # Isso garante que só rode UMA VEZ
def initialize_database():
    """
    Verifica se o banco de dados existe na nuvem. 
    Se não, executa a função run_ingestion() para criá-lo.
    """
    if not os.path.exists(PERSIST_DIRECTORY):
        st.info("Primeira inicialização: Criando a base de conhecimento...")
        
        with st.spinner("Lendo documentos e criando o banco de vetores... (Isso pode levar 1-2 minutos)"):
            # Chama a função diretamente
            success, message = run_ingestion() 
        
        if success:
            st.success(f"Base de conhecimento criada! {message}")
            time.sleep(2)
            # st.rerun() # REMOVIDO! Esta linha estava causando o loop infinito.
        else:
            st.error(f"Falha ao criar a base de conhecimento: {message}")
            st.stop()
    else:
        print("Banco de dados já existe. Carregando...")

# --- Executa a inicialização ---
initialize_database()

# --- O RESTANTE DO SEU CÓDIGO (sem mudanças) ---
st.title("🧠 Assistente de Conhecimento Interno")
st.caption(f"Utilizando o modelo: {MODEL_NAME_FOR_DISPLAY}")

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
            full_response = get_rag_response(prompt)
            
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