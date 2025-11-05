import streamlit as st
import time
import os
# --- MUDANÇA: Importações diferentes ---
from rag_core import get_rag_response, get_rag_chain # Importa o CONSTRUTOR da chain
from ingest import run_ingestion, PERSIST_DIRECTORY

SUCCESS_FLAG_FILE = os.path.join(PERSIST_DIRECTORY, "ingest_success.flag")

@st.cache_resource # Isso garante que só rode UMA VEZ
def initialize_database():
    """
    Verifica se a ingestão já foi concluída com sucesso. 
    Se não, executa a função run_ingestion() para criá-la.
    """
    if not os.path.exists(SUCCESS_FLAG_FILE):
        st.info("Primeira inicialização: Criando a base de conhecimento...")
        
        with st.spinner("Lendo documentos e criando o banco de vetores... (Isso pode levar alguns minutos)"):
            success, message = run_ingestion() 
        
        if success:
            st.success(f"Base de conhecimento criada! {message}")
            time.sleep(2)
        else:
            st.error(f"Falha ao criar a base de conhecimento: {message}")
            st.cache_resource.clear() 
            st.stop()
    else:
        print("Banco de dados já existe e está pronto. Carregando...")

# --- Executa a inicialização PRIMEIRO ---
initialize_database()

# --- MUDANÇA: Carrega a 'chain' DEPOIS da inicialização ---
@st.cache_resource # Armazena o "cérebro" em cache
def load_rag_chain():
    """Carrega a cadeia RAG agora que o DB está pronto."""
    print("Carregando a cadeia RAG para o Streamlit...")
    return get_rag_chain()

chain = load_rag_chain()

if chain is None:
    st.error("Falha ao carregar a cadeia RAG. Verifique os logs do servidor.")
    st.stop()

# --- O RESTANTE DO SEU CÓDIGO ---
st.title("🧠 Assistente de Conhecimento Interno")
st.caption(f"Utilizando o modelo: {chain.middle[1].model}") # Pega o nome do modelo da 'chain' carregada

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
            # --- MUDANÇA: Passa a 'chain' como argumento ---
            full_response = get_rag_response(chain, prompt)
            
            # (O código de streaming permanece o mesmo...)
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