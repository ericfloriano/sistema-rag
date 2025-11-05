import streamlit as st
import time
import os
from rag_core import get_rag_response, MODEL_NAME_FOR_DISPLAY
from ingest import run_ingestion, PERSIST_DIRECTORY

# --- LÓGICA DE INICIALIZAÇÃO CORRIGIDA ---
SUCCESS_FLAG_FILE = os.path.join(PERSIST_DIRECTORY, "ingest_success.flag")

@st.cache_resource # Isso garante que só rode UMA VEZ
def initialize_database():
    """
    Verifica se a ingestão já foi concluída com sucesso. 
    Se não, executa a função run_ingestion() para criá-la.
    """
    # Procura pelo "arquivo de sucesso" em vez da pasta
    if not os.path.exists(SUCCESS_FLAG_FILE):
        st.info("Primeira inicialização: Criando a base de conhecimento...")
        
        with st.spinner("Lendo documentos e criando o banco de vetores... (Isso pode levar alguns minutos)"):
            success, message = run_ingestion() 
        
        if success:
            st.success(f"Base de conhecimento criada! {message}")
            time.sleep(2)
            # st.rerun() # REMOVIDO!
        else:
            st.error(f"Falha ao criar a base de conhecimento: {message}")
            # Limpa o cache e para, para que ele tente de novo na próxima recarga
            st.cache_resource.clear() 
            st.stop()
    else:
        print("Banco de dados já existe e está pronto. Carregando...")
# --- FIM DA LÓGICA ---

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