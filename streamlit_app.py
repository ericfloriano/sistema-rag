import streamlit as st
import time
import os
import sys
from rag_core import get_rag_response, MODEL_NAME_FOR_DISPLAY 

# --- NOVO: LÓGICA DE INICIALIZAÇÃO ---
# Define o caminho para o banco de dados
PERSIST_DIRECTORY = "chroma_db"

@st.cache_resource # Isso garante que só rode UMA VEZ
def initialize_database():
    """
    Verifica se o banco de dados existe na nuvem. 
    Se não, executa o ingest.py para criá-lo.
    """
    if not os.path.exists(PERSIST_DIRECTORY):
        # Escreve uma mensagem para o usuário no app
        st.info("Primeira inicialização: Criando a base de conhecimento...")
        st.warning("Isso pode demorar 1-2 minutos. O app carregará em seguida.")
        
        # Encontra o executável do python (importante para o Streamlit Cloud)
        python_executable = sys.executable
        
        try:
            # Roda o script ingest.py
            # Usamos 'st.spinner' para mostrar que algo está acontecendo
            with st.spinner(f"Executando `python ingest.py`... Lendo documentos..."):
                result = os.system(f"{python_executable} ingest.py")
            
            if result == 0:
                st.success("Base de conhecimento criada! O assistente está pronto.")
                time.sleep(2) # Pausa para o usuário ler
                st.rerun() # Recarrega a página agora que o DB existe
            else:
                st.error("Ocorreu um erro crítico ao criar a base de conhecimento.")
                st.stop() # Para o app
        except Exception as e:
            st.error(f"Erro ao executar a ingestão: {e}")
            st.stop()
    else:
        # Isso só será impresso nos logs do servidor, não no app
        print("Banco de dados já existe. Carregando...")

# --- FIM DA NOVA LÓGICA ---

# --- Executa a inicialização ---
# Isso rodará antes de qualquer outra coisa
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