import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
# --- MUDANÇA AQUI ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings # Importa o modelo do Google

# Carrega as chaves (necessário para o Google API Key)
load_dotenv()

# Caminho para os documentos e para o banco de vetores (agora no disco persistente)
SOURCE_DIRECTORY = "documentos_fonte"
PERSIST_DIRECTORY = "/var/data/chroma_db" # Caminho do Render

def run_ingestion():
    """
    Executa o processo de ingestão de documentos.
    Retorna (True, "mensagem de sucesso") ou (False, "mensagem de erro").
    """
    try:
        print(f"Iniciando ingestão... Lendo de: {SOURCE_DIRECTORY}")
        
        if not os.path.exists(SOURCE_DIRECTORY):
            msg = f"ERRO: Pasta fonte '{SOURCE_DIRECTORY}' não encontrada."
            print(msg)
            return False, msg

        if not os.listdir(SOURCE_DIRECTORY):
            msg = f"ERRO: Pasta fonte '{SOURCE_DIRECTORY}' está vazia."
            print(msg)
            return False, msg
        
        print(f"Arquivos encontrados na fonte: {os.listdir(SOURCE_DIRECTORY)}")

        loader = DirectoryLoader(
            SOURCE_DIRECTORY,
            glob="**/*",
            recursive=True,
            show_progress=True,
            use_multithreading=True,
        )

        print("Carregando documentos...")
        documentos = loader.load()

        if not documentos:
            msg = "ERRO: O Loader não conseguiu carregar nenhum documento (verifique os tipos de arquivo)."
            print(msg)
            return False, msg

        print(f"Sucesso: {len(documentos)} documentos carregados.")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documentos)
        print(f"Sucesso: {len(chunks)} chunks criados.")

        # --- MUDANÇA PRINCIPAL AQUI ---
        print("Carregando modelo de embeddings (Google API)...")
        # Verifica se a chave foi carregada
        if not os.getenv("GOOGLE_API_KEY"):
            msg = "ERRO: GOOGLE_API_KEY não encontrada. Verifique as variáveis de ambiente."
            print(msg)
            return False, msg
            
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        # --- FIM DA MUDANÇA ---

        print(f"Criando e persistindo banco de vetores em '{PERSIST_DIRECTORY}'...")
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            persist_directory=PERSIST_DIRECTORY,
        )
        
        msg = f"Sucesso! {len(documentos)} documentos ingeridos, {len(chunks)} chunks criados."
        print(msg)
        return True, msg

    except Exception as e:
        msg = f"ERRO INESPERADO no run_ingestion: {e}"
        print(msg)
        import traceback
        traceback.print_exc() # Imprime o stack trace completo no log
        return False, msg

if __name__ == "__main__":
    success, message = run_ingestion()
    if success:
        print(f"✅ Ingestão concluída com sucesso! {message}")
    else:
        print(f"❌ Falha na ingestão: {message}")