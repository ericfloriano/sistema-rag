import os
import glob
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- NOVAS IMPORTAÇÕES DE LOADERS LEVES ---
from langchain_community.document_loaders import (
    PyPDFLoader,      # Para PDFs
    Docx2txtLoader,   # Para .docx
    TextLoader        # Para .txt
)

load_dotenv()

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

        # --- LÓGICA DE CARREGAMENTO MANUAL ---
        print("Procurando por arquivos em documentos_fonte/...")
        # Encontra todos os arquivos .pdf, .docx, .txt
        filepaths = glob.glob(os.path.join(SOURCE_DIRECTORY, "**/*"), recursive=True)
        
        all_documents = []
        for filepath in filepaths:
            print(f"Processando arquivo: {filepath}")
            if filepath.lower().endswith(".pdf"):
                loader = PyPDFLoader(filepath)
            elif filepath.lower().endswith(".docx"):
                loader = Docx2txtLoader(filepath)
            elif filepath.lower().endswith(".txt"):
                loader = TextLoader(filepath, encoding="utf-8")
            else:
                print(f"Aviso: Pulando arquivo não suportado: {filepath}")
                continue
            
            # Adiciona os documentos carregados à lista
            all_documents.extend(loader.load())
        # --- FIM DA LÓGICA DE CARREGAMENTO ---

        if not all_documents:
            msg = "ERRO: Nenhum documento válido (.pdf, .docx, .txt) foi carregado."
            print(msg)
            return False, msg

        print(f"Sucesso: {len(all_documents)} documentos carregados.")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(all_documents)
        print(f"Sucesso: {len(chunks)} chunks criados.")

        print("Carregando modelo de embeddings (Google API)...")
        if not os.getenv("GOOGLE_API_KEY"):
            msg = "ERRO: GOOGLE_API_KEY não encontrada. Verifique as variáveis de ambiente."
            print(msg)
            return False, msg
            
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        print(f"Criando e persistindo banco de vetores em '{PERSIST_DIRECTORY}'...")
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            persist_directory=PERSIST_DIRECTORY,
        )
        
        msg = f"Sucesso! {len(all_documents)} documentos ingeridos, {len(chunks)} chunks criados."
        print(msg)
        return True, msg

    except Exception as e:
        msg = f"ERRO INESPERADO no run_ingestion: {e}"
        print(msg)
        import traceback
        traceback.print_exc()
        return False, msg

if __name__ == "__main__":
    success, message = run_ingestion()
    if success:
        print(f"✅ Ingestão concluída com sucesso! {message}")
    else:
        print(f"❌ Falha na ingestão: {message}")