import os
import glob
import time
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)

load_dotenv()

SOURCE_DIRECTORY = "documentos_fonte"
PERSIST_DIRECTORY = "/var/data/chroma_db"
SUCCESS_FLAG_FILE = os.path.join(PERSIST_DIRECTORY, "ingest_success.flag")

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

        print("Procurando por arquivos em documentos_fonte/...")
        filepaths = glob.glob(os.path.join(SOURCE_DIRECTORY, "**/*"), recursive=True)
        
        all_documents = []
        for filepath in filepaths:
            if filepath.lower().endswith((".pdf", ".docx", ".txt")):
                print(f"Processando arquivo: {filepath}")
                if filepath.lower().endswith(".pdf"):
                    loader = PyPDFLoader(filepath)
                elif filepath.lower().endswith(".docx"):
                    loader = Docx2txtLoader(filepath)
                else: # .txt
                    loader = TextLoader(filepath, encoding="utf-8")
                
                all_documents.extend(loader.load())
            else:
                print(f"Aviso: Pulando arquivo não suportado: {filepath}")

        if not all_documents:
            msg = "ERRO: Nenhum documento válido (.pdf, .docx, .txt) foi carregado."
            print(msg)
            return False, msg

        print(f"Sucesso: {len(all_documents)} documentos carregados.")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(all_documents)
        print(f"Sucesso: {len(chunks)} chunks criados.")

        print("Carregando modelo de embeddings (Google API)...")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            msg = "ERRO: GOOGLE_API_KEY não encontrada. Verifique as variáveis de ambiente."
            print(msg)
            return False, msg
            
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )

        print(f"Criando e persistindo banco de vetores em '{PERSIST_DIRECTORY}'...")
        
        # --- MUDANÇA PRINCIPAL AQUI (CORREÇÃO DO BUG 500) ---
        # 1. Crie um cliente Chroma vazio primeiro
        db = Chroma(
            embedding_function=embeddings_model,
            persist_directory=PERSIST_DIRECTORY
        )
        
        # 2. Adicione os documentos em lotes para evitar o timeout da API
        batch_size = 50 # Envia 50 chunks de cada vez
        total_batches = (len(chunks) // batch_size) + 1
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            print(f"Processando lote {i//batch_size + 1} de {total_batches}...")
            db.add_documents(documents=batch_chunks)
            time.sleep(1) # Pausa de 1 segundo para não sobrecarregar a API
        
        print("Persistindo o banco de dados no disco...")
        db.persist()
        
        # 3. Crie o "arquivo de sucesso"
        with open(SUCCESS_FLAG_FILE, "w") as f:
            f.write("ok")
        # --- FIM DA MUDANÇA ---
        
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
    import time
    success, message = run_ingestion()
    if success:
        print(f"✅ Ingestão concluída com sucesso! {message}")
    else:
        print(f"❌ Falha na ingestão: {message}")