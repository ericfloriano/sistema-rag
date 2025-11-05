import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Caminho para os documentos e para o banco de vetores
SOURCE_DIRECTORY = "documentos_fonte"
PERSIST_DIRECTORY = "chroma_db"

print("Iniciando processo de ingestão de dados...")

# Carrega os documentos da pasta
# O novo DirectoryLoader detecta os tipos de arquivo automaticamente
loader = DirectoryLoader(
    SOURCE_DIRECTORY,
    glob="**/*",  # Pega todos os arquivos em todas as subpastas
    recursive=True,
    show_progress=True,
    use_multithreading=True,
)

documentos = loader.load()

if not documentos:
    print("-------------------------------------------------------------------")
    print(f"Nenhum documento encontrado na pasta '{SOURCE_DIRECTORY}'.")
    print("Por favor, adicione seus arquivos (.pdf, .docx, .txt) e tente novamente.")
    print("-------------------------------------------------------------------")
    exit()

print(f"Total de {len(documentos)} documentos carregados.")

# 2. Chunking (Divisão dos Documentos)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Tamanho de cada chunk
    chunk_overlap=200   # Sobreposição entre chunks
)
chunks = text_splitter.split_documents(documentos)

print(f"Documentos divididos em {len(chunks)} chunks.")

# 3. Embeddings (Vetorização)
print("Carregando modelo de embeddings (all-MiniLM-L6-v2)... Isso pode demorar um pouco na primeira vez.")
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},  # Use 'cuda' se tiver GPU
)

# 4. Vector Store (Criação do Banco de Vetores)
print(f"Criando e persistindo banco de vetores em '{PERSIST_DIRECTORY}'...")
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    persist_directory=PERSIST_DIRECTORY,
)

print("-------------------------------------------------------------------")
print("✅ Ingestão concluída com sucesso!")
print(f"O banco de vetores foi salvo em '{PERSIST_DIRECTORY}'.")
print("-------------------------------------------------------------------")