import os
import warnings
import pickle
from dotenv import load_dotenv
from typing import List

# --- IMPORTAÇÕES PRINCIPAIS ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# --- IMPORTAÇÕES PARA NOSSA LÓGICA HÍBRIDA MANUAL ---
from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import RunnableLambda # <-- Chave para nossa solução

LLM_PROVIDER = "gemini" 
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
load_dotenv()

PERSIST_DIRECTORY = "chroma_db"

RAG_PROMPT_TEMPLATE = """
ATENÇÃO: Você é um assistente de IA focado em responder perguntas sobre processos internos e produtos da empresa.
CONTEXTO:
{context}
PERGUNTA:
{question}
INSTRUÇÕES:
1. Responda à PERGUNTA **estritamente** com base no CONTEXTO fornecido.
2. Se o CONTEXTO não contiver a resposta, diga **exatamente**: "Desculpe, não tenho informações sobre isso no meu banco de dados."
3. Não invente informações, não faça suposições e não use conhecimento externo.
4. Responda em português brasileiro, de forma clara e objetiva.
"""

def get_llm(provider: str):
    if provider == "gemini":
        print("Carregando LLM: Google Gemini 2.0 Flash (Gratuito)")
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    # ... (outro código de LLM)
    else:
        raise ValueError(f"Provedor de LLM desconhecido: {provider}.")

def get_rag_chain():
    """
    Função que CONSTRÓI e RETORNA a cadeia RAG HÍBRIDA MANUAL.
    """
    try:
        llm = get_llm(LLM_PROVIDER)

        print("Carregando modelo de embeddings (Google API)...")
        if not os.getenv("GOOGLE_API_KEY"):
             raise ValueError("GOOGLE_API_KEY não encontrada.")

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        print(f"Carregando banco de vetores de: {PERSIST_DIRECTORY}")
        vector_store = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
        )

        # --- ARQUITETURA DE BUSCA HÍBRIDA (MANUAL) ---
        print("Configurando o Retriever Híbrido (Vetorial + Keyword)...")
        
        # 1. O Retriever Vetorial (Carrega do ChromaDB)
        vector_retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 7, "fetch_k": 20})

        # 2. O Retriever de Palavra-Chave (Carrega do arquivo .pkl)
        bm25_path = os.path.join(PERSIST_DIRECTORY, "bm25_retriever.pkl")
        print(f"Carregando BM25 de: {bm25_path}")
        
        if not os.path.exists(bm25_path):
            print(f"ERRO: Arquivo {bm25_path} não encontrado.")
            return None
            
        with open(bm25_path, "rb") as f:
            bm25_retriever = pickle.load(f)
        bm25_retriever.k = 5

        # 3. NOSSA LÓGICA DE "ENSEMBLE" MANUAL
        def hybrid_retrieve(query: str) -> List[Document]:
            """
            Combina os resultados do BM25 e da busca vetorial,
            e remove documentos duplicados.
            """
            print(f"Buscando (Híbrido): {query}")
            # --- CORREÇÃO: Usando .invoke() ---
            bm25_docs = bm25_retriever.invoke(query)
            vector_docs = vector_retriever.invoke(query)
            # --- FIM DA CORREÇÃO ---

            # Combina e deduplica
            all_docs = {} # Usar um dict para deduplicação baseada no conteúdo
            for doc in bm25_docs + vector_docs:
                all_docs[doc.page_content] = doc
            
            return list(all_docs.values())
        
        # --- FIM DA NOVA ARQUITETURA ---

        rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

        def format_docs(docs):
            if not docs:
                return "Nenhum contexto relevante encontrado."
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            # Usamos RunnableLambda para inserir nossa função na cadeia
            {"context": RunnableLambda(hybrid_retrieve) | format_docs, "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
        )
        
        print(f"Cadeia RAG HÍBRIDA (Manual) montada com sucesso usando: {LLM_PROVIDER}")
        return rag_chain

    except Exception as e:
        print(f"Erro ao montar a cadeia RAG: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- Funções de resposta (sem alteração) ---

def get_rag_response(chain, question_text: str) -> str:
    if chain is None:
        return "Erro: A cadeia RAG não foi inicializada corretamente."
    if not question_text:
        return "Por favor, faça uma pergunta."
    try:
        response = chain.invoke(question_text)
        return response
    except Exception as e:
        print(f"Erro durante a invocação da cadeia: {e}")
        return f"Ocorreu um erro ao processar sua pergunta: {e}"

async def get_rag_response_async(chain, question_text: str) -> str:
    if chain is None:
        return "Erro: A cadeia RAG não foi inicializada corretamente."
    if not question_text:
        return "Por favor, faça uma pergunta."
    try:
        response = await chain.ainvoke(question_text)
        return response
    except Exception as e:
        print(f"Erro durante a invocação da cadeia (async): {e}")
        return f"Ocorreu um erro ao processar sua pergunta: {e}"