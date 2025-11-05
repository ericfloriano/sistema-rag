import os
import warnings
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

LLM_PROVIDER = "gemini" 
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
load_dotenv()

PERSIST_DIRECTORY = "/var/data/chroma_db"

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
    # (O código do OpenAI permanece o mesmo...)
    elif provider == "openai":
        print("Carregando LLM: OpenAI GPT-4o (Pago)")
        return ChatOpenAI(
            model="gpt-4o",
            temperature=0.0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    else:
        raise ValueError(f"Provedor de LLM desconhecido: {provider}. Escolha 'gemini' ou 'openai'.")

def get_rag_chain():
    """
    Função que CONSTRÓI e RETORNA a cadeia RAG.
    Ela não é mais chamada globalmente.
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

        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6, "fetch_k": 50}
        )
        rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

        def format_docs(docs):
            if not docs:
                return "Nenhum contexto relevante encontrado."
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
        )
        
        print(f"Cadeia RAG montada com sucesso usando: {LLM_PROVIDER}")
        return rag_chain

    except Exception as e:
        print(f"Erro ao montar a cadeia RAG: {e}")
        return None

# --- REMOVEMOS O CARREGAMENTO GLOBAL DA 'chain' ---

# --- MUDANÇA: Funções agora recebem a 'chain' ---

def get_rag_response(chain, question_text: str) -> str:
    """Função SÍNCRONA para obter a resposta (usada pelo Streamlit)."""
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
    """Função ASSÍNCRONA para obter a resposta (usada pelo Telegram)."""
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