import os
import warnings
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI  # <-- IMPORTADO
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- CONFIGURAÇÃO PRINCIPAL DO LLM ---
# Mude esta variável para "openai" se quiser usar o ChatGPT (pago)
# Deixe como "gemini" para usar o Gemini (gratuito)
LLM_PROVIDER = "gemini" 
# ------------------------------------

# Ignorar avisos
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
load_dotenv()

# --- Configuração Global ---
PERSIST_DIRECTORY = "chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Template do Prompt (Guardrail)
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
    """
    Carrega o modelo LLM com base no provedor escolhido.
    """
    if provider == "gemini":
        print("Carregando LLM: Google Gemini 2.0 Flash (Gratuito)")
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    elif provider == "openai":
        print("Carregando LLM: OpenAI GPT-4o (Pago)")
        return ChatOpenAI(
            model="gpt-4o",  # Ou "gpt-3.5-turbo" para mais barato
            temperature=0.0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    else:
        raise ValueError(f"Provedor de LLM desconhecido: {provider}. Escolha 'gemini' ou 'openai'.")

def get_rag_chain():
    """
    Monta e retorna a cadeia RAG (RAG Chain) pronta para ser usada.
    """
    try:
        # 1. Carregar o LLM (baseado na escolha)
        llm = get_llm(LLM_PROVIDER)

        # 2. Carregar o modelo de Embeddings (Local)
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
        )

        # 3. Carregar o Vector Store (ChromaDB)
        vector_store = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
        )

        # 4. Criar o Retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})

        # 5. Criar o Prompt Template
        rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

        # 6. Criar a Cadeia RAG (RAG Chain)
        
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

# Instância global da cadeia
try:
    chain = get_rag_chain()
    # Pega o nome do modelo real para exibir na interface
    MODEL_NAME_FOR_DISPLAY = chain.middle[1].model if chain else "Não carregado"
except Exception as e:
    print(f"Falha fatal ao inicializar o rag_core: {e}")
    chain = None
    MODEL_NAME_FOR_DISPLAY = "Erro ao carregar"


def get_rag_response(question_text: str) -> str:
    """
    Função de alto nível para obter a resposta do RAG.
    """
    if chain is None:
        return "Erro: A cadeia RAG não foi inicializada corretamente."
    if not question_text:
        return "Por favor, faça uma pergunta."
    try:
        response = chain.invoke(question_text)
        return response
    except Exception as e:
        # Erro comum é a API Key faltando
        if "api key" in str(e).lower():
             return f"Erro de API: Verifique se sua {LLM_PROVIDER.upper()}_API_KEY está correta no arquivo .env"
        print(f"Erro durante a invocação da cadeia: {e}")
        return f"Ocorreu um erro ao processar sua pergunta: {e}"

# Teste
if __name__ == "__main__":
    print("Testando o rag_core.py...")
    if chain:
        pergunta_teste = "O que é o ReCARE?"
        print(f"Pergunta: {pergunta_teste}")
        resposta = get_rag_response(pergunta_teste)
        print(f"Resposta: {resposta}")
    else:
        print("Não foi possível executar o teste, cadeia RAG falhou ao carregar.")

async def get_rag_response_async(question_text: str) -> str:
    """
    Função ASSÍNCRONA para obter a resposta do RAG.
    Usa ainvoke() para não bloquear o event loop (ex: do Telegram).
    """
    if chain is None:
        return "Erro: A cadeia RAG não foi inicializada corretamente."
        
    if not question_text:
        return "Por favor, faça uma pergunta."

    try:
        # Usa ainvoke (async invoke) em vez de invoke
        response = await chain.ainvoke(question_text)
        return response
    except Exception as e:
        print(f"Erro durante a invocação da cadeia (async): {e}")
        return f"Ocorreu um erro ao processar sua pergunta: {e}"