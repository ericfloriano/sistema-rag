import os
import warnings
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
# --- MUDANÇA AQUI ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings # Importa o modelo do Google
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

LLM_PROVIDER = "gemini" 
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
load_dotenv()

# Caminho do disco persistente do Render
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
            model="gemini-2.0-flash", # Mantendo o seu modelo que funcionou
            temperature=0.0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
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
    try:
        llm = get_llm(LLM_PROVIDER)

        # --- MUDANÇA PRINCIPAL AQUI ---
        print("Carregando modelo de embeddings (Google API)...")
        if not os.getenv("GOOGLE_API_KEY"):
             raise ValueError("GOOGLE_API_KEY não encontrada.")

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        # --- FIM DA MUDANÇA ---

        vector_store = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
        )

        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
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

try:
    chain = get_rag_chain()
    MODEL_NAME_FOR_DISPLAY = (
        chain.middle[1].model if chain else "Não carregado"
    )
except Exception as e:
    print(f"Falha fatal ao inicializar o rag_core: {e}")
    chain = None
    MODEL_NAME_FOR_DISPLAY = "Erro ao carregar"


def get_rag_response(question_text: str) -> str:
    if chain is None:
        return "Erro: A cadeia RAG não foi inicializada corretamente."
    if not question_text:
        return "Por favor, faça uma pergunta."
    try:
        response = chain.invoke(question_text)
        return response
    except Exception as e:
        if "api key" in str(e).lower():
             return f"Erro de API: Verifique se sua {LLM_PROVIDER.upper()}_API_KEY está correta no arquivo .env"
        print(f"Erro durante a invocação da cadeia: {e}")
        return f"Ocorreu um erro ao processar sua pergunta: {e}"

async def get_rag_response_async(question_text: str) -> str:
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

if __name__ == "__main__":
    print("Testando o rag_core.py...")
    if chain:
        pergunta_teste = "O que é o ReCARE?"
        print(f"Pergunta: {pergunta_teste}")
        resposta = get_rag_response(pergunta_teste)
        print(f"Resposta: {resposta}")
    else:
        print("Não foi possível executar o teste, cadeia RAG falhou ao carregar.")