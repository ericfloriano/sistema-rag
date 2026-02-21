import os
import logging
import asyncio
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from dotenv import load_dotenv
load_dotenv()

# --- Dummy Web Server para o Render (Evita erro de 'No open ports' em Web Services) ---
class DummyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot is healthy")

def run_dummy_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(('0.0.0.0', port), DummyHandler)
    server.serve_forever()

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from rag_core import get_rag_response_async, get_rag_chain

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Funções do Bot ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_name = update.effective_user.first_name
    welcome_text = (
        f"Olá, {user_name}! 👋 Seja muito bem-vindo(a)!\n\n"
        "Eu sou o <b>Assistente Virtual de Inteligência Artificial focado no ReCARE</b> 🤖.\n\n"
        "Estou aqui à sua inteira disposição para tirar todas as suas dúvidas sobre o equipamento ReCARE. "
        "Você pode me perguntar sobre especificações, indicações de uso, dados do manual ou suporte técnico.\n\n"
        "Basta digitar a sua pergunta abaixo! Exemplos:\n"
        "👉 <i>'Para quais perfis de pacientes o ReCARE é indicado?'</i>\n"
        "👉 <i>'Quantos canais o equipamento possui?'</i>\n\n"
        "<i>⚙️ Nota: Sou uma IA em constante aprendizado. Minhas respostas são baseadas estritamente e unicamente nos documentos "
        "oficiais da empresa para garantir máxima confiabilidade. Se eu não souber algo, te avisarei de forma honesta! 💬</i>"
    )
    await update.message.reply_html(welcome_text)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Como usar:\n"
        "1. Faça uma pergunta direta sobre um processo ou produto."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    question = update.message.text
    chat_id = update.effective_chat.id
    logger.info(f"Recebida pergunta do Chat ID {chat_id}: {question}")

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    try:
        chain = context.bot_data.get("chain")
        if chain is None:
            await update.message.reply_text("Desculpe, meu cérebro está offline no momento. O suporte técnico já foi notificado!")
            return
            
        answer = await get_rag_response_async(chain, question)
        await update.message.reply_text(answer)
        logger.info(f"Resposta enviada para o Chat ID {chat_id}: {answer[:50]}...")

    except Exception as e:
        logger.error(f"Erro ao processar mensagem: {e}", exc_info=True)
        await update.message.reply_text(
            "Desculpe, ocorreu um erro inesperado ao processar sua pergunta."
        )

# --- A função main() assíncrona ---
async def main() -> None:
    if not TELEGRAM_TOKEN:
        logger.error("Token do Telegram não encontrado. Verifique seu arquivo .env")
        return
    
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    logger.info("Tentando carregar a cadeia RAG para o Telegram...")
    try:
        chain = get_rag_chain()
        application.bot_data["chain"] = chain
        logger.info("Cadeia RAG anexada ao bot com sucesso.")
    except Exception as e:
        logger.error(f"Falha fatal ao carregar a cadeia RAG. O bot será iniciado, mas as respostas podem falhar: {e}", exc_info=True)
        application.bot_data["chain"] = None

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    try:
        logger.info("Inicializando o bot (async)...")
        
        # Iniciando o Dummy Server em uma thread separada para não bloquear o Asyncio do Telegram
        threading.Thread(target=run_dummy_server, daemon=True).start()

        await application.initialize()
        
        logger.info("Iniciando o polling (async)...")
        await application.start()
        await application.updater.start_polling()
        
        logger.info("Bot iniciado com sucesso. Pressione Ctrl+C para parar.")
        
        while True:
            await asyncio.sleep(3600)

    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot está parando...")
        await application.updater.stop()
        await application.stop()
        await application.shutdown()
        logger.info("Bot parado com sucesso.")
    except Exception as e:
        logger.error(f"Erro fatal ao executar o bot: {e}", exc_info=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "can't be re-entered" in str(e):
            logger.info("O bot já estava rodando ou o loop de eventos está ativo.")
        else:
            logger.error(f"Erro ao iniciar o loop asyncio: {e}")