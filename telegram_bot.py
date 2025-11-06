import os
import logging
import asyncio
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
# --- MUDANÇA: Importações diferentes ---
from rag_core import get_rag_response_async, get_rag_chain

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- MUDANÇA: Carrega a 'chain' globalmente para o bot ---
# Isso é seguro, pois o Streamlit já construiu o DB
try:
    logger.info("Carregando a cadeia RAG para o Telegram...")
    chain = get_rag_chain()
    if chain is None:
        logger.error("Falha fatal ao carregar a cadeia RAG para o Telegram.")
except Exception as e:
    logger.error(f"Erro ao carregar 'chain' do Telegram: {e}")
    chain = None

# --- Funções do Bot ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_name = update.effective_user.first_name
    await update.message.reply_html(
        f"Olá, {user_name}! 👋\n\n"
        "Eu sou seu assistente."
        "Basta me enviar sua pergunta!"
    )

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
        # --- MUDANÇA: Passa a 'chain' como argumento ---
        answer = await get_rag_response_async(chain, question)
        
        await update.message.reply_text(answer)
        logger.info(f"Resposta enviada para o Chat ID {chat_id}: {answer[:50]}...")

    except Exception as e:
        logger.error(f"Erro ao processar mensagem: {e}", exc_info=True)
        await update.message.reply_text(
            "Desculpe, ocorreu um erro inesperado ao processar sua pergunta."
        )

# --- A função main() assíncrona permanece a mesma ---
async def main() -> None:
    if not TELEGRAM_TOKEN:
        logger.error("Token do Telegram não encontrado. Verifique seu arquivo .env")
        return
    
    if chain is None:
        logger.error("A cadeia RAG não foi carregada. O bot não pode iniciar.")
        return

    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    try:
        logger.info("Inicializando o bot (async)...")
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