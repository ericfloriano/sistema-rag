import os
import logging
import asyncio # Precisamos disto agora
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
# Importamos a nossa nova função async
from rag_core import get_rag_response_async 

# Carrega o token do .env
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Configura o logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Funções do Bot (já são async, não precisam mudar) ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Envia uma mensagem de boas-vindas."""
    user_name = update.effective_user.first_name
    await update.message.reply_html(
        f"Olá, {user_name}! 👋\n\n"
        "Eu sou seu assistente de processos internos. "
        "Basta me enviar sua pergunta!"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Envia uma mensagem de ajuda."""
    await update.message.reply_text(
        "Como usar:\n"
        "1. Faça uma pergunta direta sobre um processo ou produto.\n"
        "Eu vou pesquisar nos documentos internos para encontrar a melhor resposta."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Responde a qualquer mensagem de texto."""
    question = update.message.text
    chat_id = update.effective_chat.id
    logger.info(f"Recebida pergunta do Chat ID {chat_id}: {question}")

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    try:
        # Chama a versão ASSÍNCRONA do RAG
        answer = await get_rag_response_async(question)
        
        await update.message.reply_text(answer)
        logger.info(f"Resposta enviada para o Chat ID {chat_id}: {answer[:50]}...")

    except Exception as e:
        logger.error(f"Erro ao processar mensagem: {e}", exc_info=True)
        await update.message.reply_text(
            "Desculpe, ocorreu um erro inesperado ao processar sua pergunta."
        )

# --- A GRANDE MUDANÇA ESTÁ AQUI ---

async def main() -> None:
    """
    Configura e executa o bot de forma 100% assíncrona.
    """
    if not TELEGRAM_TOKEN:
        logger.error("Token do Telegram não encontrado. Verifique seu arquivo .env")
        return

    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Registra os handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    # Executa o bot de forma assíncrona
    try:
        logger.info("Inicializando o bot (async)...")
        await application.initialize()
        
        logger.info("Iniciando o polling (async)...")
        await application.start()
        await application.updater.start_polling()
        
        logger.info("Bot iniciado com sucesso. Pressione Ctrl+C para parar.")
        
        # Mantém o script vivo (executando)
        while True:
            await asyncio.sleep(3600) # Dorme por uma hora, indefinidamente

    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot está parando...")
        await application.updater.stop()
        await application.stop()
        await application.shutdown()
        logger.info("Bot parado com sucesso.")
    except Exception as e:
        logger.error(f"Erro fatal ao executar o bot: {e}", exc_info=True)

if __name__ == "__main__":
    # Esta é a nova forma de executar a função main()
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "can't be re-entered" in str(e):
            logger.info("O bot já estava rodando ou o loop de eventos está ativo.")
        else:
            logger.error(f"Erro ao iniciar o loop asyncio: {e}")