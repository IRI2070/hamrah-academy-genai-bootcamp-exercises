from telegram.ext import Application, CommandHandler, MessageHandler, filters
from src.pipeline import RAGPipeline
from src.config import config
from src.utils.logger import logger

pipeline = RAGPipeline()


async def start(update, context):
    chat_id = str(update.effective_chat.id)
    pipeline.reset_session(chat_id)
    await update.message.reply_text("سلام! یک سؤال دارویی بپرس.")


async def answer(update, context):
    query = update.message.text
    chat_id = str(update.effective_chat.id)
    logger.info(f"[chat={chat_id}] User asked: {query}")
    await update.message.reply_text("در حال تولید پاسخ ...")

    try:
        result = pipeline.ask(query, session_id=chat_id)
        await update.message.reply_text(result)
    except Exception as e:
        logger.error(f"Error: {e}")
        await update.message.reply_text(f"خطا: {e}")


def run_bot():
    app = Application.builder().token(config.BOT_TOKEN).base_url(config.BOT_BASE_URL).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, answer))

    app.run_polling()


if __name__ == "__main__":
    run_bot()
