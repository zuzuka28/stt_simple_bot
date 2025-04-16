import os
import logging
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
import torch
from audio_processor import AppConfig, TranscriptionPipeline

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class TelegramBot:
    def __init__(self):
        self.token = os.getenv("TELEGRAM_TOKEN")
        self.app = ApplicationBuilder().token(self.token).build()
        self.setup_handlers()

        self.pipeline = TranscriptionPipeline(
            AppConfig(
                audio_file="",
                whisper_model=os.getenv("WHISPER_MODEL", "turbo"),
                llm_model=os.getenv("LLM_MODEL", "ilyagusev/saiga_llama3"),
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        )

    def setup_handlers(self):
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("help", self.help))
        self.app.add_handler(
            MessageHandler(filters.AUDIO | filters.VOICE, self.handle_audio)
        )

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Send Voice message or Audio to process")

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Send Voice message or Audio to process")

    async def handle_audio(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.message.from_user
        logger.info(f"got request from {user.id}")

        try:
            await update.message.reply_text("⏳ Processing...")

            audio_file = await update.message.effective_attachment.get_file()
            file_path = f"temp/{audio_file.file_id}.ogg"
            await audio_file.download_to_drive(file_path)

            result_file = self.pipeline.run(file_path)

            with open(result_file, "r") as f:
                await update.message.reply_markdown(f.read())

            os.remove(file_path)

        except Exception as e:
            logger.error(f"Error: {e}")
            await update.message.reply_text("❌ Error!")

    def run(self):
        self.app.run_polling()


if __name__ == "__main__":
    TelegramBot().run()
