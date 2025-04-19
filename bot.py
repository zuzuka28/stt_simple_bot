import os
import logging
import asyncio
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
                whisper_model=os.getenv("WHISPER_MODEL", "large"),
                llm_model=os.getenv("LLM_MODEL", "ilyagusev/saiga_llama3"),
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        )

    def setup_handlers(self):
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("help", self.help))
        self.app.add_handler(
            MessageHandler(filters.AUDIO | filters.VOICE, self._handle_audio)
        )

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Send Voice message or Audio to process")

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Send Voice message or Audio to process")

    async def _stream_text_response(
        self, update: Update, file_path: str, chunk_size: int = 4096, delay: float = 0.3
    ) -> None:
        with open(file_path, "r", encoding="utf-8") as f:
            current_chunk = ""

            for line in f:
                if len(current_chunk.strip()) + len(line) > chunk_size:
                    await update.message.reply_text(current_chunk.strip())
                    await asyncio.sleep(delay)
                    current_chunk = ""

                current_chunk += line

            if current_chunk.strip():
                await update.message.reply_text(current_chunk.strip())

    async def _handle_audio(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.message.from_user
        logger.info(f"Got audio processing request from {user.id}")

        try:
            await update.message.reply_text("⏳ Processing your audio...")

            os.makedirs("temp", exist_ok=True)

            audio_file = await update.message.effective_attachment.get_file()
            file_path = f"temp/{audio_file.file_id}.ogg"
            await audio_file.download_to_drive(file_path)

            result_file = self.pipeline.run(file_path)

            await self._stream_text_response(update, result_file)

        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
            await update.message.reply_text("❌ Sorry, I couldn't process your audio.")

        finally:
            if "file_path" in locals() and os.path.exists(file_path):
                os.remove(file_path)
            if "result_file" in locals() and os.path.exists(result_file):
                os.remove(result_file)

    def run(self):
        self.app.run_polling()


if __name__ == "__main__":
    TelegramBot().run()
