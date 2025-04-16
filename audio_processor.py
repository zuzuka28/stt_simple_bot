import os
import logging
import time
import torch
from dataclasses import dataclass
from typing import List
from pydub import AudioSegment
import whisper
from wtpsplit import SaT
import ollama


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    whisper_model: str = "turbo"
    sat_model: str = "sat-3l-sm"
    llm_model: str = "ilyagusev/saiga_llama3"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_cache_dir: str = "models_cache"
    temp_dir: str = "temp"


class AudioProcessor:
    def __init__(self, config: AppConfig):
        self.config = config
        os.makedirs(config.temp_dir, exist_ok=True)

    def convert_to_wav(self, path: str) -> str:
        if path.endswith(".wav"):
            return path

        output_path = os.path.join(
            self.config.temp_dir,
            f"{os.path.splitext(os.path.basename(path))[0]}.wav",
        )

        logger.info("Convert to WAV...")
        audio = AudioSegment.from_file(path)

        audio.export(output_path, format="wav")

        return output_path


class WhisperTranscriber:
    def __init__(self, config: AppConfig):
        self.config = config
        self.model = self._load_model()

    def _load_model(self):
        logger.info(f"Load Whisper ({self.config.whisper_model})...")
        return whisper.load_model(
            self.config.whisper_model,
            download_root=os.path.join(self.config.model_cache_dir, "whisper"),
            device=self.config.device,
        )

    def transcribe(self, audio_path: str) -> str:
        logger.info("Transcribe...")
        result = self.model.transcribe(audio_path, language="ru", verbose=False)
        return result["text"]


class TextSegmenter:
    def __init__(self, config: AppConfig):
        self.config = config
        self.model = self._load_model()

    def _load_model(self):
        """Загружает модель WTPSplit"""
        logger.info(f"Load SoT ({self.config.sat_model})...")
        sat = SaT(self.config.sat_model)
        sat.half().to(self.config.device)
        return sat

    def segment(self, text: str) -> List[str]:
        logger.info("Segmentation...")
        segments = self.model.split(text)
        return [s.strip() for s in segments if s.strip()]


class TextPostProcessor:
    def __init__(self, config: AppConfig):
        self.config = config
        self.ollama_model = self.config.llm_model.replace("ollama/", "")
        logger.info(f"Init TextPostProcessor with model: {self.ollama_model}")

    def _find_topic_shifts(self, sentences: List[str]) -> List[int]:
        """Определяет индексы предложений с резкой сменой темы"""
        system_prompt = """Ты — TopicShiftAnalyzer.
        Проанализируй последовательность пронумерованных предложений и определи номера предложений, где происходит РЕЗКАЯ смена темы.
        Ответь ТОЛЬКО числами через запятую (например: "3,7"). Если смен нет — верни "0"."""

        numbered_sentences = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, 1))

        response = ollama.chat(
            model=self.ollama_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": numbered_sentences},
            ],
            options={"temperature": 0},
        )

        shifts = [
            int(x.strip())
            for x in response["message"]["content"].split(",")
            if x.strip().isdigit()
        ]

        return [s - 1 for s in shifts if 0 < s <= len(sentences)]

    def _group_sentences(
        self, sentences: List[str], breaks: List[int]
    ) -> List[List[str]]:
        if not breaks:
            return [sentences]

        all_breaks = sorted({0, *breaks, len(sentences)})

        return [
            sentences[start:end] for start, end in zip(all_breaks[:-1], all_breaks[1:])
        ]

    def _format_markdown(self, sentences: List[str]) -> str:
        topic_shifts = self._find_topic_shifts(sentences)

        groups = self._group_sentences(sentences, topic_shifts)

        return "\n\n".join([" ".join(group) for group in groups])

    def _wrap_source_text(self, text: str) -> str:
        return "\n\n".join(
            [
                "<!--",
                "### SOURCE START ###",
                text,
                "### SOURCE END ###-->",
            ]
        )

    def process(self, segments: List[str]) -> str:
        source = self._wrap_source_text("\n\n".join(segments))

        logger.info("Markdown formating...")

        structured = self._format_markdown(segments)

        return "\n\n".join([structured, source])


class TranscriptionPipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        os.makedirs(config.model_cache_dir, exist_ok=True)
        self.audio_processor = AudioProcessor(config)
        self.transcriber = WhisperTranscriber(config)
        self.segmenter = TextSegmenter(config)
        self.post_processor = TextPostProcessor(config)

    def run(self, path: str) -> str:
        start_time = time.time()

        wav_path = self.audio_processor.convert_to_wav(path)
        raw_text = self.transcriber.transcribe(wav_path)
        segments = self.segmenter.segment(raw_text)
        processed = self.post_processor.process(segments)

        output_path = self._save_result(path, processed)

        duration = time.time() - start_time
        logger.info(f"Processing Duration: {duration:.2f}s.")

        return output_path

    def _save_result(self, path: str, text: str) -> str:
        os.makedirs("output", exist_ok=True)

        output_file = os.path.join("output", f"transcript_{path}_{int(time.time())}.md")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)

        logger.info(f"File saved: {output_file}")

        return output_file
