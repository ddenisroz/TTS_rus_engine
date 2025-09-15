#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Правильная реализация F5-TTS с русским языком на основе рабочего кода
"""

import logging
import os
from pathlib import Path
from typing import Optional
import tempfile

import numpy as np
import soundfile as sf
import torch
from f5_tts.api import F5TTS
from huggingface_hub import hf_hub_download

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Константы для F5-TTS Russian Model
F5_TTS_MODEL = "Misha24-10/F5-TTS_RUSSIAN"
F5_TTS_SUBFOLDER_BASE = "F5TTS_v1_Base_v2"
F5_TTS_SUBFOLDER_ACCENT = "F5TTS_v1_Base_accent_tune"
RUSSIAN_MODEL_CHECKPOINT = "model_last_inference.safetensors"

# Транскрипция для референсного голоса
DEFAULT_VOICE_TRANSCRIPTION = "Секреты всегда рядом, Скуф. Нужно лишь тихо прислушаться и услышать их."


class CorrectTTS:
    def __init__(self, model_type="base"):
        logger.info(f"Инициализация F5-TTS Service с моделью: {model_type}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = model_type
        logger.info(f"F5-TTS использует устройство: {self.device}")

        # F5-TTS модель
        self.f5tts = None
        self._load_models()

    def _load_models(self):
        """Загружает F5-TTS русскую модель используя официальный API."""
        try:
            logger.info(f"Загружаем F5-TTS русскую модель ({self.model_type})...")
            
            cache_dir = Path("f5_tts_cache")
            cache_dir.mkdir(exist_ok=True)

            # Выбираем подпапку в зависимости от типа модели
            if self.model_type == "accent":
                subfolder = F5_TTS_SUBFOLDER_ACCENT
                logger.info("Используем модель с accent tuning")
            else:
                subfolder = F5_TTS_SUBFOLDER_BASE
                logger.info("Используем базовую модель")

            # Скачиваем русскую модель checkpoint
            logger.info(f"Скачиваем русский checkpoint из {F5_TTS_MODEL}...")
            russian_ckpt_path = hf_hub_download(
                repo_id=F5_TTS_MODEL,
                filename=f"{subfolder}/{RUSSIAN_MODEL_CHECKPOINT}",
                cache_dir=cache_dir
            )
            logger.info(f"Русский checkpoint скачан в: {russian_ckpt_path}")

            # Скачиваем vocab.txt
            vocab_path = hf_hub_download(
                repo_id=F5_TTS_MODEL,
                filename="F5TTS_v1_Base/vocab.txt",
                cache_dir=cache_dir
            )
            logger.info(f"vocab.txt скачан в: {vocab_path}")

            self.f5tts = F5TTS(
                model="F5TTS_v1_Base",
                ckpt_file=russian_ckpt_path,  # Передаем локальный путь к скачанному checkpoint
                vocab_file=vocab_path,  # Используем скачанный vocab.txt
                ode_method="euler",
                use_ema=True,
                device=self.device,
                hf_cache_dir=str(cache_dir)
            )
            
            logger.info(f"F5-TTS русская модель ({self.model_type}) загружена успешно.")

        except Exception as e:
            logger.error(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить F5-TTS модель. Ошибка: {e}", exc_info=True)
            self.f5tts = None

    def preprocess_text_for_tts(self, text: str) -> str:
        """Предобработка текста для TTS (упрощенная версия)."""
        processed_text = text.strip()
        if not processed_text:
            return ""
        
        logger.info(f"Исходный текст: '{processed_text}'")
        
        # Простая предобработка
        # Убираем лишние пробелы
        processed_text = ' '.join(processed_text.split())
        
        # Добавляем точку в конце если нет знаков препинания
        if processed_text and not processed_text.endswith(('.', '!', '?')):
            processed_text += '.'
        
        logger.info(f"Обработанный текст: '{processed_text}'")
        
        return processed_text

    def synthesize_speech(self, text: str, ref_audio_path: str, ref_text: str = "") -> Optional[str]:
        """Синтезирует речь используя F5-TTS русскую модель."""
        if not self.f5tts:
            logger.error("TTS Service не готов. F5-TTS модель не загружена.")
            return None
            
        if not os.path.exists(ref_audio_path):
            logger.error(f"Референсное аудио не найдено: {ref_audio_path}")
            return None

        # Предобработка текста
        processed_text = self.preprocess_text_for_tts(text)
        if not processed_text:
            logger.warning("Текст пустой после предобработки")
            return None

        # Используем предопределенную транскрипцию если ref_text пустой
        ref_text_to_use = ref_text if ref_text else DEFAULT_VOICE_TRANSCRIPTION

        logger.info(f"Синтезируем аудио для: '{processed_text}' используя голос '{ref_audio_path}'")

        try:
            # Создаем выходную директорию
            output_dir = Path("audio_output")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Создаем уникальное имя файла
            output_filename = f"output_{hash(text)}.wav"
            output_path = output_dir / output_filename

            # Используем F5-TTS для синтеза речи
            wav, sr, spect = self.f5tts.infer(
                ref_file=ref_audio_path,
                ref_text=ref_text_to_use,
                gen_text=processed_text,
                cross_fade_duration=0.15,
                speed=1.0
            )

            # Добавляем небольшое количество тишины в конец аудио
            silence_duration_ms = 200
            silence_samples = int(sr * (silence_duration_ms / 1000.0))
            silence = np.zeros(silence_samples, dtype=np.float32)
            wav_padded = np.concatenate([wav, silence])

            # Сохраняем сгенерированное аудио
            sf.write(str(output_path), wav_padded, sr)
            
            logger.info(f"Аудио синтезировано и сохранено в {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Ошибка во время F5-TTS синтеза: {e}", exc_info=True)
            return None


def main():
    """Тестовая функция"""
    try:
        tts = CorrectTTS()
        
        if not tts.f5tts:
            logger.error("Не удалось загрузить F5-TTS модель")
            return
        
        # Тестовые данные
        test_text = "Привет! Это тестовое сообщение на русском языке."
        ref_audio = "test_ref.wav"  # Нужен референсный аудиофайл
        ref_text = "Создавая уникальные цифровые объекты, вы размышляете о том, насколько интересны вашей дей миру."
        
        if os.path.exists(ref_audio):
            output = tts.synthesize_speech(test_text, ref_audio, ref_text)
            if output:
                print(f"✅ Результат сохранен: {output}")
            else:
                print("❌ Ошибка синтеза")
        else:
            print(f"❌ Референсный аудиофайл не найден: {ref_audio}")
            print("Создайте референсный аудиофайл для тестирования")
            
    except Exception as e:
        logger.error(f"Ошибка: {e}")


if __name__ == "__main__":
    main()
