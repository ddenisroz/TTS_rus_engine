#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Веб-интерфейс для мультиязычной F5-TTS системы
"""

import logging
import os
import tempfile
from pathlib import Path
import gradio as gr
import whisper

from russian_tts import RussianTTS

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Настройка для PyTorch 2.4.0
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


class RussianTTSWebInterface:
    def __init__(self):
        logger.info("Запуск русского F5-TTS веб-интерфейса...")
        
        # Инициализируем TTS с ударениями (turbo модель по умолчанию)
        self.tts = RussianTTS(enable_accent=True, accent_model_size="turbo")
        
        # Загружаем Whisper для транскрипции
        self.whisper_model = None
        self.load_whisper_model()

    def load_whisper_model(self):
        """Загружает модель Whisper для транскрипции."""
        try:
            logger.info("Загружаем модель Whisper...")
            self.whisper_model = whisper.load_model("base", device="cpu")
            logger.info("Whisper модель загружена успешно")
        except Exception as e:
            logger.error(f"Ошибка загрузки Whisper: {e}")
            self.whisper_model = None

    def transcribe_reference_audio(self, audio_file):
        """Транскрибирует референсное аудио в текст."""
        if not audio_file:
            return ""
        
        if not self.whisper_model:
            return "Whisper модель не загружена"
        
        try:
            logger.info(f"Начинаю транскрипцию: {audio_file}")
            
            # Транскрибируем аудио
            result = self.whisper_model.transcribe(audio_file)
            transcribed_text = result["text"].strip()
            
            logger.info(f"Транскрипция завершена: '{transcribed_text[:100]}...'")
            return transcribed_text
            
        except Exception as e:
            logger.error(f"Ошибка при транскрипции: {e}")
            return f"Ошибка транскрипции: {e}"

    def synthesize_speech(self, text, ref_audio, ref_text):
        """Синтезирует речь с автоматическими настройками."""
        if not text.strip():
            return None, "Введите текст для синтеза"
        
        if not ref_audio:
            return None, "Загрузите референсное аудио"
        
        try:
            # Синтезируем речь с автоматическими параметрами
            output_path = self.tts.synthesize_speech(
                text=text,
                ref_audio_path=ref_audio,
                ref_text=ref_text
            )
            
            if output_path and Path(output_path).exists():
                return output_path, "✅ Синтез завершен успешно!"
            else:
                return None, "❌ Ошибка синтеза"
                
        except Exception as e:
            logger.error(f"Ошибка в synthesize_speech: {e}")
            return None, f"❌ Ошибка: {e}"


    def create_interface(self):
        """Создает веб-интерфейс"""
        with gr.Blocks(title="Multilingual F5-TTS", theme=gr.themes.Soft()) as interface:
            gr.Markdown("""
            # 🎤 Multilingual F5-TTS Voice Cloning
            
            **Возможности:**
            - 🗣️ Клонирование голоса с референсного аудио
            - 🇷🇺 Автоматическое определение языка (русский/английский)
            - 🌍 Мультиязычная поддержка
            - 📝 Транскрипция аудио в текст
            - 🎵 Высококачественный синтез речи
            
            **Инструкция:**
            1. Загрузите референсное аудио для клонирования голоса
            2. Введите текст референсного аудио (что говорится в аудио)
            3. Введите текст для синтеза (русский или английский)
            4. Настройте параметры синтеза при необходимости
            5. Нажмите "Синтезировать речь"
            """)
            
            with gr.Row():
                with gr.Column():
                    # Референсное аудио
                    ref_audio = gr.Audio(
                        label="🎵 Референсное аудио",
                        type="filepath"
                    )
                    
                    # Кнопка транскрипции
                    transcribe_btn = gr.Button("🎤 Расшифровать аудио", variant="secondary")
                    
                    # Текст референсного аудио
                    ref_text = gr.Textbox(
                        label="📄 Текст референсного аудио",
                        placeholder="Введите что говорится в референсном аудио...",
                        lines=2
                    )
                    
                    # Текст для синтеза
                    gen_text = gr.Textbox(
                        label="📝 Текст для синтеза",
                        placeholder="Введите текст который нужно озвучить...",
                        lines=3
                    )
                    
                    # Кнопка синтеза
                    synthesize_btn = gr.Button("🎵 Синтезировать речь", variant="primary")
                
                with gr.Column():
                    # Результат
                    output_audio = gr.Audio(
                        label="🎧 Результат",
                        type="filepath"
                    )
                    
                    # Статус
                    status = gr.Textbox(
                        label="📊 Статус",
                        value="Готов к работе",
                        interactive=False
                    )
            
            # Автоматические настройки (все параметры определяются автоматически)
            gr.Markdown("""
            🤖 **Автоматические настройки:**
            - Скорость: ≤3 символов = 0.1, 4-8 = 0.3, 9-16 = 0.6, 17-30 = 0.9, >30 = 1.0
            - NFE Steps: 26 для коротких текстов, 18 для длинных (>120 символов)
            - RUAccent: Turbo модель с tiny режимом для русского текста
            - Тишина в конце: 100мс
            - Все остальные параметры оптимизированы для лучшего качества
            """)
            
            # Обработчики событий
            transcribe_btn.click(
                fn=self.transcribe_reference_audio,
                inputs=[ref_audio],
                outputs=[ref_text]
            )
            
            synthesize_btn.click(
                fn=self.synthesize_speech,
                inputs=[gen_text, ref_audio, ref_text],
                outputs=[output_audio, status]
            )
        
        return interface

    def launch(self):
        """Запускает веб-интерфейс"""
        interface = self.create_interface()
        interface.launch(share=False, server_port=7864)


if __name__ == "__main__":
    app = RussianTTSWebInterface()
    app.launch()
