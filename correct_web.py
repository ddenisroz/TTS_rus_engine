#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Правильный веб-интерфейс для F5-TTS с русским языком
"""

import gradio as gr
import tempfile
import os
import logging
from pathlib import Path
from correct_tts import CorrectTTS
import whisper

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CorrectTTSWebInterface:
    def __init__(self):
        self.tts = None  # Будет инициализирован при выборе модели
        self.whisper_model = None
        self.temp_dir = Path(tempfile.gettempdir()) / "f5tts_web"
        self.temp_dir.mkdir(exist_ok=True)
        self.current_model_type = "base"
        
    def load_whisper_model(self):
        """Загружает модель Whisper для транскрипции"""
        if self.whisper_model is None:
            try:
                logger.info("Загружаем модель Whisper...")
                # Загружаем модель на CPU чтобы избежать проблем с CUDA
                self.whisper_model = whisper.load_model("base", device="cpu")
                logger.info("Модель Whisper загружена на CPU")
            except Exception as e:
                logger.error(f"Ошибка загрузки Whisper: {e}")
                return False
        return True
    
    def transcribe_audio(self, audio_file):
        """Транскрибирует аудиофайл"""
        if audio_file is None:
            return "Загрузите аудиофайл для транскрипции"
        
        try:
            if not self.load_whisper_model():
                return "Ошибка загрузки модели Whisper"
            
            logger.info(f"Начинаю транскрипцию: {audio_file}")
            
            # Используем CPU для Whisper чтобы избежать проблем с CUDA
            result = self.whisper_model.transcribe(audio_file)
            text = result["text"].strip()
            language = result["language"]
            
            logger.info(f"Определенный язык: {language}")
            logger.info(f"Транскрипция завершена: '{text[:50]}...'")
            
            return f"Язык: {language}\n\nТекст: {text}"
            
        except Exception as e:
            logger.error(f"Ошибка при транскрипции: {e}")
            return f"Ошибка транскрипции: {e}"
    
    def load_tts_model(self, model_type):
        """Загружает TTS модель выбранного типа"""
        try:
            if self.tts is None or self.current_model_type != model_type:
                logger.info(f"Загружаем TTS модель: {model_type}")
                self.tts = CorrectTTS(model_type=model_type)
                self.current_model_type = model_type
                return f"✅ Модель {model_type} загружена успешно!"
            else:
                return f"✅ Модель {model_type} уже загружена"
        except Exception as e:
            logger.error(f"Ошибка загрузки модели {model_type}: {e}")
            return f"❌ Ошибка загрузки модели: {e}"

    def synthesize_speech(self, text, ref_audio, ref_text, model_type):
        """Синтезирует речь"""
        if not text or not ref_audio or not ref_text:
            return None, "Заполните все поля"
        
        try:
            # Загружаем модель если нужно
            if self.tts is None or self.current_model_type != model_type:
                self.load_tts_model(model_type)
            
            if not self.tts or not self.tts.f5tts:
                return None, "TTS модель не загружена"
            
            # ref_audio уже является путем к файлу (строка)
            ref_audio_path = str(ref_audio)
            
            # Синтезируем речь
            logger.info(f"Синтез речи ({model_type}): '{text[:50]}...'")
            output_path = self.tts.synthesize_speech(text, ref_audio_path, ref_text)
            
            if output_path:
                return output_path, f"Синтез завершен успешно! (модель: {model_type})"
            else:
                return None, "Ошибка синтеза речи"
            
        except Exception as e:
            logger.error(f"Ошибка в synthesize_speech: {e}")
            return None, f"Ошибка синтеза: {e}"
    
    def create_interface(self):
        """Создает веб-интерфейс"""
        with gr.Blocks(title="F5-TTS Russian", theme=gr.themes.Soft()) as interface:
            gr.Markdown("""
            # 🎤 F5-TTS Russian Voice Cloning
            
            **Возможности:**
            - 🗣️ Клонирование голоса с референсного аудио
            - 🇷🇺 Поддержка русского языка
            - 📝 Транскрипция аудио в текст
            - 🎵 Высококачественный синтез речи
            - 🔄 Выбор между базовой и accent-tuned моделями
            
            **Инструкция:**
            1. Выберите модель (Base или Accent-tuned)
            2. Загрузите референсное аудио (голос для клонирования)
            3. Введите текст референсного аудио (что говорится в аудио)
            4. Введите текст для синтеза
            5. Нажмите "Синтезировать речь"
            """)
            
            with gr.Row():
                with gr.Column():
                    # Выбор модели
                    model_choice = gr.Radio(
                        choices=[
                            ("Base Model (F5TTS_v1_Base_v2)", "base"),
                            ("Accent-tuned Model (F5TTS_v1_Base_accent_tune)", "accent")
                        ],
                        value="base",
                        label="🤖 Выберите модель TTS",
                        info="Base - стандартная модель, Accent-tuned - улучшенная для акцентов"
                    )
                    
                    # Кнопка загрузки модели
                    load_model_btn = gr.Button("🔄 Загрузить модель", variant="secondary")
                    
                    # Статус загрузки модели
                    model_status = gr.Textbox(
                        label="📊 Статус модели",
                        value="Выберите модель и нажмите 'Загрузить модель'",
                        interactive=False
                    )
                    
                    # Референсное аудио
                    ref_audio = gr.Audio(
                        label="🎵 Референсное аудио (голос для клонирования)",
                        type="filepath"
                    )
                    
                    # Текст референсного аудио
                    ref_text = gr.Textbox(
                        label="📝 Текст референсного аудио",
                        placeholder="Введите текст, который произносится в референсном аудио...",
                        lines=3
                    )
                    
                    # Кнопка транскрипции
                    transcribe_btn = gr.Button("🎤 Расшифровать аудио", variant="secondary")
                    
                    # Результат транскрипции
                    transcribe_output = gr.Textbox(
                        label="📄 Результат транскрипции",
                        lines=4,
                        interactive=False
                    )
                
                with gr.Column():
                    # Текст для синтеза
                    gen_text = gr.Textbox(
                        label="✍️ Текст для синтеза",
                        placeholder="Введите текст, который нужно синтезировать...",
                        lines=5
                    )
                    
                    # Кнопка синтеза
                    synthesize_btn = gr.Button("🎵 Синтезировать речь", variant="primary")
                    
                    # Результат синтеза
                    output_audio = gr.Audio(
                        label="🎵 Результат синтеза",
                        type="filepath"
                    )
                    
                    # Статус синтеза
                    status = gr.Textbox(
                        label="📊 Статус синтеза",
                        value="Выберите модель и загрузите референсное аудио",
                        interactive=False
                    )
            
            # Обработчики событий
            load_model_btn.click(
                fn=self.load_tts_model,
                inputs=[model_choice],
                outputs=[model_status]
            )
            
            transcribe_btn.click(
                fn=self.transcribe_audio,
                inputs=[ref_audio],
                outputs=[transcribe_output]
            )
            
            synthesize_btn.click(
                fn=self.synthesize_speech,
                inputs=[gen_text, ref_audio, ref_text, model_choice],
                outputs=[output_audio, status]
            )
            
            # Автоматическое заполнение референсного текста
            ref_audio.change(
                fn=self.transcribe_audio,
                inputs=[ref_audio],
                outputs=[transcribe_output]
            )
        
        return interface
    
    def launch(self, port=7862, share=False):
        """Запускает веб-интерфейс"""
        interface = self.create_interface()
        
        logger.info("Запуск F5-TTS веб-интерфейса...")
        interface.launch(
            server_port=port,
            share=share,
            show_error=True,
            quiet=False
        )

def main():
    """Главная функция"""
    try:
        app = CorrectTTSWebInterface()
        app.launch(port=7862, share=False)
    except Exception as e:
        logger.error(f"Ошибка запуска: {e}")

if __name__ == "__main__":
    main()
