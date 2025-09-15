#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Продвинутый веб-интерфейс для F5-TTS с поддержкой ударений и мультиязычности
"""

import gradio as gr
import tempfile
import os
import logging
from pathlib import Path
from advanced_tts import AdvancedTTS
import whisper

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedTTSWebInterface:
    def __init__(self):
        self.tts = None  # Будет инициализирован при выборе модели
        self.whisper_model = None
        self.temp_dir = Path(tempfile.gettempdir()) / "f5tts_web"
        self.temp_dir.mkdir(exist_ok=True)
        self.current_model_type = "base"
        self.current_accent_enabled = True
        
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
    
    def load_tts_model(self, model_type, enable_accent, accent_model_size, accent_use_dictionary, 
                      accent_tiny_mode, ode_method, use_ema):
        """Загружает TTS модель выбранного типа"""
        try:
            if (self.tts is None or 
                self.current_model_type != model_type or 
                self.current_accent_enabled != enable_accent):
                
                logger.info(f"Загружаем TTS модель: {model_type}, ударения: {enable_accent}")
                self.tts = AdvancedTTS(
                    model_type=model_type, 
                    enable_accent=enable_accent,
                    accent_model_size=accent_model_size,
                    accent_use_dictionary=accent_use_dictionary,
                    accent_tiny_mode=accent_tiny_mode,
                    ode_method=ode_method,
                    use_ema=use_ema
                )
                self.current_model_type = model_type
                self.current_accent_enabled = enable_accent
                
                # Проверяем какие модели загружены
                status_parts = []
                if self.tts.russian_tts:
                    status_parts.append("✅ Русская модель F5-TTS")
                if self.tts.accentizer:
                    status_parts.append(f"✅ RUAccent (ударения, размер: {accent_model_size})")
                status_parts.append("✅ Транслитерация латиницы в кириллицу")
                
                status = f"Модель {model_type} загружена!\n" + "\n".join(status_parts)
                return status
            else:
                return f"✅ Модель {model_type} уже загружена"
        except Exception as e:
            logger.error(f"Ошибка загрузки модели {model_type}: {e}")
            return f"❌ Ошибка загрузки модели: {e}"

    def synthesize_speech(self, text, ref_audio, ref_text, model_type, enable_accent, 
                         cross_fade_duration, speed, silence_duration_ms, target_rms,
                         sway_sampling_coef, cfg_strength, nfe_step, fix_duration,
                         remove_silence, seed):
        """Синтезирует речь"""
        if not text or not ref_audio or not ref_text:
            return None, "Заполните все поля"
        
        try:
            # Загружаем модель если нужно
            if (self.tts is None or 
                self.current_model_type != model_type or 
                self.current_accent_enabled != enable_accent):
                self.load_tts_model(model_type, enable_accent, "turbo2", True, False, "euler", True)
            
            if not self.tts:
                return None, "TTS модель не загружена"
            
            # ref_audio уже является путем к файлу (строка)
            ref_audio_path = str(ref_audio)
            
            # Обрабатываем параметры
            fix_duration_val = None if fix_duration is None or fix_duration == "" or fix_duration == 0 else float(fix_duration)
            seed_val = None if seed is None or seed == "" or seed == 0 else int(seed)
            
            # Синтезируем речь
            logger.info(f"Синтез речи ({model_type}, ударения: {enable_accent}): '{text[:50]}...'")
            output_path = self.tts.synthesize_speech(
                text, ref_audio_path, ref_text, 
                cross_fade_duration, speed, silence_duration_ms, target_rms,
                sway_sampling_coef, cfg_strength, nfe_step, fix_duration_val,
                remove_silence, seed_val
            )
            
            if output_path:
                return output_path, f"Синтез завершен успешно! (модель: {model_type}, ударения: {'включены' if enable_accent else 'отключены'})"
            else:
                return None, "Ошибка синтеза речи"
            
        except Exception as e:
            logger.error(f"Ошибка в synthesize_speech: {e}")
            return None, f"Ошибка синтеза: {e}"
    
    def reset_to_defaults(self):
        """Сбрасывает параметры к значениям по умолчанию"""
        return (
            "base",      # model_type
            True,        # enable_accent
            "turbo2",    # accent_model_size
            True,        # accent_use_dictionary
            False,       # accent_tiny_mode
            "euler",     # ode_method
            True,        # use_ema
            0.15,        # cross_fade_duration
            1.0,         # speed
            200,         # silence_duration_ms
            0.3,         # target_rms
            -1,          # sway_sampling_coef
            2,           # cfg_strength
            32,          # nfe_step
            0,           # fix_duration
            False,       # remove_silence
            0            # seed
        )
    
    def create_interface(self):
        """Создает веб-интерфейс"""
        with gr.Blocks(title="Advanced F5-TTS", theme=gr.themes.Soft()) as interface:
            gr.Markdown("""
            # 🎤 Advanced F5-TTS Multilingual Voice Cloning
            
            **Возможности:**
            - 🗣️ Клонирование голоса с референсного аудио
            - 🇷🇺 Поддержка русского языка с автоматическими ударениями
            - 🔤 Автоматическая транслитерация латинских букв в кириллицу
            - 📝 Транскрипция аудио в текст
            - 🎵 Высококачественный синтез речи
            - 🔄 Выбор между базовой и accent-tuned моделями
            - ⚡ Все тексты озвучиваются на русском языке
            
            **Инструкция:**
            1. Выберите модель (Base или Accent-tuned)
            2. Включите/отключите автоматические ударения
            3. Нажмите "Загрузить модель" - дождитесь загрузки
            4. Загрузите референсное аудио для клонирования голоса
            5. Введите текст референсного аудио (что говорится в аудио)
            6. Введите текст для синтеза (русский или английский - будет транслитерирован)
            7. Нажмите "Синтезировать речь"
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
                        label="🤖 Выберите русскую модель TTS",
                        info="Base - стандартная модель, Accent-tuned - улучшенная для акцентов"
                    )
                    
                    # Включение ударений
                    enable_accent = gr.Checkbox(
                        value=True,
                        label="⚡ Автоматические ударения для русского",
                        info="Включает RUAccent для улучшения произношения русских слов"
                    )
                    
                    # Кнопка загрузки модели
                    load_model_btn = gr.Button("🔄 Загрузить модель", variant="secondary")
                    
                    # Статус загрузки модели
                    model_status = gr.Textbox(
                        label="📊 Статус модели",
                        value="Выберите модель и нажмите 'Загрузить модель'",
                        lines=4,
                        interactive=False
                    )
                    
                    # Секция настроек модели
                    with gr.Group():
                        gr.Markdown("### ⚙️ Настройки модели")
                        
                        # Настройки RUAccent
                        accent_model_size = gr.Dropdown(
                            choices=["tiny", "turbo", "turbo2"],
                            value="turbo2",
                            label="📏 Размер модели RUAccent",
                            info="Размер модели для расстановки ударений (tiny - быстрее, turbo2 - точнее)"
                        )
                        
                        accent_use_dictionary = gr.Checkbox(
                            value=True,
                            label="📚 Использовать словарь RUAccent",
                            info="Включает использование словаря для более точной расстановки ударений"
                        )
                        
                        accent_tiny_mode = gr.Checkbox(
                            value=False,
                            label="⚡ Tiny режим RUAccent",
                            info="Включает быстрый режим работы (менее точный, но быстрее)"
                        )
                        
                        # Настройки F5-TTS
                        ode_method = gr.Dropdown(
                            choices=["euler", "rk4", "dopri5"],
                            value="euler",
                            label="🔢 ODE метод",
                            info="Метод численного интегрирования (euler - быстрее, dopri5 - точнее)"
                        )
                        
                        use_ema = gr.Checkbox(
                            value=True,
                            label="📈 Использовать EMA",
                            info="Включает экспоненциальное скользящее среднее для стабильности"
                        )
                    
                    # Секция настроек синтеза
                    with gr.Group():
                        gr.Markdown("### 🎵 Настройки синтеза")
                        
                        # Основные параметры синтеза
                        cross_fade_duration = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.15,
                            step=0.01,
                            label="🔄 Cross-fade Duration",
                            info="Длительность плавного перехода между сегментами (0.0-1.0)"
                        )
                        
                        speed = gr.Slider(
                            minimum=0.5,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="⚡ Скорость речи",
                            info="Скорость произношения (0.5-2.0)"
                        )
                        
                        silence_duration_ms = gr.Slider(
                            minimum=0,
                            maximum=1000,
                            value=200,
                            step=50,
                            label="🔇 Тишина в конце (мс)",
                            info="Длительность тишины в конце аудио (0-1000 мс)"
                        )
                    
                    # Секция настроек F5-TTS
                    with gr.Group():
                        gr.Markdown("### 🔧 Настройки F5-TTS")
                        
                        # Параметры качества
                        target_rms = gr.Slider(
                            minimum=0.01,
                            maximum=1.0,
                            value=0.3,
                            step=0.01,
                            label="📊 Target RMS",
                            info="Целевой уровень громкости аудио (0.01-1.0, 0.3 лучше для мультиязычной модели)"
                        )
                        
                        sway_sampling_coef = gr.Slider(
                            minimum=-1,
                            maximum=1,
                            value=-1,
                            step=0.1,
                            label="🌊 Sway Sampling",
                            info="Коэффициент случайности выборки (-1 = авто, 0-1 = ручная настройка)"
                        )
                        
                        cfg_strength = gr.Slider(
                            minimum=0,
                            maximum=10,
                            value=2,
                            step=0.1,
                            label="🎯 CFG Strength",
                            info="Сила классификатора свободного управления (0-10)"
                        )
                        
                        nfe_step = gr.Slider(
                            minimum=1,
                            maximum=100,
                            value=32,
                            step=1,
                            label="🔢 NFE Steps",
                            info="Количество шагов численного интегрирования (1-100)"
                        )
                        
                        # Дополнительные параметры
                        fix_duration = gr.Number(
                            value=0,
                            label="⏱️ Фиксированная длительность (сек)",
                            info="Фиксированная длительность аудио в секундах (0 = авто)"
                        )
                        
                        remove_silence = gr.Checkbox(
                            value=False,
                            label="🔇 Удалить тишину",
                            info="Автоматически удалять тишину в начале и конце"
                        )
                        
                        seed = gr.Number(
                            value=0,
                            label="🎲 Seed (случайность)",
                            info="Семя для воспроизводимости результатов (0 = случайность)"
                        )
                        
                        # Кнопка сброса к умолчаниям
                        reset_btn = gr.Button("🔄 Сбросить к умолчаниям", variant="secondary")
                    
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
                        label="✍️ Текст для синтеза (любой язык)",
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
                inputs=[model_choice, enable_accent, accent_model_size, accent_use_dictionary, 
                       accent_tiny_mode, ode_method, use_ema],
                outputs=[model_status]
            )
            
            transcribe_btn.click(
                fn=self.transcribe_audio,
                inputs=[ref_audio],
                outputs=[transcribe_output]
            )
            
            synthesize_btn.click(
                fn=self.synthesize_speech,
                inputs=[gen_text, ref_audio, ref_text, model_choice, enable_accent, 
                       cross_fade_duration, speed, silence_duration_ms, target_rms,
                       sway_sampling_coef, cfg_strength, nfe_step, fix_duration,
                       remove_silence, seed],
                outputs=[output_audio, status]
            )
            
            # Обработчик сброса к умолчаниям
            reset_btn.click(
                fn=self.reset_to_defaults,
                outputs=[model_choice, enable_accent, accent_model_size, accent_use_dictionary, 
                        accent_tiny_mode, ode_method, use_ema, cross_fade_duration, speed, silence_duration_ms,
                        target_rms, sway_sampling_coef, cfg_strength, nfe_step, fix_duration,
                        remove_silence, seed]
            )
            
            # Автоматическое заполнение референсного текста
            ref_audio.change(
                fn=self.transcribe_audio,
                inputs=[ref_audio],
                outputs=[transcribe_output]
            )
        
        return interface
    
    def launch(self, port=7863, share=False):
        """Запускает веб-интерфейс"""
        interface = self.create_interface()
        
        logger.info("Запуск Advanced F5-TTS веб-интерфейса...")
        interface.launch(
            server_port=port,
            share=share,
            show_error=True,
            quiet=False
        )

def main():
    """Главная функция"""
    try:
        app = AdvancedTTSWebInterface()
        app.launch(port=7863, share=False)
    except Exception as e:
        logger.error(f"Ошибка запуска: {e}")

if __name__ == "__main__":
    main()
