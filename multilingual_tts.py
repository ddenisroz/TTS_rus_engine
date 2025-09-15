#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Мультиязычная реализация F5-TTS с поддержкой русского и английского языков
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional
import tempfile

import numpy as np
import soundfile as sf
import torch
from f5_tts.api import F5TTS
from huggingface_hub import hf_hub_download
from ruaccent import RUAccent
from yoficator_module import yoficate_text

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Константы для русской модели F5-TTS
RUSSIAN_MODEL = "Misha24-10/F5-TTS_RUSSIAN"
RUSSIAN_CHECKPOINT = "F5TTS_v1_Base_v2/model_last_inference.safetensors"
RUSSIAN_VOCAB = "F5TTS_v1_Base/vocab.txt"

# Транскрипция для референсного голоса
DEFAULT_VOICE_TRANSCRIPTION = "Секреты всегда рядом, Скуф. Нужно лишь тихо прислушаться и услышать их."


class MultilingualTTS:
    def __init__(self, enable_accent=True, 
                 accent_model_size="turbo", ode_method="euler", use_ema=True):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.enable_accent = enable_accent
        self.accent_model_size = accent_model_size
        self.ode_method = ode_method
        self.use_ema = use_ema
        logger.info(f"F5-TTS использует устройство: {self.device}")

        # F5-TTS модели
        self.russian_tts = None
        self.accentizer = None
        
        # Загружаем модели
        self._load_models()


    def _load_models(self):
        """Загружает F5-TTS модели и RUAccent."""
        try:
            # Загружаем RUAccent для ударений
            if self.enable_accent:
                logger.info(f"Загружаем RUAccent для расстановки ударений (модель: {self.accent_model_size})...")
                try:
                    self.accentizer = RUAccent()
                    # Инициализируем модель с выбранным размером
                    self.accentizer.load(omograph_model_size=self.accent_model_size, use_dictionary=True)
                    logger.info(f"RUAccent загружен успешно (модель: {self.accent_model_size})")
                except Exception as e:
                    logger.warning(f"Не удалось загрузить RUAccent: {e}")
                    self.accentizer = None
            
            # Загружаем русскую модель
            self._load_russian_model()

        except Exception as e:
            logger.error(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить модели. Ошибка: {e}", exc_info=True)


    def _load_russian_model(self):
        """Загружает русскую F5-TTS модель."""
        try:
            logger.info("Загружаем русскую F5-TTS модель...")
            
            cache_dir = Path("f5_tts_cache")
            cache_dir.mkdir(exist_ok=True)

            # Скачиваем русскую модель checkpoint
            russian_ckpt_path = hf_hub_download(
                repo_id=RUSSIAN_MODEL,
                filename=RUSSIAN_CHECKPOINT,
                cache_dir=cache_dir
            )
            logger.info(f"Русский checkpoint скачан в: {russian_ckpt_path}")

            # Скачиваем vocab.txt для русской модели
            russian_vocab_path = hf_hub_download(
                repo_id=RUSSIAN_MODEL,
                filename=RUSSIAN_VOCAB,
                cache_dir=cache_dir
            )
            logger.info(f"Русский vocab.txt скачан в: {russian_vocab_path}")

            self.russian_tts = F5TTS(
                model="F5TTS_v1_Base",
                ckpt_file=russian_ckpt_path,
                vocab_file=russian_vocab_path,
                ode_method=self.ode_method,
                use_ema=self.use_ema,
                device=self.device,
                hf_cache_dir=str(cache_dir)
            )
            
            logger.info("Русская F5-TTS модель загружена успешно.")

        except Exception as e:
            logger.error(f"Ошибка загрузки русской модели: {e}", exc_info=True)
            self.russian_tts = None

    def detect_language(self, text: str) -> str:
        """Определяет язык текста."""
        # Подсчитываем количество кириллических и латинских символов
        cyrillic_pattern = re.compile(r'[а-яё]', re.IGNORECASE)
        latin_pattern = re.compile(r'[a-z]', re.IGNORECASE)
        
        cyrillic_count = len(cyrillic_pattern.findall(text))
        latin_count = len(latin_pattern.findall(text))
        
        logger.info(f"Анализ языка: кириллица={cyrillic_count}, латиница={latin_count}")
        
        # Если больше кириллических символов - русский
        if cyrillic_count > latin_count:
            logger.info(f"Выбран русский язык (кириллица > латиницы)")
            return "russian"
        # Если больше латинских символов - английский
        elif latin_count > cyrillic_count:
            logger.info(f"Выбран английский язык (латиница > кириллицы)")
            return "english"
        # Если равное количество - проверяем наличие кириллицы
        elif cyrillic_count > 0:
            logger.info(f"Выбран русский язык (есть кириллица при равном количестве)")
            return "russian"
        else:
            # Если нет букв вообще (только числа и символы) - по умолчанию русский
            logger.info(f"Выбран русский язык (по умолчанию для чисел и символов)")
            return "russian"

    def normalize_text(self, text: str, language: str) -> str:
        """Нормализует текст с помощью кастомной конвертации чисел и ёфикатора."""
        if not text.strip():
            return text
        
        # Предобработка для исправления проблем с GPT
        preprocessed_text = self._preprocess_gpt_issues(text)
        
        # Используем кастомную конвертацию чисел
        normalized_text = self._simple_number_conversion(preprocessed_text, language)
        logger.info(f"Кастомная нормализация: '{preprocessed_text[:50]}...' -> '{normalized_text[:50]}...'")
        
        # Применяем ёфикатор для русского текста
        if language == "russian":
            try:
                yoficated_text = yoficate_text(normalized_text)
                logger.info(f"Ёфикация: '{normalized_text[:50]}...' -> '{yoficated_text[:50]}...'")
                return yoficated_text
            except Exception as e:
                logger.warning(f"Ошибка ёфикации: {e}")
                return normalized_text
        
        return normalized_text
    
    def _preprocess_gpt_issues(self, text: str) -> str:
        """Предобработка текста (заглушка для совместимости)."""
        # GPT теперь обрабатывается в общем словаре аббревиатур
        return text

    def _simple_number_conversion(self, text: str, language: str) -> str:
        """Улучшенная конвертация чисел в слова с поддержкой тысяч и аббревиатур."""
        import re
        
        if language == "russian":
            # Обрабатываем числа от 0 до 999999
            def convert_russian_number(num_str):
                try:
                    num = int(num_str)
                    if num == 0:
                        return "ноль"
                    elif num < 20:
                        return self._russian_units[num]
                    elif num < 100:
                        tens = num // 10
                        units = num % 10
                        if units == 0:
                            return self._russian_tens[tens]
                        else:
                            return f"{self._russian_tens[tens]} {self._russian_units[units]}"
                    elif num < 1000:
                        hundreds = num // 100
                        remainder = num % 100
                        if remainder == 0:
                            return self._russian_hundreds[hundreds]
                        else:
                            return f"{self._russian_hundreds[hundreds]} {convert_russian_number(remainder)}"
                    elif num < 1000000:
                        thousands = num // 1000
                        remainder = num % 1000
                        if remainder == 0:
                            if thousands == 1:
                                return "тысяча"
                            elif thousands < 5:
                                return f"{convert_russian_number(thousands)} тысячи"
                            else:
                                return f"{convert_russian_number(thousands)} тысяч"
                        else:
                            if thousands == 1:
                                return f"тысяча {convert_russian_number(remainder)}"
                            elif thousands < 5:
                                return f"{convert_russian_number(thousands)} тысячи {convert_russian_number(remainder)}"
                            else:
                                return f"{convert_russian_number(thousands)} тысяч {convert_russian_number(remainder)}"
                    else:
                        return num_str  # Слишком большое число
                except:
                    return num_str
            
            # Словари для русского языка
            self._russian_units = {
                1: "один", 2: "два", 3: "три", 4: "четыре", 5: "пять",
                6: "шесть", 7: "семь", 8: "восемь", 9: "девять", 10: "десять",
                11: "одиннадцать", 12: "двенадцать", 13: "тринадцать", 14: "четырнадцать",
                15: "пятнадцать", 16: "шестнадцать", 17: "семнадцать", 18: "восемнадцать",
                19: "девятнадцать"
            }
            
            self._russian_tens = {
                2: "двадцать", 3: "тридцать", 4: "сорок", 5: "пятьдесят",
                6: "шестьдесят", 7: "семьдесят", 8: "восемьдесят", 9: "девяносто"
            }
            
            self._russian_hundreds = {
                1: "сто", 2: "двести", 3: "триста", 4: "четыреста",
                5: "пятьсот", 6: "шестьсот", 7: "семьсот", 8: "восемьсот", 9: "девятьсот"
            }
            
            # Расширенный словарь аббревиатур на основе RUNorm
            abbreviations = {
                # Сначала обрабатываем сложные аббревиатуры с точками
                'к.м.н.': 'кандидат медицинских наук', 'д.м.н.': 'доктор медицинских наук',
                'к.т.н.': 'кандидат технических наук', 'д.т.н.': 'доктор технических наук',
                'т.е.': 'то есть', 'т.к.': 'так как', 'т.д.': 'так далее',
                'т.п.': 'тому подобное', 'и т.д.': 'и так далее', 'и т.п.': 'и тому подобное',
                
                # Медицинские и научные
                'др.': 'доктор', 'проф.': 'профессор', 'акад.': 'академик',
                
                # Временные (приоритет: год, век)
                'гг.': 'годы', 'вв.': 'века', 'г.': 'год',
                'мин.': 'минута', 'сек.': 'секунда', 'ч.': 'час', 'сут.': 'сутки',
                
                # Издательские
                'тт.': 'тома', 'стр.': 'страница', 'с.': 'страница', 
                'гл.': 'глава', 'пп.': 'пункты', 'чч.': 'части',
                'т.': 'том', 'п.': 'пункт', 'ч.': 'часть',
                
                # Технические аббревиатуры (побуквенно)
                'GPT': 'джи пи ти', 'AI': 'эй ай', 'CPU': 'си пи ю', 'GPU': 'джи пи ю',
                'RAM': 'рэм', 'ROM': 'ром', 'USB': 'ю эс би', 'HDD': 'хэ дэ дэ',
                'SSD': 'эс эс дэ', 'API': 'эй пи ай', 'URL': 'ю ар эл', 'HTTP': 'хэ тэ тэ пи',
                'HTTPS': 'хэ тэ тэ пи эс', 'HTML': 'хэ тэ эм эл', 'CSS': 'си эс эс',
                'JS': 'джей эс', 'JSON': 'джейсон', 'XML': 'икс эм эл', 'PDF': 'пи дэ эф',
                'MP3': 'эм пи три', 'MP4': 'эм пи фо', 'AVI': 'эй ви ай', 'MKV': 'эм кей ви',
                'JPG': 'джей пи джи', 'PNG': 'пи эн джи', 'GIF': 'джи ай эф',
                'ZIP': 'зип', 'RAR': 'рар', '7Z': 'семь зет',
                
                # Организации и стандарты
                'UN': 'ю эн', 'UNESCO': 'юнеско', 'NASA': 'наса', 'FBI': 'эф би ай',
                'CIA': 'си ай эй', 'KGB': 'ка гэ бэ', 'USSR': 'у эс эс эр',
                'EU': 'и ю', 'NATO': 'нато', 'WHO': 'дабл ю эйч о',
                
                # Единицы измерения (исключаем "В" и "г" чтобы избежать конфликтов)
                'кг': 'килограмм', 'км': 'километр', 'см': 'сантиметр', 'мм': 'миллиметр', 
                'л': 'литр', 'мл': 'миллилитр', 'Вт': 'ватт', 'кВт': 'киловатт', 
                'А': 'ампер', 'Гц': 'герц', 'кГц': 'килогерц', 'МГц': 'мегагерц', 'ГГц': 'гигагерц',
                'м': 'метр',
                
                # Другие распространенные
                'см.': 'смотри', 'напр.': 'например', 'ок.': 'около', 
                'прим.': 'примерно', 'макс.': 'максимум', 'мин.': 'минимум', 
                'ср.': 'средний',
            }
            
            result = text
            
            # Специальная обработка для "В" в начале предложения
            result = re.sub(r'^В\s+', 'В ', result)
            
            # Заменяем аббревиатуры (сначала длинные, потом короткие)
            for abbr, full in sorted(abbreviations.items(), key=lambda x: len(x[0]), reverse=True):
                # Для аббревиатур с точками используем более точный паттерн
                if '.' in abbr:
                    # Экранируем точки и ищем точные совпадения
                    escaped_abbr = re.escape(abbr)
                    # Ищем аббревиатуру, которая либо в начале строки, либо после пробела/знака препинания
                    pattern = r'(?<!\w)' + escaped_abbr + r'(?!\w)'
                else:
                    # Для обычных аббревиатур используем word boundaries
                    pattern = r'\b' + re.escape(abbr) + r'\b'
                
                result = re.sub(pattern, full, result, flags=re.IGNORECASE)
            
            # Заменяем числа
            def replace_numbers(match):
                return convert_russian_number(match.group())
            
            result = re.sub(r'\b\d+\b', replace_numbers, result)
            
        else:
            # Английская конвертация (упрощенная)
            number_map = {
                '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
                '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
                '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen',
                '14': 'fourteen', '15': 'fifteen', '16': 'sixteen', '17': 'seventeen',
                '18': 'eighteen', '19': 'nineteen', '20': 'twenty', '30': 'thirty',
                '40': 'forty', '50': 'fifty', '60': 'sixty', '70': 'seventy',
                '80': 'eighty', '90': 'ninety', '100': 'one hundred', '1000': 'one thousand'
            }
            
            result = text
            for num, word in sorted(number_map.items(), key=lambda x: len(x[0]), reverse=True):
                result = result.replace(num, word)
        
        return result

    def _is_only_symbols(self, text: str) -> bool:
        """Проверяет, состоит ли текст только из знаков препинания и символов."""
        # Убираем пробелы и проверяем, остались ли только символы
        text_no_spaces = text.replace(" ", "")
        if not text_no_spaces:
            return True
        
        # Проверяем, есть ли хотя бы одна буква или цифра
        has_letter_or_digit = any(c.isalnum() for c in text_no_spaces)
        return not has_letter_or_digit

    def _remove_long_symbol_sequences(self, text: str) -> str:
        """Удаляет последовательности из более чем 3 знаков подряд."""
        import re
        # Заменяем последовательности из 4+ одинаковых символов на 3
        pattern = r'(.)\1{3,}'
        return re.sub(pattern, r'\1\1\1', text)

    def add_accents(self, text: str) -> str:
        """Добавляет ударения к русскому тексту."""
        if not self.accentizer or not text.strip():
            return text
        
        try:
            # Пробуем разные методы RUAccent
            if hasattr(self.accentizer, 'process_all'):
                accented_text = self.accentizer.process_all(text)
            elif hasattr(self.accentizer, 'process'):
                accented_text = self.accentizer.process(text)
            else:
                # Если ничего не работает, возвращаем исходный текст
                logger.warning("RUAccent не поддерживает доступные методы")
                return text
                
            logger.info(f"Добавлены ударения: '{text[:50]}...' -> '{accented_text[:50]}...'")
            return accented_text
        except Exception as e:
            logger.warning(f"Ошибка добавления ударений: {e}")
            return text

    def preprocess_text_for_tts(self, text: str) -> str:
        """Предобработка текста с учетом языка и конвертацией чисел."""
        processed_text = text.strip()
        if not processed_text:
            return ""
        
        logger.info(f"Исходный текст: '{processed_text}'")
        
        # Проверяем, состоит ли сообщение только из знаков препинания/символов
        if self._is_only_symbols(processed_text):
            logger.warning("Сообщение состоит только из знаков - игнорируем")
            return ""
        
        # Убираем последовательности из более чем 3 знаков подряд
        processed_text = self._remove_long_symbol_sequences(processed_text)
        if not processed_text.strip():
            logger.warning("После удаления длинных последовательностей символов текст стал пустым")
            return ""
        
        # Определяем язык
        language = self.detect_language(processed_text)
        logger.info(f"Определенный язык: {language}")
        
        # Убираем лишние пробелы
        processed_text = ' '.join(processed_text.split())
        
        # Нормализуем текст (конвертация чисел, ёфикация и т.д.)
        processed_text = self.normalize_text(processed_text, language)
        logger.info(f"После нормализации: '{processed_text}'")
        
        # Для русского текста добавляем ударения
        if language == "russian" and self.enable_accent:
            processed_text = self.add_accents(processed_text)
        
        # Обработка окончаний - добавляем только точку если ее не было
        if processed_text:
            # Убираем лишние пробелы в конце
            processed_text = processed_text.rstrip()
            
            # Добавляем точку в конце если нет знаков препинания
            if not processed_text.endswith(('.', '!', '?')):
                processed_text += '.'
        
        logger.info(f"Обработанный текст: '{processed_text}'")
        
        return processed_text

    def synthesize_speech(self, text: str, ref_audio_path: str, ref_text: str = "", 
                         cross_fade_duration: float = 0.15, speed: float = None, 
                         silence_duration_ms: int = 100, target_rms: float = 0.1,
                         sway_sampling_coef: float = -1, cfg_strength: float = 2,
                         nfe_step: int = None, fix_duration: Optional[float] = None,
                         remove_silence: bool = False, seed: Optional[int] = None) -> Optional[str]:
        """Синтезирует речь с автоматическим выбором модели по языку."""
        
        # Предобработка текста
        processed_text = self.preprocess_text_for_tts(text)
        if not processed_text:
            logger.warning("Текст пустой после предобработки")
            return None

        # Определяем язык и выбираем модель
        language = self.detect_language(processed_text)
        
        # Автоматическое определение скорости на основе длины обработанного текста
        if speed is None:
            length_without_spaces = len(processed_text.replace(" ", ""))
            if length_without_spaces <= 3:
                speed = 0.1
            elif length_without_spaces <= 8:
                speed = 0.3
            elif length_without_spaces <= 18:
                speed = 0.6
            elif length_without_spaces <= 35:
                speed = 0.8
            elif length_without_spaces <= 45:
                speed = 0.9
            else:
                speed = 1.0
            logger.info(f"Автоматически определена скорость: {speed} (длина обработанного текста: {length_without_spaces})")
        
        # Автоматическое определение NFE steps на основе длины обработанного текста
        if nfe_step is None:
            length_without_spaces = len(processed_text.replace(" ", ""))
            if length_without_spaces > 120:
                nfe_step = 18
            else:
                nfe_step = 26
            logger.info(f"Автоматически определен NFE steps: {nfe_step} (длина обработанного текста: {length_without_spaces})")
        
        # Используем русскую модель
        tts_model = self.russian_tts
        model_name = "Russian"
        
        if not tts_model:
            logger.error(f"Модель {model_name} не загружена")
            return None

        # Используем предопределенную транскрипцию если ref_text пустой
        ref_text_to_use = ref_text if ref_text else DEFAULT_VOICE_TRANSCRIPTION

        logger.info(f"Синтезируем аудио ({model_name}): '{processed_text}' используя голос '{ref_audio_path}'")
        logger.info(f"Параметры: cross_fade={cross_fade_duration}, speed={speed}, silence={silence_duration_ms}ms")
        logger.info(f"F5-TTS параметры: target_rms={target_rms}, sway={sway_sampling_coef}, cfg={cfg_strength}, nfe={nfe_step}")

        try:
            # Создаем выходную директорию
            output_dir = Path("audio_output")
            output_dir.mkdir(exist_ok=True)
            
            # Создаем уникальное имя файла с временной меткой
            import time
            timestamp = int(time.time() * 1000)
            output_filename = f"russian_{language}_{timestamp}.wav"
            output_path = output_dir / output_filename

            # Параметры для F5-TTS
            infer_params = {
                "ref_file": ref_audio_path,
                "ref_text": ref_text_to_use,
                "gen_text": processed_text,
                "cross_fade_duration": cross_fade_duration,
                "speed": speed,
                "target_rms": target_rms,
                "sway_sampling_coef": sway_sampling_coef,
                "cfg_strength": cfg_strength,
                "nfe_step": nfe_step,
                "remove_silence": remove_silence
            }
            
            # Добавляем опциональные параметры
            if fix_duration is not None:
                infer_params["fix_duration"] = fix_duration
            if seed is not None:
                infer_params["seed"] = seed

            logger.info(f"Параметры: cross_fade={cross_fade_duration}, speed={speed}, silence={silence_duration_ms}ms")
            logger.info(f"F5-TTS параметры: target_rms={target_rms}, sway={sway_sampling_coef}, cfg={cfg_strength}, nfe={nfe_step}")

            # Синтезируем аудио
            wav, sr, spect = tts_model.infer(**infer_params)

            # Улучшенная обработка для предотвращения обрывов окончаний
            
            # 1. Добавляем больше тишины в конец (увеличиваем с 200ms до 800ms)
            extended_silence_ms = max(silence_duration_ms, 800)  # Минимум 800ms
            silence_samples = int(sr * (extended_silence_ms / 1000.0))
            silence = np.zeros(silence_samples, dtype=np.float32)
            wav_padded = np.concatenate([wav, silence])
            
            # 2. Добавляем более длинный fade-out (увеличиваем с 100ms до 300ms)
            fade_samples = int(sr * 0.3)  # 300ms fade-out для более плавного затухания
            if len(wav_padded) > fade_samples:
                # Используем косинусоидальное окно для более естественного затухания
                fade = np.cos(np.linspace(0, np.pi/2, fade_samples))
                wav_padded[-fade_samples:] *= fade
            
            # 3. Добавляем дополнительную тишину после fade-out
            post_fade_silence = int(sr * 0.1)  # 100ms тишины после fade-out
            post_silence = np.zeros(post_fade_silence, dtype=np.float32)
            wav_padded = np.concatenate([wav_padded, post_silence])

            # Сохраняем аудио
            sf.write(str(output_path), wav_padded, sr)
            
            logger.info(f"Аудио синтезировано и сохранено в {output_path} (модель: {model_name})")
            return str(output_path)

        except Exception as e:
            logger.error(f"Ошибка при синтезе: {e}", exc_info=True)
            return None


if __name__ == "__main__":
    # Тестирование
    tts = MultilingualTTS()
    
    if tts.russian_tts:
        print("✅ Русская модель загружена успешно!")
        
        # Тест с русским текстом
        print("\n🧪 Тест с русским текстом:")
        result = tts.synthesize_speech(
            text="Привет! Как дела?",
            ref_audio_path="test_ref.wav",
            ref_text="Это тестовое аудио"
        )
        if result:
            print(f"✅ Результат: {result}")
        
        # Тест с английским текстом
        print("\n🧪 Тест с английским текстом:")
        result = tts.synthesize_speech(
            text="Hello! How are you?",
            ref_audio_path="test_ref.wav",
            ref_text="This is a test audio"
        )
        if result:
            print(f"✅ Результат: {result}")
    else:
        print("❌ Ошибка загрузки моделей!")
