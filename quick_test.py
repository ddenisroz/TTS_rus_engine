#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Быстрый тест F5-TTS
"""

import logging
import os
from correct_tts import CorrectTTS

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_test():
    """Быстрый тест синтеза"""
    logger.info("🚀 Быстрый тест F5-TTS")
    
    try:
        # Инициализация TTS
        tts = CorrectTTS()
        
        if not tts.f5tts:
            logger.error("❌ F5-TTS модель не загружена")
            return
        
        # Проверяем референсное аудио
        ref_audio = "test_ref.wav"
        if not os.path.exists(ref_audio):
            logger.error(f"❌ Референсный аудиофайл не найден: {ref_audio}")
            logger.info("Создайте референсный аудиофайл с помощью: python create_test_audio.py")
            return
        
        # Тестовый синтез
        text = "Привет! Это быстрый тест F5-TTS с русским языком."
        ref_text = "Это референсный текст для клонирования голоса."
        
        logger.info(f"Синтезируем: '{text}'")
        
        output_path = tts.synthesize_speech(text, ref_audio, ref_text)
        
        if output_path and os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logger.info(f"✅ Успех! Результат: {output_path} ({file_size} байт)")
            logger.info("🎵 Можете прослушать результат!")
        else:
            logger.error("❌ Ошибка синтеза")
            
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    quick_test()
