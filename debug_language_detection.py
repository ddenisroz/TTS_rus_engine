#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Отладка определения языка
"""

from multilingual_tts import MultilingualTTS

def debug_language_detection():
    """Отладка определения языка"""
    
    print("🔍 Отладка определения языка")
    print()
    
    # Инициализируем TTS
    tts = MultilingualTTS(enable_accent=True)
    
    # Тестовые случаи
    test_cases = [
        "Hello! How are you?",
        "Привет! Как дела?",
        "Hello привет!",
        "kill four pill",
        "юра kill4pill отсосал все хуи"
    ]
    
    for text in test_cases:
        print(f"📝 Текст: '{text}'")
        
        # Определяем язык
        language = tts.detect_language(text)
        print(f"🌍 Определенный язык: {language}")
        
        # Предобработка
        processed = tts.preprocess_text_for_tts(text)
        print(f"⚙️ Обработанный текст: '{processed}'")
        
        # Определяем язык после обработки
        language_after = tts.detect_language(processed)
        print(f"🌍 Язык после обработки: {language_after}")
        
        print("-" * 50)
        print()

if __name__ == "__main__":
    debug_language_detection()
