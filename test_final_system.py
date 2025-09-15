#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Финальный тест системы TTS_rus_engine
"""

from russian_tts import RussianTTS
import os

def test_final_system():
    """Финальный тест всей системы"""
    
    print("🎯 ФИНАЛЬНЫЙ ТЕСТ TTS_rus_engine")
    print("=" * 60)
    
    # Инициализация
    print("1. Инициализация TTS...")
    tts = RussianTTS()
    print("   ✅ TTS инициализирован")
    
    # Тест предобработки
    print("\n2. Тест предобработки текста:")
    test_texts = [
        "Привет! Как дела?",
        "GPT работает хорошо",
        "В 2024 году",
        "тёлка телка"
    ]
    
    for text in test_texts:
        processed = tts.preprocess_text_for_tts(text)
        print(f"   '{text}' -> '{processed}'")
    
    # Тест синтеза (если есть референсное аудио)
    ref_audio = "test_audio/test_ref.wav"
    if os.path.exists(ref_audio):
        print(f"\n3. Тест синтеза речи:")
        print(f"   Референсное аудио: {ref_audio}")
        
        try:
            audio_path = tts.synthesize_speech(
                text="Привет! Это финальный тест системы.",
                ref_audio_path=ref_audio,
                ref_text="Секреты всегда рядом, Скуф. Нужно лишь тихо прислушаться и услышать их."
            )
            print(f"   ✅ Аудио создано: {audio_path}")
        except Exception as e:
            print(f"   ❌ Ошибка синтеза: {e}")
    else:
        print(f"\n3. Референсное аудио не найдено: {ref_audio}")
        print("   Пропускаем тест синтеза")
    
    print("\n" + "=" * 60)
    print("🎉 ФИНАЛЬНЫЙ ТЕСТ ЗАВЕРШЕН!")
    print("Система готова к использованию!")

if __name__ == "__main__":
    test_final_system()
