#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Диагностика синхронизации текста и озвучки
"""

from advanced_tts import AdvancedTTS
import os

def debug_text_sync():
    """Диагностирует проблемы с синхронизацией текста"""
    
    # Инициализируем TTS
    tts = AdvancedTTS(model_type="base", enable_accent=False)
    
    if not tts.multilingual_tts:
        print("❌ Мультиязычная модель не загружена!")
        return
    
    # Тестовые случаи
    test_cases = [
        {
            "name": "Простой английский",
            "text": "Hello world",
            "ref_text": "This is a test"
        },
        {
            "name": "Русский текст",
            "text": "Привет мир",
            "ref_text": "Это тест"
        },
        {
            "name": "Смешанный текст",
            "text": "Hello привет world мир",
            "ref_text": "This is a test"
        },
        {
            "name": "Длинный английский",
            "text": "The quick brown fox jumps over the lazy dog",
            "ref_text": "This is a reference text for voice cloning"
        }
    ]
    
    ref_audio = "test_ref.wav"
    
    if not os.path.exists(ref_audio):
        print(f"❌ Референсный файл не найден: {ref_audio}")
        return
    
    print("🔍 Диагностика синхронизации текста и озвучки")
    print(f"🎵 Референсное аудио: {ref_audio}")
    print()
    
    for i, case in enumerate(test_cases, 1):
        print(f"🧪 Тест {i}: {case['name']}")
        print(f"📝 Входной текст: '{case['text']}'")
        print(f"📄 Референсный текст: '{case['ref_text']}'")
        
        # Проверяем языковое определение
        language = tts.detect_language(case['text'])
        print(f"🌍 Определенный язык: {language}")
        
        # Проверяем предобработку
        processed_text = tts.preprocess_text_for_tts(case['text'])
        print(f"⚙️ Обработанный текст: '{processed_text}'")
        
        try:
            output_path = tts.synthesize_speech(
                text=case['text'],
                ref_audio_path=ref_audio,
                ref_text=case['ref_text'],
                cross_fade_duration=0.15,
                speed=1.0,
                silence_duration_ms=200,
                target_rms=0.3,
                sway_sampling_coef=-1,
                cfg_strength=2,
                nfe_step=32,
                fix_duration=None,
                remove_silence=False,
                seed=None
            )
            
            if output_path and os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"✅ Результат: {output_path} ({file_size} bytes)")
            else:
                print(f"❌ Ошибка синтеза")
                
        except Exception as e:
            print(f"❌ Ошибка: {e}")
        
        print("-" * 50)
        print()

if __name__ == "__main__":
    debug_text_sync()
