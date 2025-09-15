#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Создание тестового аудио файла для F5-TTS
"""

import numpy as np
import soundfile as sf
import os

def create_test_audio():
    """Создает тестовый аудио файл"""
    
    # Создаем простой синусоидальный сигнал
    sample_rate = 22050
    duration = 3.0  # 3 секунды
    frequency = 440  # A4 нота
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(2 * np.pi * frequency * t) * 0.3
    
    # Добавляем небольшой шум для реалистичности
    noise = np.random.normal(0, 0.01, len(audio))
    audio = audio + noise
    
    # Создаем папку если не существует
    os.makedirs("test_audio", exist_ok=True)
    
    # Сохраняем файл
    output_path = "test_audio/test_ref.wav"
    sf.write(output_path, audio, sample_rate)
    
    print(f"✅ Тестовый аудио файл создан: {output_path}")
    print(f"📊 Параметры: {sample_rate} Hz, {duration} сек, {len(audio)} сэмплов")
    
    return output_path

if __name__ == "__main__":
    create_test_audio()