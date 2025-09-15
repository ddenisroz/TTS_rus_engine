#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скачивание правильного vocab.txt для F5-TTS Russian
"""

import logging
from pathlib import Path
from huggingface_hub import hf_hub_download

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_vocab():
    """Скачивает правильный vocab.txt для F5-TTS Russian"""
    try:
        logger.info("Скачиваем vocab.txt из Misha24-10/F5-TTS_RUSSIAN...")
        
        # Создаем директорию для модели
        model_dir = Path("models/russian/F5TTS_v1_Base_v2")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Скачиваем vocab.txt
        vocab_path = hf_hub_download(
            repo_id="Misha24-10/F5-TTS_RUSSIAN",
            filename="F5TTS_v1_Base/vocab.txt",
            cache_dir="f5_tts_cache"
        )
        
        # Копируем в нужную директорию
        import shutil
        target_path = model_dir / "vocab.txt"
        shutil.copy(vocab_path, target_path)
        
        logger.info(f"✅ vocab.txt скачан и сохранен в: {target_path}")
        
        # Проверяем размер файла
        file_size = target_path.stat().st_size
        logger.info(f"📊 Размер файла: {file_size} байт")
        
        # Читаем первые несколько строк для проверки
        with open(target_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:10]
            logger.info("📝 Первые 10 строк vocab.txt:")
            for i, line in enumerate(lines, 1):
                logger.info(f"  {i}: {line.strip()}")
        
        return str(target_path)
        
    except Exception as e:
        logger.error(f"❌ Ошибка при скачивании vocab.txt: {e}")
        return None

if __name__ == "__main__":
    result = download_vocab()
    if result:
        print(f"✅ vocab.txt успешно скачан: {result}")
    else:
        print("❌ Ошибка при скачивании vocab.txt")
