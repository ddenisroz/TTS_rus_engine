# 🚀 Руководство по подключению TTS_rus_engine

## 📋 Системные требования

### Минимальные требования:
- **OS**: Windows 10/11, Linux Ubuntu 18.04+, macOS 10.15+
- **Python**: 3.8 или выше
- **RAM**: 8GB (рекомендуется 16GB)
- **GPU**: NVIDIA с поддержкой CUDA 12.4+ (рекомендуется)
- **Место на диске**: 5GB свободного места

### Рекомендуемые требования:
- **GPU**: NVIDIA RTX 3060 или выше
- **RAM**: 16GB+
- **CPU**: Intel i5/AMD Ryzen 5 или выше

## 🔧 Пошаговая установка

### Шаг 1: Подготовка системы

#### Windows:
1. Установите Python 3.8+ с [python.org](https://www.python.org/downloads/)
2. Установите Git с [git-scm.com](https://git-scm.com/download/win)
3. Установите NVIDIA CUDA Toolkit 12.4+ с [developer.nvidia.com](https://developer.nvidia.com/cuda-downloads)

#### Linux (Ubuntu/Debian):
```bash
# Обновление системы
sudo apt update && sudo apt upgrade -y

# Установка Python и Git
sudo apt install python3 python3-pip python3-venv git -y

# Установка CUDA (если есть NVIDIA GPU)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-ubuntu2004-12-4-local_12.4.0-550.54.14-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-4-local_12.4.0-550.54.14-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

#### macOS:
```bash
# Установка Homebrew (если не установлен)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Установка Python и Git
brew install python@3.9 git
```

### Шаг 2: Клонирование репозитория

```bash
# Клонирование репозитория
git clone https://github.com/your-username/TTS_rus_engine.git
cd TTS_rus_engine
```

### Шаг 3: Создание виртуального окружения

```bash
# Создание виртуального окружения
python -m venv f5tts_env

# Активация виртуального окружения
# Windows:
f5tts_env\Scripts\activate
# Linux/macOS:
source f5tts_env/bin/activate
```

### Шаг 4: Установка зависимостей

```bash
# Обновление pip
python -m pip install --upgrade pip

# Установка PyTorch с CUDA поддержкой
pip install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# Установка остальных зависимостей
pip install -r requirements.txt
```

### Шаг 5: Проверка установки

```bash
# Проверка CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Запуск тестового синтеза
python russian_tts.py
```

## 🎯 Первый запуск

### Веб-интерфейс:
```bash
python russian_web.py
```
Откройте браузер: http://localhost:7864

### Программное использование:
```python
from russian_tts import RussianTTS

# Инициализация (займет ~10 секунд при первом запуске)
tts = RussianTTS()

# Синтез речи
audio_path = tts.synthesize_speech(
    text="Привет! Это тестовый синтез речи.",
    ref_audio="test_audio/test_ref.wav",
    ref_text="Секреты всегда рядом, Скуф. Нужно лишь тихо прислушаться и услышать их."
)

print(f"Аудио сохранено: {audio_path}")
```

## 🔧 Настройка

### Переменные окружения:
```bash
# Для оптимизации памяти PyTorch
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# Для отключения CUDA (если нужно использовать CPU)
export CUDA_VISIBLE_DEVICES=""
```

### Конфигурация модели:
```python
# В russian_tts.py можно изменить параметры:
tts = RussianTTS(
    enable_accent=True,        # Включить ударения
    accent_model_size="turbo", # Размер модели ударений
    ode_method="euler",        # Метод ODE
    use_ema=True              # Использовать EMA
)
```

## 🐛 Решение проблем

### Проблема: CUDA не найдена
```bash
# Проверьте установку CUDA
nvidia-smi
nvcc --version

# Переустановите PyTorch с CUDA
pip uninstall torch torchaudio
pip install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

### Проблема: Недостаточно памяти GPU
```python
# Уменьшите batch size или используйте CPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Отключить GPU
```

### Проблема: Медленная загрузка моделей
```bash
# Модели кэшируются в f5_tts_cache/
# При первом запуске они скачиваются (~2GB)
# Последующие запуски будут быстрее
```

### Проблема: Ошибки с RUAccent
```bash
# Переустановите ruaccent
pip uninstall ruaccent
pip install ruaccent
```

## 📊 Производительность

### Время загрузки:
- **Первый запуск**: ~30 секунд (скачивание моделей)
- **Последующие запуски**: ~8 секунд

### Время синтеза:
- **Короткий текст** (< 50 символов): ~2-3 секунды
- **Средний текст** (50-200 символов): ~5-10 секунд
- **Длинный текст** (> 200 символов): ~15-30 секунд

### Использование памяти:
- **GPU**: 2-4GB VRAM
- **RAM**: 4-8GB

## 🔄 Обновление

```bash
# Обновление до последней версии
git pull origin main
pip install -r requirements.txt --upgrade
```

## 📞 Поддержка

Если у вас возникли проблемы:

1. Проверьте [Issues](https://github.com/your-username/TTS_rus_engine/issues)
2. Создайте новый Issue с описанием проблемы
3. Приложите логи ошибок и информацию о системе

## 🎉 Готово!

Теперь вы можете использовать TTS_rus_engine для синтеза русской речи!
