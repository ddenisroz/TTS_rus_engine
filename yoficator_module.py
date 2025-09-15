#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль для ёфикации русского текста
Основан на python-yoficator: https://github.com/Text-extend-tools/python-yoficator
"""

import os
import re
import codecs
from pathlib import Path

class Yoficator:
    """Ёфикатор для русского текста"""
    
    def __init__(self, dictionary_path=None):
        """
        Инициализация ёфикатора
        
        Args:
            dictionary_path: Путь к файлу словаря yo.dat
        """
        if dictionary_path is None:
            # Ищем файл yo.dat в папке python-yoficator
            current_dir = Path(__file__).parent
            dictionary_path = current_dir / "python-yoficator" / "yo.dat"
        
        self.dictionary_path = dictionary_path
        self.dictionary = {}
        self.splitter = re.compile(r'(\s+|\w+|\W+|\S+)', re.UNICODE)
        
        # Загружаем словарь
        self._load_dictionary()
    
    def _load_dictionary(self):
        """Загружает словарь для ёфикации"""
        try:
            with codecs.open(self.dictionary_path, "r", "utf-8") as f:
                for line in f:
                    if not "*" in line:
                        cline = line.rstrip('\n')
                        if "(" in cline:
                            bline, sline = cline.split("(")
                            sline = re.sub(r'\)', '', sline)
                        else:
                            bline = cline
                            sline = ""
                        
                        if "|" in sline:
                            ssline = sline.split("|")
                            for ss in ssline:
                                value = bline + ss
                                key = re.sub(r'ё', 'е', value)
                                self.dictionary[key] = value
                        else:
                            value = bline
                            key = re.sub(r'ё', 'е', value)
                            self.dictionary[key] = value
        except FileNotFoundError:
            print(f"Предупреждение: Словарь {self.dictionary_path} не найден. Ёфикация отключена.")
            self.dictionary = {}
    
    def yoficate(self, text):
        """
        Ёфицирует русский текст
        
        Args:
            text: Исходный текст
            
        Returns:
            Ёфицированный текст
        """
        if not self.dictionary:
            return text
        
        # Разбиваем текст на токены
        tokens = self.splitter.findall(text)
        result = []
        
        for token in tokens:
            if token in self.dictionary:
                result.append(self.dictionary[token])
            else:
                # Дополнительные правила для слов, которых нет в словаре
                yoficated_token = self._apply_additional_rules(token)
                result.append(yoficated_token)
        
        return ''.join(result)
    
    def _apply_additional_rules(self, word: str) -> str:
        """Применяет дополнительные правила ёфикации для слов, которых нет в словаре."""
        if not word or not word.isalpha():
            return word
        
        # Правила для замены "е" на "ё" в определенных позициях
        import re
        
        # 1. "телка" -> "тёлка"
        if word.lower() == "телка":
            return word.replace("е", "ё")
        
        # 2. "осел" -> "осёл" 
        if word.lower() == "осел":
            return word.replace("е", "ё")
        
        # 3. "еще" -> "ещё"
        if word.lower() == "еще":
            return "ещё"
        
        # 4. "ее" -> "её" (местоимение)
        if word.lower() == "ее":
            return "её"
        
        # 5. "произнес" -> "произнёс"
        if word.lower() == "произнес":
            return word.replace("е", "ё")
        
        # 6. Общие правила для слов, заканчивающихся на "ел" (кроме исключений)
        if word.lower().endswith("ел") and len(word) > 3:
            # Исключения
            exceptions = ["медведь", "привет", "как", "дела", "удел", "предел", "умел", "смел", "дела"]
            if word.lower() not in exceptions:
                return word.replace("ел", "ёл")
        
        # 7. Слова с "е" в корне, заканчивающиеся на "а" (женский род)
        if word.lower().endswith("а") and "е" in word.lower()[:-1]:
            # Исключения
            exceptions = ["телка", "щелка", "мелка", "желтка", "дела"]
            if word.lower() not in exceptions:
                # Заменяем "е" на "ё" в корне
                return re.sub(r'е([^ё]*)$', r'ё\1', word)
        
        return word

# Создаем глобальный экземпляр для удобства использования
yoficator = Yoficator()

def yoficate_text(text):
    """
    Удобная функция для ёфикации текста
    
    Args:
        text: Исходный текст
        
    Returns:
        Ёфицированный текст
    """
    return yoficator.yoficate(text)

if __name__ == "__main__":
    # Тестирование
    test_text = "елка, медведь, осел, все, еще, ее"
    result = yoficate_text(test_text)
    print(f"Исходный текст: {test_text}")
    print(f"Ёфицированный: {result}")
