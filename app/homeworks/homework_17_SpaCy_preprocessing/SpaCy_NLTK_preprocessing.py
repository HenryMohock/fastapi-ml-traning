import signal
import pandas as pd
import re
from collections import Counter
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import spacy
import time

# Регистрация обработчика сигнала прерывания
signal.signal(signal.SIGINT, signal.default_int_handler)

# Параметры текста
encoding = 'utf8'
lang = 'english'
len_text = 2000000  # 65521550  # 200000

# Путь к файлу CSV
file_path = "../../data/IMDB Dataset.csv"

# Имя извлекаемой колонки файла CSV
column_name = 'review'

# Загрузка данных из файла CSV
data = pd.read_csv(file_path, encoding=encoding)

# Извлечение колонки column_name и объединение ее в текстовую переменную (внутреннее свойство)
text = ' '.join(data[column_name].astype(str))[:len_text]


def format_time(seconds):
    """
    Форматирование секунд к виду ЧЧ:ММ:СС

      Возвращает строку отформатированного времени
    """
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return "{:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds))


def spacy_preprocessing(text):
    # Загрузка модели SpaCy
    nlp = spacy.load("en_core_web_sm")

    # Функция для обработки части текста
    def process_text_chunk(chunk):
        doc = nlp(chunk)
        tokens = [token.text for token in doc if not token.is_stop and token.is_alpha]
        return tokens

    # Функция для лемматизации части текста
    def lemmatize_text_chunk(chunk):
        doc = nlp(chunk)
        return [token.lemma_ for token in doc]

    # Обработка текста с помощью регулярного выражения
    regex_patterns = r'blah|_|fffc|br|oz|< />|<br />|\d+|\W+'
    text = re.sub(regex_patterns, ' ', text)

    # Разбиваем текст на части по 100000 символов
    chunk_size = 100000
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    # Обрабатываем каждую часть и объединяем результаты
    all_tokens = []
    for chunk in chunks:
        all_tokens.extend(process_text_chunk(chunk))

    # Подсчет частоты слов
    word_freq = Counter(all_tokens)

    # Определение редких слов (слова, которые встречаются менее 5 раз)
    rare_words = {word for word, count in word_freq.items() if count < 5}

    # Удаление редких слов
    filtered_tokens = [word for word in all_tokens if word not in rare_words]

    # Разбиваем текст на части по 100000 символов для лемматизации
    filtered_text = ' '.join(filtered_tokens)
    filtered_chunks = [filtered_text[i:i + chunk_size] for i in range(0, len(filtered_text), chunk_size)]

    # Лемматизация каждой части и объединение результатов
    lemmatized_tokens = []
    for chunk in filtered_chunks:
        lemmatized_tokens.extend(lemmatize_text_chunk(chunk))

    # Объединение обработанных токенов в строку
    cleaned_text = ' '.join(lemmatized_tokens)

    # Сохранение обработанного текста в файл
    try:
        output_file_path = "../../data/spacy_processed_text.txt"
        with open(output_file_path, 'w', encoding=encoding) as f:
            f.write(cleaned_text)
        return True
    except Exception as e:
        print(f"Не удалось записать {output_file_path}: {str(e)}")
        return False


def nltk_preprocessing(text):
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    # Преобразование текста к нижнему регистру
    text = text.lower()

    # Токенизация текста с использованием регулярного выражения
    regex_patterns = r'blah|_|fffc|br|oz|< />|<br />|\d|\w+'
    tokenizer = RegexpTokenizer(regex_patterns)
    tokens = tokenizer.tokenize(text)

    # Токенизация текста
    tokens = word_tokenize(text)

    # Загрузка стоп-слов
    stop_words = set(stopwords.words('english'))

    # Удаление стоп-слов
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]

    # Подсчет частоты слов
    word_freq = Counter(tokens)

    # Определение редких слов (слова, которые встречаются менее 5 раз)
    rare_words = {word for word, count in word_freq.items() if count < 5}

    # Удаление редких слов
    tokens = [word for word in tokens if word not in rare_words]

    # Лемматизация
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Объединение обработанных токенов в строку
    cleaned_text = ' '.join(tokens)

    # Сохранение обработанного текста в файл
    try:
        output_file_path = "../../data/nltk_processed_text.txt"
        with open(output_file_path, 'w', encoding=encoding) as f:
            f.write(cleaned_text)
        return True
    except Exception as e:
        print(f"Не удалось записать {output_file_path}: {str(e)}")
        return False


start_time = time.time()
if nltk_preprocessing(text):
    print("Текст успешно обработан (NLTK) и сохранен в файл.")
print("~ Время обработки с использованием NLTK:", format_time(time.time() - start_time))


start_time = time.time()
if spacy_preprocessing(text):
    print("Текст успешно обработан (SpaCy) и сохранен в файл.")
print("~ Время обработки с использованием SpaCy:", format_time(time.time() - start_time))
