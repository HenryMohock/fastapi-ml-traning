import requests
import time
from typing import List
import signal
import pandas as pd
import json


def format_time(seconds):
    """
    Форматирование секунд к виду ЧЧ:ММ:СС

      Возвращает строку отформатированного времени
    """
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return "{:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds))


def get_url(server, post_method):
    slash = '/'
    return f"{server}{post_method}{slash}"


def post_request(url: str, data: str, output_file_path: str):
    try:
        response = requests.post(url, json={'data': data})
        if response.status_code == 200:
            response_data = response.json()
            with open(output_file_path, "w", encoding="utf-8") as json_file:
                json.dump(response_data, json_file, ensure_ascii=False, indent=4)
                print(f"Текст успешно обработан и сохранен в файл {output_file_path}")
        else:
            print(f'Ошибка запроса на сервер: {response.text}')

    except requests.exceptions.HTTPError as http_err:
        print(f"Возникла ошибка HTTP: {http_err}")


# Регистрация обработчика сигнала прерывания
signal.signal(signal.SIGINT, signal.default_int_handler)

# Параметры текста
encoding = 'utf8'
lang = 'english'
len_text = 65521550  # 65521550 - это длина в символах всего текста. Можно брать короче, например: 200000

# Путь к файлу CSV
file_path = "../../data/IMDB Dataset.csv"

# Имя извлекаемой колонки файла CSV
column_name = 'review'

# Загрузка данных из файла CSV
data = pd.read_csv(file_path, encoding=encoding)

# Извлечение колонки column_name и объединение ее в текстовую переменную
text = ' '.join(data[column_name].astype(str))[:len_text]

if __name__ == "__main__":
    # Адрес сервера
    host = '127.0.0.1'
    port = '8000'
    server = 'http://' + host + ':' + port + '/'
    sentences = text

    # NLTK
    # Метод на сервере возвращающий результат запроса
    post_method = 'nltk_preprocessing'
    # Получаем строку запроса
    url = get_url(server, post_method)  # "http://127.0.0.1:8000/nltk_preprocessing/"

    print()
    print(f"NLTK: Запрос на сервер: {url}")
    output_file_path = "../../data/nltk_processed_text.json"
    start_time = time.time()
    post_request(url, sentences, output_file_path)
    print("~ Время обработки с использованием NLTK:", format_time(time.time() - start_time))

    # SPACY
    # Метод на сервере возвращающий результат запроса
    post_method = 'spacy_preprocessing'
    # Получаем строку запроса
    url = get_url(server, post_method)  # "http://127.0.0.1:8000/spacy_preprocessing/"

    print()
    print(f"SPACY: Запрос на сервер: {url}")
    output_file_path = "../../data/spacy_processed_text.json"
    start_time = time.time()
    post_request(url, sentences, output_file_path)
    print("~ Время обработки с использованием SPACY:", format_time(time.time() - start_time))
