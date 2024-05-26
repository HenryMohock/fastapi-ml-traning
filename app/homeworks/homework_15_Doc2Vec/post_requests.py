import requests
import time
from typing import List


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


def post_request(url: str, sentences: List[str], max_clusters: int = 10):
    start_time = time.time()
    try:
        response = requests.post(url, json={"sentences": sentences, "max_clusters": max_clusters})
        print("Время получения данных с сервера:", format_time(time.time() - start_time))
        response.raise_for_status()
        if response.status_code == 200:
            answer = response.json()
            for key, sentences in answer.items():
                print(f"Key: {key}")
                for sentence in sentences:
                    print(f"  - {sentence}")
        else:
            print('Error server:', response.text)
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")


if __name__ == "__main__":
    # Адрес сервера
    host = '127.0.0.1'
    port = '8000'
    server = 'http://' + host + ':' + port + '/'
    # Метод на сервере возвращающий результат запроса
    post_method = 'group_sentences'
    # Получаем строку запроса
    url = get_url(server, post_method)  #"http://127.0.0.1:8000/group_sentences/"

    sentences = [
        "The stock market is down today.",
        "The weather is sunny and warm.",
        "Investors are concerned about the economy.",
        "It might rain tomorrow.",
        "The Federal Reserve raised interest rates.",
        "Had the front panel of my car stereo stolen.",
    ]

    print(f"Запрос на сервер: {url}")
    post_request(url, sentences)


