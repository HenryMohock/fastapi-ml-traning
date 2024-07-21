import requests


def get_url(server, post_method):
    slash = '/'
    return f"{server}{post_method}{slash}"


# Адрес сервера
host = '127.0.0.1'
port = '8000'
server = 'http://'+host+':'+port+'/'
# Метод на сервере возвращающий результат запроса
post_method = 'translator_nllb'

# URL эндпоинта переводчика
url = get_url(server=server, post_method=post_method)  # "http://127.0.0.1:8000/translator_nllb/"

# Данные для отправки запросов
input_text = "Hello, how are you?"
source_language = "eng_Latn"
target_language = "ukr_Cyrl"

print(f"\nТекст для перекладу: {input_text}")
print(f"Мова тексту: {source_language}\n")

# Создание JSON запроса
data = {
    "input_text": input_text,
    "source_language": source_language,
    "target_language": target_language
}

try:
    # Отправка POST запроса к эндпоинту
    response = requests.post(url, json=data)

    # Проверка статуса ответа
    if response.status_code == 200:
        # Получение JSON данных из ответа
        response_data = response.json()
        print("Текст перекладу:", response_data['response_text'])
        print("Мова перекладу:", response_data['target_language'])
    else:
        print(f"Помилка запиту з кодом статусу {response.status_code}")
        print("Текст відповіді:", response.text)  # Вывод полного текста ответа
except Exception as e:
    print(f"An error occurred: {e}")
