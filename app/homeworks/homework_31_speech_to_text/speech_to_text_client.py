import requests


# Функція для формування URL-адреси
def get_url(server, post_method):
    """
    Функція get_url(server, post_method) приймає сервер та метод POST і формує URL-адресу.

    Параметри:
        server (str): Базова URL-адреса сервера.
        post_method (str): Метод POST, який додається до URL-адреси сервера.

    Повертає:
        str: Сформована URL-адреса, яка складається з базової URL-адреси сервера, методу POST та символу '/'.

    Функція працює таким чином:
    1. Встановлює символ '/'.
    2. Формує і повертає рядок, який складається з базової URL-адреси сервера, методу POST та символу '/'.
    """

    slash = '/'
    return f"{server}{post_method}{slash}"


def send_audio_file(file_path):
    """
    Функція send_audio_file завантажує аудіофайл на сервер і отримує відповідь у форматі JSON.

    Параметри:
    file_path (str): шлях до аудіофайлу, який потрібно надіслати на сервер.
    Операції:
    Викликає функцію get_url для отримання повної URL-адреси сервера з базовою URL та частиною URL для POST-запиту.
    Відкриває файл за зазначеним шляхом (file_path) у режимі бінарного читання ("rb").
    Надсилає POST-запит на сервер, включаючи файл у тілі запиту.
    Повертає відповідь сервера у форматі JSON.

    Повертає:
    dict: JSON-об'єкт, що містить відповідь сервера.
    """

    url = get_url(server=server, post_method=post_method)  # "http://127.0.0.1:8000/speech_to_text/"
    files = {"file": open(file_path, "rb")}
    response = requests.post(url, files=files)
    return response.json()

# Адреса серверу
host = '127.0.0.1'
port = '8000'
server = 'http://'+host+':'+port+'/'

# Метод на сервері, що повертає результат запиту
post_method = 'speech_to_text'

# URL ендпоінта перетворювача аудіо у текст
url = get_url(server=server, post_method=post_method)

# Надсилання запитів з аудіо-файлами на сервер
base_path = "../../data/wav/arctic_a"
for i in range(1, 11):
    audio_file_path = f"{base_path}{i:04d}.wav"
    print(f"\nНадсилання файлу    {audio_file_path} на сервер...")

    result = send_audio_file(audio_file_path)

    # Разбор JSON-ответа
    original_text = result.get("original_text", "Оригінальний текст не знайдено!")
    translated_text = result.get("translated_text", "Перекладений текст не знайдено!")

    print(f"Результат для файлу {audio_file_path}:")
    print(f"Оригінальний текст     (en): {original_text}")
    print(f"Переклад на українську (uk): {translated_text}")
    print("-" * 50)
