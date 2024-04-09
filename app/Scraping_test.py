import requests
import csv
import tkinter as tk
from tkinter import filedialog
from bs4 import BeautifulSoup


def check_page_availability(url_index):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print(f"Сторінка {url_index} доступна.")
            return True
        else:
            print(f"Помилка: {response.status_code} - сторінка {url_index} не доступна.")
            return False
    except requests.RequestException as e:
        print(f"Помилка при спробі доступу до сторінки {url_index}: {e}")
        return False


url = "http://127.0.0.1:8000/static/index.html"

# Запит у користувача обрати папку
root = tk.Tk()
root.withdraw()
data_folder = filedialog.askdirectory()

if check_page_availability(url):
    print("Локальний веб-сервер запущено.")
    response = requests.get(url)
    response.raise_for_status()  # Перевірка на помилки
    soup = BeautifulSoup(response.text, 'html.parser')
else:
    print("Локальний веб-сервер не запущено.")
    path_file = data_folder + 'index.html'
    with open(path_file, 'r', encoding='utf-8') as file:
        html_content = file.read()
    soup = BeautifulSoup(html_content, 'html.parser')

# Вилучення заголовка сторінки
title = soup.title.string
print(f"Заголовок сторінки: {title}")

# Вилучення всіх посилань на сторінці
links = [link.get('href') for link in soup.find_all('a')]
print(f"Посилання на сторінці: {links}")

print("==========Web Scraping===========")

# Шапка таблиці csv
data = [['Вакансія', 'Компанія', 'Опис']]

for url in links:

    if url != 'https://www.realpython.com':
        print(f"Адреса сторінки: {url}")

        response = requests.get(url)
        response.raise_for_status()  # Перевірка на помилки

        # Створення об'єкту BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Вилучення класу box
        boxs = soup.find_all('div', class_='box')

        for box in boxs:
            print(f"Вакансія: {box.h1.text}")  # Вилучення назви віскі
            print(f"Компанія: {box.h2.text}")  # Вилучення ціни
            print(f"Опис: {box.p.text}")  # Вилучення валюти
            print("================================================")
            data.append(
                [box.h1.text, box.h2.text, box.p.text]
            )

# Шлях до файлу csv з вилученими даними
file_path = data_folder + 'data.csv'

# Запис даних в файл csv
try:
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print(f"Дані успішно записані у файл: {file_path}")
except:
    print(f"Не вдалося записати дані до файлу: {file_path}")
