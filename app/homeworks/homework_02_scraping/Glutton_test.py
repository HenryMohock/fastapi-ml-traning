import requests
import csv
import json
import re
import tkinter as tk
from tkinter import filedialog
from bs4 import BeautifulSoup


def ends_with_digit(url_whiskey_price: str) -> bool:
    pattern = r'^https:\/\/.*viski\/\d{5}\/$'
    if re.match(pattern, url_whiskey_price):
        return True
    else:
        return False


url_whiskey = "https://produkty24.com.ua/alkogol-i-napitki/alkogolnyie-napitki/viski/pp/96/"

# Запит у користувача обрати папку
root = tk.Tk()
root.withdraw()
data_folder = filedialog.askdirectory()

response = requests.get(url_whiskey)
response.raise_for_status()  # Перевірка на помилки
soup = BeautifulSoup(response.text, 'html.parser')

# Вилучення заголовка сторінки
title = soup.title.string
print(f"Заголовок сторінки: {title}")

# Вилучення всіх посилань на сторінці
links = [link.get('href') for link in soup.find_all('a')]
print(f"Посилання на сторінці: {links}")

print("==========Web Scraping===========")

# Шапка таблиці
data = [['Name', 'Price', 'Cents', 'Valute']]

for url in links:
    if type(url) is str:
        if ends_with_digit(url):
            print(f"Адреса сторінки whiskey: {url}")

            response = requests.get(url)
            response.raise_for_status()  # Перевірка на помилки

            # Створення об'єкту BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Вилучення тексту з елемента <h1> з класом "gray padding-barcode"
            h1_element = soup.find('h1', class_='gray padding-barcode')
            product_name = h1_element.text.strip()
            print(f"Назва: {product_name}")

            # Вилучення тексту з елемента <span> з ідентифікатором "for-item"
            span_element = soup.find('span', id='for-item')

            # Вилучення тексту з дочірніх елементів
            price = span_element.find(class_='price').text.strip()
            cents = span_element.find(class_='cents').text.strip()
            valute = span_element.find(class_='valute').text.strip()

            print(f"Ціла частина: {price}")
            print(f"Дробна частина: {cents}")
            print(f"Одиниця виміру: {valute}")
            print("================================================")

            data.append(
                [product_name, price, cents, valute]
            )

# Шлях до файлу csv з вилученими даними
file_csv = data_folder + 'data_whiskey.csv'

# Запис даних в файл csv
try:
    with open(file_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print(f"Дані успішно записані у файл: {file_csv}")
except:
    print(f"Не вдалося записати дані до файлу: {file_csv}")

# Шлях до файлу json з вилученими даними
file_json = data_folder + 'data_whiskey.json'

# Запис даних в файл json
try:
    with open(file_json, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False)
    print(f"Дані успішно записані у файл: {file_json}")
except:
    print(f"Не вдалося записати дані до файлу: {file_json}")
