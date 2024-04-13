import pandas as pd
from textblob import TextBlob

# Загрузка данных из файла CSV
df = pd.read_csv("articles.csv")


# Определение функции для извлечения ключевых слов из текста с использованием TextBlob
def extract_keywords(text):
    blob = TextBlob(text)
    # Извлечение ключевых слов
    keywords = blob.words.lower().singularize()
    return keywords[:5]  # Возвращаем только первые 5 слов


# Группировка статей по темам и определение ключевых слов для каждой темы
themes_keywords = {}
for _, article in df.iterrows():
    theme = article['title']  # Название статьи это тема
    text = article['text']

    if theme not in themes_keywords:
        themes_keywords[theme] = []

    keywords = extract_keywords(text)
    themes_keywords[theme].extend(keywords)

# Вывод результатов
for theme, keywords in themes_keywords.items():
    unique_keywords = list(set(keywords)) # Нас интересуют не повторяющиеся слова
    keywords_str = ", ".join(unique_keywords)
    print(f"Тема: {theme}")
    print(f"Ключевые слова: {keywords_str}")
    print()
