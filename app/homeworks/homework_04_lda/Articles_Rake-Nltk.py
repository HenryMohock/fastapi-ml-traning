import pandas as pd
from rake_nltk import Rake

# Загрузка данных из файла CSV
df = pd.read_csv("../../data/articles.csv")

# Инициализация объекта Rake
r = Rake()


# Определение функции для извлечения ключевых слов из текста с использованием Rake-Nltk
def extract_keywords(text):
    # Извлечение ключевых слов
    r.extract_keywords_from_text(text)
    # Получение списка ключевых слов
    keywords = r.get_ranked_phrases()[:5]  # Возвращаем только первые 5 ключевых фраз
    # keywords = r.get_word_frequency_distribution()  # Ключевые слова, распределение частоты
    # keywords = r.get_word_degrees()  # Ключевые слова? степени
    # keywords = r.get_ranked_phrases_with_scores()  # Ключевые слова
    return keywords


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
    unique_keywords = list(set(keywords))  # Нас интересуют не повторяющиеся слова
    keywords_str = ", ".join(unique_keywords)
    print(f"Тема: {theme}")
    print(f"Ключевые фразы: {keywords_str}")
    print()
