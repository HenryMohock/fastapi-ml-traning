import pandas as pd
import yake

# Загрузка данных из файла CSV
df = pd.read_csv("../../data/articles.csv")

# Инициализация объекта YAKE
kw_extractor = yake.KeywordExtractor()


# Определение функции для извлечения ключевых слов из текста с использованием YAKE
def extract_keywords(text):
    # Извлечение ключевых слов с помощью YAKE
    keywords = kw_extractor.extract_keywords(text)
    first_keywords = [keyword[0] for keyword in keywords]
    return first_keywords
    # return keywords[:5]  # Возвращаем только первые 5 слов


# Группировка статей по темам и определение ключевых слов для каждой темы
themes_keywords = {}
for _, article in df.iterrows():
    theme = article['title']  # Пусть название статьи будет темой
    text = article['text']

    if theme not in themes_keywords:
        themes_keywords[theme] = []

    keywords = extract_keywords(text)
    themes_keywords[theme].extend(keywords)

# Вывод результатов
for theme, keywords in themes_keywords.items():
    unique_keywords = list(set(keywords[:5]))  # Нас интересуют не повторяющиеся слова
    keywords_str = ", ".join(unique_keywords)
    print(f"Тема: {theme}")
    print(f"Ключевые слова: {keywords_str}")
    print()
