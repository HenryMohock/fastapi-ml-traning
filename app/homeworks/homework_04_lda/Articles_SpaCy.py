import pandas as pd
import spacy

# Загрузка предварительно обученной модели SpaCy для английского языка
# python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

# Загрузка данных из файла CSV
df = pd.read_csv("../../data/articles.csv")


# Определение функции для извлечения ключевых слов из текста с использованием SpaCy
def extract_keywords(text):
    doc = nlp(text)
    # Извлечение существительных и прилагательных в качестве ключевых слов
    keywords = [token.lemma_ for token in doc if token.pos_ in ["NOUN", "PROPN", "ADJ"]]
    return keywords


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
