import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter

# Загрузка данных из файла CSV
df = pd.read_csv("articles.csv")

# Инициализация стеммера и списка стоп-слов
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# Определение функции для извлечения ключевых слов из текста
def extract_keywords(text):
    # Токенизация текста
    words = word_tokenize(text.lower())

    # Удаление стоп-слов и пунктуации
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]

    # Лемматизация слов
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

    # Подсчет частоты встречаемости слов
    word_freq = Counter(lemmatized_words)

    # Выбор 5 самых часто встречаемых слов в качестве ключевых
    keywords = [word for word, _ in word_freq.most_common(5)]

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
    print(f"Ключевые слова: {keywords_str}")
    print()

