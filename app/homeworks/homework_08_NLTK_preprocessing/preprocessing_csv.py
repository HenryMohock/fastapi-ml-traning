import re
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import spacy
import time


# Обработка текста
# ===================================================================
class CsvPreProcessing:
    def __init__(self, file_path, column_name, regular_expression, threshold=5, method='lemmatization'):
        self.file_path = file_path
        self.column_name = column_name
        self.regular_expression = regular_expression
        self.threshold = threshold
        self.method = method

    def reading_csv(self):
        """
        Читает csv-файл и извлекает колонку 'review' в текстовую переменную.

        Возвращает:
            str: Текст из колонки 'review'.
        """
        # Загрузка данных из файла CSV
        data = pd.read_csv(self.file_path, encoding='utf8')

        # Извлечение колонки column_name и объединение ее в текстовую переменную
        text = ' '.join(data[self.column_name].astype(str))

        return text

    def text_to_lower(self, text):
        """
        Преобразует текст к нижнему регистру.

        Параметры:
            text (str): Исходный текст.

        Возвращает:
            str: Текст в нижнем регистру.
        """
        return text.lower()

    def remove_punctuation(self, text):
        """
        Удаляет всю пунктуацию из текста.

        Параметры:
            text (str): Исходный текст.

        Возвращает:
            str: Текст без пунктуации.
        """

        # Удаление знаков пунктуации с помощью регулярного выражения
        text = re.sub(r'[^\w\s]', '', text)

        return text

    def remove_other_characters(self, text):
        """
        Удаляет все символы, соответствующие регулярному выражению из текста.

        Параметры:
            text (str): Исходный текст.

        Возвращает:
            str: Текст без указанных символов.
        """

        # Удаление символов, соответствующих регулярному выражению
        text = re.sub(self.regular_expression, '', text)

        # Замена по итогу множественных пробелов на один пробел
        text = re.sub(r'\s+', ' ', text)

        return text

    def remove_digits(self, text):
        """
        Удаляет все цифры из текста

        Параметры:
            text (str): Исходный текст.

        Возвращает:
            str: Текст без указанных символов.
        """

        # Удаление всех цифр из текста
        text = re.sub(r'\d', '', text)

        return text

    def remove_stopwords(self, text):
        """
        Удаляет все стоп-слова из текста

        Параметры:
            text (str): Исходный текст.

        Возвращает:
            str: Текст без указанных символов.
        """

        # Загрузка английских стоп-слов из NLTK
        stop_words = set(stopwords.words('english'))

        # Удаление стоп-слов
        words = text.split()
        filtered_text = [word for word in words if word.lower() not in stop_words]

        return ' '.join(filtered_text)

    def remove_rare_words(self, text):
        """
        Удаляет все редкие слова из текста в соответствии с порогом встречаемости

        Параметры:
            text (str): Исходный текст.

        Возвращает:
            str: Текст без указанных слов.
        """
        # Разделение текста на слова
        words = text.split()

        # Подсчет частоты встречаемости слов
        word_counts = Counter(words)

        # Удаление редко встречающихся слов
        filtered_words = [word for word in words if word_counts[word] >= self.threshold]

        # Сбор слов обратно в текст
        filtered_text = ' '.join(filtered_words)

        return filtered_text

    def stem_text(self, text):
        """
        Производит стемминг текста

        Параметры:
            text (str): Исходный текст.

        Возвращает:
            str: Текст стеммированных слов.
        """
        # Создание объекта стеммера
        stemmer = PorterStemmer()

        # Стемминг каждого слова в тексте
        stemmed_words = [stemmer.stem(word) for word in text.split()]

        # Сбор стеммированных слов обратно в текст
        stemmed_text = ' '.join(stemmed_words)

        return stemmed_text

    def lemmatize_text(self, text):
        """
        Производит лемматизацию текста

        Параметры:
            text (str): Исходный текст.

        Возвращает:
            str: Текст лемматизированных слов.
        """
        # Создание объекта лемматизатора
        lemmatizer = WordNetLemmatizer()

        # Лемматизация каждого слова в тексте
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text.split()]

        # Сбор лемматизированных слов обратно в текст
        lemmatized_text = ' '.join(lemmatized_words)

        return lemmatized_text

    @property
    def get_text(self):
        """
        Получение обработанного текста

            Возвращает:
                str: Текст обработанных слов.
        """

        text = self.reading_csv()
        text = self.text_to_lower(text)
        text = self.remove_punctuation(text)
        text = self.remove_other_characters(text)
        text = self.remove_digits(text)
        text = self.remove_stopwords(text)
        text = self.remove_rare_words(text)

        if self.method == 'lemmatization':
            text = self.lemmatize_text(text)
        else:
            text = self.stem_text(text)

        return text


    def get_count_words(self, text):
        """
        Подсчитывает количество слов в тексте

        Возвращает:
            int: Количество слов.
        """
        # Разбиваем текст на слова, используя пробел как разделитель
        words = text.split()

        # Возвращаем количество слов
        return len(words)


# Анализ текста
# ===================================================================
class TextAnalysis:
    def __init__(self, text, chunk_size=1000000, type_tokenization='simple', n=3):
        self.text = text
        self.chunk_size = chunk_size
        self.type_tokenization = type_tokenization
        self.n = n

    def tokenize_text(self):
        """
        Токенизация текста

        Параметры:
            text (str): Исходный текст.

        Возвращает:
            str: Тоекнизированные данные.
        """

        tokens = word_tokenize(self.text)

        return tokens

    def tokenize_spacy(self):
        """
        Токенизация текста с использованием SpaCy.

            Возвращает:
                list: Список токенов.
        """
        # Загрузка модели 'en_core_web_sm' из SpaCy

        nlp = spacy.load('en_core_web_sm')

        # Разбивка текста на части
        chunks = [self.text[i:i + self.chunk_size] for i in range(0, len(self.text), self.chunk_size)]

        # Токенизация каждой части
        tokens = []

        for idx, chunk in enumerate(chunks):
            start_time = time.time()  # Засекаем время перед обработкой чанка
            doc = nlp(chunk)
            tokens.extend([token.text for token in doc])
            end_time = time.time()  # Засекаем время после обработки чанка
            print(f"Chunk {idx + 1} processed in {end_time - start_time:.2f} seconds")

        return tokens

    def get_tokens(self):
        """
        Получение токенов текста в зависимости от выбранного метода токенизации.

            Возвращает:
                list: Список токенов.
        """

        if self.type_tokenization == 'simple':
            tokens = self.tokenize_text()
        else:
            tokens = self.tokenize_spacy()

        return tokens

    def pos_tag_text(self):
        """
        POS-тегирование текста.

            Возвращает:
                list: Список кортежей (слово, тег).
        """

        # Токенизация текста
        tokens = self.get_tokens()

        # Выполнение POS тегирования
        pos_tags = nltk.pos_tag(tokens)

        return pos_tags

    def analyze_sentiment(self):
        """
        Анализ тональности текста.

            Возвращает:
                dict: Результаты анализа тональности.
        """

        # Создание объекта для анализа тональности
        sia = SentimentIntensityAnalyzer()

        # Анализ тональности текста
        sentiment_score = sia.polarity_scores(self.text)
        """ 
        Результаты анализа тональности текста:
        
            - neg: Оценка негативной тональности (от 0 до 1).
            - neu: Оценка нейтральной тональности (от 0 до 1).
            - pos: Оценка позитивной тональности (от 0 до 1).
            - compound: Общая компонента тональности (от -1 до 1).
        """
        return sentiment_score

    def word_frequency_sorted(self):
        """
        Частота встречаемости слов и сортировка по убыванию частоты.

            Возвращает:
                dict: Словарь слов и их частот встречаемости.
        """

        # Получаем токены
        tokens = self.get_tokens()

        # Вычисляем частоту встречаемости каждого слова (токена)
        word_counts = Counter(tokens)

        # Сортируем результат по частоте встречаемости
        sorted_word_counts = dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True))

        return sorted_word_counts

    def extract_ngrams(self):
        """
        Извлечение n-грамм из текста.

            Возвращает:
                list: Список n-грамм.
        """

        # Токенизация текста
        tokens = self.get_tokens()

        # Определение количества слов
        num_words = len(tokens)

        # Если параметр n не указан, вычисляем его
        if self.n is None:
            ng = int(round(pow(num_words, 1 / 2)))  # Округляем до целого
        else:
            ng = self.n

        # Извлечение n-грамм
        ngrams = list(nltk.ngrams(tokens, ng))

        return ngrams


