import re
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.probability import FreqDist
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from collections import Counter
import spacy
import time
import random



# Обработка текста
# ===================================================================
class CsvPreProcessing:
    def __init__(self, file_path, column_name, regular_expression, threshold=5, method='lemmatization',
                 encoding='utf8', lang='english'):
        self.file_path = file_path
        self.column_name = column_name
        self.regular_expression = regular_expression
        self.threshold = threshold
        self.method = method
        self.encoding = encoding
        self.lang = lang
        self._text = None

    def _reading_csv(self):
        """
        Читает csv-файл и извлекает колонку 'review' в текстовую переменную.

        Возвращает:
            str: Текст из колонки 'review'.
        """
        # Загрузка данных из файла CSV
        data = pd.read_csv(self.file_path, encoding=self.encoding)
        # Извлечение колонки column_name и объединение ее в текстовую переменную (внутреннее свойство)
        self._text = ' '.join(data[self.column_name].astype(str))

    def _text_to_lower(self):
        """
        Преобразует текст к нижнему регистру.

        Приводит:
            str: Текст к нижнему регистру.
        """
        self._text = self._text.lower()

    def _apply_regex(self, regex):
        """
            Удаляет все лишние символы и слова.

            согласно переданному регулярному выражению
        """
        self._text = re.sub(r'[^\w\s]', '', self._text)
        self._text = re.sub(r'\d+', '', self._text)
        self._text = re.sub(regex, '', self._text)

    def _remove_stopwords(self):
        """
        Удаляет все стоп-слова из текста

        Приводит:
            str: Текст без указанных символов.
        """
        stop_words = set(stopwords.words(self.lang))
        words = self._text.split()
        self._text = ' '.join([word for word in words if word.lower() not in stop_words])

    def _remove_rare_words(self):
        """
        Удаляет все редкие слова из текста

        в соответствии с порогом встречаемости
        """
        words = self._text.split()
        word_counts = Counter(words)
        self._text = ' '.join([word for word in words if word_counts[word] >= self.threshold])

    def _stem_or_lemmatize(self):
        """
        Производит стемминг или лемматизацию

        текста
        """
        if self.method == 'lemmatization':
            lemmatizer = WordNetLemmatizer()
            self._text = ' '.join([lemmatizer.lemmatize(word) for word in self._text.split()])
        else:
            stemmer = PorterStemmer()
            self._text = ' '.join([stemmer.stem(word) for word in self._text.split()])

    @property
    def roh_text(self):  # сырой текст
        """
        Получение не обработанного текста

        Возвращает:
            str: Текст сырых слов.
        """
        if self._text is None:
            self._reading_csv()
        return self._text

    @property
    def get_text(self):
        """
        Получение обработанного текста

        последоваетльным вызовом методов очищающих текст

        Возвращает:
            str: Текст обработанных слов.
        """
        self._reading_csv()  # чтение файла
        # Запуск внутренних методов обработки текста:
        self._text_to_lower()  # приведение текста к нижнему регистру
        self._apply_regex(r'[^\w\s]')  # удаление пунктуации
        self._apply_regex(self.regular_expression)  # удаление прочих символов
        self._apply_regex(r'\d')  # удаление цифр из текста
        self._remove_stopwords()  # удаление стоп-слов
        self._remove_rare_words()  # удаление редко встречающихся слов
        self._stem_or_lemmatize()  # проведение стемминга либо лемматизации
        return self._text

    @property
    def get_count_words(self):
        """
            Подсчитывает количество слов в тексте

            Возвращает:
                int: Количество слов.
            """
        if not self._text:
            return 0
        return len(self._text.split())

    def _extract_features(self, document, word_features):
        document_words = set(document)
        features = {}
        for word in word_features:
            features['contains({})'.format(word)] = (word in document_words)
        return features

    @property
    def get_filtered_tokens(self):
        filtered_tokens = self.get_text.split()
        return filtered_tokens


class TextAnalysis:
    def __init__(self, text, chunk_size=1000000, type_tokenization='simple', n=3):
        self.text = text
        self.chunk_size = chunk_size
        self.type_tokenization = type_tokenization
        self.n = n
        self.tokens = None  # Сохраняем токены при создании объекта класса

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

        if self.tokens is None:
            if self.type_tokenization == 'simple':
                self.tokens = self.tokenize_text()
            else:
                self.tokens = self.tokenize_spacy()
        return self.tokens

    def pos_tag_text(self):
        """
        POS-тегирование текста.

            Возвращает:
                list: Список кортежей (слово, тег).
        """

        # Выполнение POS тегирования
        pos_tags = nltk.pos_tag(self.get_tokens())

        return pos_tags

    def analyze_sentiment(self):
        """
        Анализ тональности текста.

            Возвращает:
                dict: Результаты анализа тональности.
        """
        # Анализ тональности текста
        sia = SentimentIntensityAnalyzer()  # Создание объекта для анализа тональности
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

        # Вычисляем частоту встречаемости каждого слова (токена)
        word_counts = Counter(self.get_tokens())

        # Сортируем результат по частоте встречаемости
        sorted_word_counts = dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True))

        return sorted_word_counts

    def extract_ngrams(self):
        """
        Извлечение n-грамм из текста.

            Возвращает:
                list: Список n-грамм.
        """

        # Определение количества слов
        num_words = len(self.get_tokens())

        # Если параметр n не указан, вычисляем его
        if self.n is None:
            ng = int(round(pow(num_words, 1 / 2)))  # Округляем до целого
        else:
            ng = self.n

        # Извлечение n-грамм
        ngrams = list(nltk.ngrams(self.get_tokens(), ng))

        return ngrams



    # Класс для обучения классификатора с использованием NLTK


class NLTKClassifier:
    def preprocess_text(self, text):
        words = text.split()
        return FreqDist(words)

    def create_training_and_testing_sets(self, data):
        featuresets = [(self.preprocess_text(text), label) for (text, label) in data]
        random.shuffle(featuresets)
        cutoff = int(0.8 * len(featuresets))
        training_set = featuresets[:cutoff]
        testing_set = featuresets[cutoff:]
        return training_set, testing_set

    def train_and_evaluate_model(self, classifier, training_set, testing_set):
        classifier = SklearnClassifier(classifier)
        classifier.train(training_set)
        y_true = [label for (text, label) in testing_set]
        y_pred = classifier.classify_many([text for (text, label) in testing_set])
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy


