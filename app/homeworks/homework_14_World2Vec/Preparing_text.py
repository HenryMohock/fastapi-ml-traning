import re
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter


class Preliminary_preparation_data:
    def __init__(self, file_path, column_name='', regular_expression='', threshold=5, method='lemmatization',
                 encoding='utf8', lang='english', type_file='txt'):
        self.file_path = file_path
        self.column_name = column_name
        self.regular_expression = regular_expression
        self.threshold = threshold
        self.method = method  # lemmatization, stemming, ''
        self.encoding = encoding
        self.lang = lang
        self.type_file = type_file
        self._text = None
        self._roh_text = None

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

    def _reading_txt(self):
        """
        Читает txt-файл и извлекает текст в текстовую переменную.

        Возвращает:
            str: Текст файла.
        """
        try:
            with open(self.file_path, 'r', encoding=self.encoding) as file:
                self._text = file.read()
        except FileNotFoundError:
            print("Файл не найден.")
        except Exception as e:
            print(f"Ошибка при чтении файла: {e}")

    def _reading_file(self):
        if self.type_file == 'txt':
            self._reading_txt()
        elif self.type_file == 'csv':
            self._reading_csv()
        else:
            raise ValueError(f"Не поддерживаемый тип файла: {self.type_file} !!!")

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
        Производит стемминг или лемматизацию или ничего

        текста
        """
        if self.method == 'lemmatization':
            lemmatizer = WordNetLemmatizer()
            self._text = ' '.join([lemmatizer.lemmatize(word) for word in self._text.split()])
        elif self.method == 'stemming':
            stemmer = PorterStemmer()
            self._text = ' '.join([stemmer.stem(word) for word in self._text.split()])

    @property
    def roh_text(self):  # сырой текст
        """
        Получение не обработанного текста

        Возвращает:
            str: Текст сырых слов.
        """
        if self._roh_text is None:
            self._reading_file()
            self._roh_text = self._text
        return self._roh_text

    @property
    def get_text(self):
        """
        Получение обработанного текста

        последоваетльным вызовом методов очищающих текст

        Возвращает:
            str: Текст обработанных слов.
        """
        self._reading_file() # чтение файла
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



