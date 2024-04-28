import signal
import sys
from preprocessing_csv import CsvPreProcessing, TextAnalysis


def prepare_the_text(csv_pp):
    text = csv_pp.get_text
    print('Подготовленный текст')
    print(text)

    return text


def perform_text_analysis(text, chunk_size, type_tokenization, n, pos_tag=None, word_frequency=None):
    try:
        if input("Хотите выполнить анализ текста? (yes/no): ").lower() == "yes":
            # Создание объекта класса анализа подготовленного текста
            txt_analysis = TextAnalysis(text, chunk_size, type_tokenization, n)

            # Запрашиваем разрешение перед каждой операцией
            if input("Хотите выполнить операцию токенизации? (yes/no): ").lower() == "yes":
                tokens = txt_analysis.get_tokens()
                print('Токены')
                print(tokens)
            else:
                return  # прерываем выполнение функции

            if input("Хотите выполнить операцию POS-тегирование текста? (yes/no): ").lower() == "yes":
                pos_tag = txt_analysis.pos_tag_text()
                print('POS-тегирование текста')
                print(pos_tag)

            if input("Хотите выполнить Анализ тональности текста? (yes/no): ").lower() == "yes":
                analyze_sentiment = txt_analysis.analyze_sentiment()
                print('Анализ тональности текста')
                print(analyze_sentiment)

            # Самый долгий процесс
            if input("Хотите выполнить Вычисление частоты встречаемости слов с сортировкой результатов? (yes/no): ").lower() == "yes":
                word_frequency = txt_analysis.word_frequency_sorted()
                print('Вычисление частоты встречаемости слов и сортировка результатов')
                print(word_frequency)

            if input("Хотите выполнить Извлечение n-грамм из текста()? (yes/no): ").lower() == "yes":
                extract_ngrams = txt_analysis.extract_ngrams()
                print('Извлечение n-грамм из текста')
                print(extract_ngrams)

    except KeyboardInterrupt:
        print("Прервано пользователем")
        sys.exit()


# Регистрация обработчика сигнала прерывания
signal.signal(signal.SIGINT, signal.default_int_handler)

# Путь к файлу CSV
file_path = "../../data/IMDB Dataset.csv"

# Имя колонки
column_name = 'review'

# Регулярное выражение для удаления прочих символов
regular_expression = r'blah|_|fffc|br|oz'

# Порог встречаемости слов
threshold = 6

# Размер блока слов для SpaCy
chunk_size = 1000000

# Метод обрезки слов (lemmatization или стемминг)
method = 'lemmatization'

# Тип токенизации (simple или с использованием SpaCy)
type_tokenization = 'simple'

# Параметр n-грамм
n = 2

group_percentages = [5, 15, 30, 50]

# Создание объекта класса подготовки текста к анализу
csv_pp = CsvPreProcessing(file_path, column_name, regular_expression, threshold, method)

# Получение подготовленного текста
text = prepare_the_text(csv_pp)

# Вывод количества полученных слов
count_words = csv_pp.get_count_words(text)
print(f'Количество слов = {count_words}')

# Анализ предварительно обработанного текста
pos_tag = None
word_frequency = None
perform_text_analysis(text, chunk_size, type_tokenization, n, pos_tag, word_frequency)











