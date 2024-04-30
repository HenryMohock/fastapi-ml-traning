import signal
import sys
import time
from preprocessing_csv import CsvPreProcessing, TextAnalysis
import matplotlib.pyplot as plt


def format_time(seconds):
    """
    Форматирование секунд к виду ЧЧ:ММ:СС

      Возвращает строку отформатированного времени
    """
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return "{:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds))


def prepare_the_text(csv_pp):
    """
    Готовит текст для предварительного анализа

    Возвращает обработанные слова
    """
    text = csv_pp.get_text
    print('Подготовленный текст')
    print(text)

    return text


def perform_text_analysis(text, chunk_size, type_tokenization, n):
    """
    Проводит элементы предварительного анализа текста

    Перед каждой обработкой спрашивает пользователя, давая возможность отказаться
    """
    try:
        if input("Хотите выполнить анализ текста? (yes/no): ").lower() == "yes":
            # Создание объекта класса анализа подготовленного текста
            txt_analysis = TextAnalysis(text, chunk_size, type_tokenization, n)

            # Запрашиваем разрешение перед каждой операцией
            if input("Хотите выполнить операцию токенизации? (yes/no): ").lower() == "yes":
                start_time = time.time()
                tokens = txt_analysis.get_tokens()
                print('Токены')
                print(tokens)
                print("Время выполнения операции токенизации:", format_time(time.time() - start_time))

            if input("Хотите выполнить операцию POS-тегирование текста? (yes/no): ").lower() == "yes":
                start_time = time.time()
                pos_tag = txt_analysis.pos_tag_text()
                print('POS-тегирование текста')
                print(pos_tag)
                print("Время выполнения операции POS-тегирование:", format_time(time.time() - start_time))

            # Самый долгий процесс
            if input("Хотите выполнить Анализ тональности текста? (yes/no): ").lower() == "yes":
                start_time = time.time()
                analyze_sentiment = txt_analysis.analyze_sentiment()
                print('Анализ тональности текста')
                print(analyze_sentiment)
                print("Время выполнения операции анализа тональности:", format_time(time.time() - start_time))
                plot_sentiment_pie(analyze_sentiment)

            if input("Хотите выполнить Вычисление частоты встречаемости слов с сортировкой результатов? (yes/no): ").lower() == "yes":
                start_time = time.time()
                word_frequency = txt_analysis.word_frequency_sorted()
                print('Вычисление частоты встречаемости слов и сортировка результатов')
                print(word_frequency)
                print("Время выполнения операции вычисления частоты:", format_time(time.time() - start_time))

            if input("Хотите выполнить Извлечение n-грамм из текста()? (yes/no): ").lower() == "yes":
                start_time = time.time()
                extract_ngrams = txt_analysis.extract_ngrams()
                print('Извлечение n-грамм из текста')
                print(extract_ngrams)
                print("Время выполнения операции извлечения n-грамм:", format_time(time.time() - start_time))

    except KeyboardInterrupt:
        print("Прервано пользователем")
        sys.exit()


def plot_sentiment_pie(sentiment_score):
    """
    Функция для построения круговой диаграммы на основе словаря sentiment_score.

    Аргументы:
    - sentiment_score (dict): Словарь, содержащий оценки настроений. Должен содержать ключи 'neg', 'neu' и 'pos'.

    Пример использования:
    sentiment_score = {'neg': 0.161, 'neu': 0.621, 'pos': 0.218}
    plot_sentiment_pie(sentiment_score)
    """

    # Задаем метки для диаграммы
    labels = ['Негативное', 'Нейтральное', 'Позитивное']
    # Задаем размеры сегментов диаграммы
    sizes = [sentiment_score['neg'], sentiment_score['neu'], sentiment_score['pos']]
    # Задаем цвета для сегментов
    colors = ['#ff9999', '#66b3ff', '#99ff99']

    # Создаем объект рисунка
    plt.figure(figsize=(8, 6))
    # Строим круговую диаграмму
    plt.pie(sizes, colors=colors, labels=labels, autopct='%1.1f%%', startangle=140)
    # Задаем заголовок диаграммы
    plt.title('Анализ настроений')
    # Устанавливаем соотношение сторон для круга, чтобы диаграмма была кругом
    plt.axis('equal')
    # Показываем диаграмму, блокируя выполнение кода до закрытия окна пользователем
    plt.show(block=True)


# Регистрация обработчика сигнала прерывания
signal.signal(signal.SIGINT, signal.default_int_handler)

# Параметры текста
encoding = 'utf8'
lang = 'english'

# Путь к файлу CSV
file_path = "../../data/IMDB Dataset.csv"

# Имя извлекаемой колонки файла CSV
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
type_tokenization = 'simple'  # 'spacy'  # 'simple'

# Параметр n-грамм
n = 2


# Создание объекта класса подготовки текста к анализу
csv_pp = CsvPreProcessing(file_path, column_name, regular_expression, threshold, method, encoding, lang)

# Получение подготовленного текста
text = prepare_the_text(csv_pp)

# Вывод количества полученных слов
count_words = csv_pp.get_count_words
print(f'Количество слов = {count_words}')

# Анализ предварительно обработанного текста
perform_text_analysis(text, chunk_size, type_tokenization, n)











