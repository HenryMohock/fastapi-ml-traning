import requests
import time
from Classifier import CsvPreProcessing, NLTKClassifier
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt


def format_time(seconds):
    """
    Форматирование секунд к виду ЧЧ:ММ:СС

      Возвращает строку отформатированного времени
    """
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return "{:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds))


def plot_bar(data):
    # Сортировка данных по значению Accuracy
    sorted_data = sorted(data.items(), key=lambda x: x[1])

    labels = [item[0] for item in sorted_data]
    accuracy_values = [item[1] for item in sorted_data]

    # Определение цветов для столбцов гистограммы
    colors = ['skyblue', 'orange', 'green', 'red', 'purple']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, accuracy_values, color=colors)
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Models')
    plt.xticks(rotation=45, ha='right')

    # Добавление текста в середину каждого бара
    for bar, accuracy in zip(bars, accuracy_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, f'{accuracy:.4f}', ha='center', va='center')

    # Находим модель с самой высокой точностью
    most_accurate_model = max(data, key=data.get)
    explanation = f"The most accurate model is {most_accurate_model}"

    # Добавление пояснения к графику
    plt.text(0.5, 1.15, explanation, ha='center', va='center', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.show()


def get_data(word_count: int = 0):
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

    # Метод обрезки слов (lemmatization или стемминг)
    method = 'lemmatization'

    # Инициализация экземпляра CsvPreProcessing
    csv_processor = CsvPreProcessing(file_path, column_name, regular_expression, threshold, method, encoding, lang)

    # Получаем текст из CsvPreProcessing
    text = csv_processor.get_text
    if word_count != 0:
        text = " ".join(text.split()[:word_count])
        words = text.split()
    else:
        words = text.split()

    # Создание объекта для анализа тональности
    sia = SentimentIntensityAnalyzer()
    sentiment_labels = []

    # Анализируем настроение каждого слова и добавляем метку в список
    for word in words:
        sentiment_score = sia.polarity_scores(word)
        # Определяем метку настроения на основе compound score
        if sentiment_score['compound'] >= 0.05:
            label = 'positive'
        elif sentiment_score['compound'] <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        sentiment_labels.append((word, label))

    return sentiment_labels


def get_training_and_testing_sets(word_count=1000):
    # Получение данных
    data = get_data(word_count)

    # Создание классификатора
    classifier = NLTKClassifier()

    # Создание обучающего и тестового наборов
    training_set, testing_set = classifier.create_training_and_testing_sets(data)

    return training_set, testing_set


def post_request(url, training_set, testing_set, model=None):
    data = {
        "training_set": training_set,
        "testing_set": testing_set,
        "model": model
    }
    start_time = time.time()
    response = requests.post(url, json=data)
    print("Время получения данных с сервера:", format_time(time.time() - start_time))
    if response.status_code == 200:
        answer = response.json()
        print(answer)
        plot_bar(answer)
    else:
        print('Error:', response.text)


def get_url(server, post_method):
    slash = '/'
    return f"{server}{post_method}{slash}"


# Адрес сервера
host = '127.0.0.1'
port = '8000'
server = 'http://'+host+':'+port+'/'
# Метод на сервере возвращающий результат запроса
post_method = 'train_classify_models'

# Создание обучающего и тестового наборов
start_time = time.time()
word_count = 100000
training_set, testing_set = get_training_and_testing_sets(word_count)
print("Время получения данных для тренировки:", format_time(time.time() - start_time))

# Получаем строку запроса
url = get_url(server, post_method)
print(url)

# Посылаем запрос серверу
print("Результаты тренировки моделей по их точности: ")
post_request(url, training_set, testing_set)



