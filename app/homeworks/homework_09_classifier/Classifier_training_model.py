from Classifier import CsvPreProcessing, NLTKClassifier
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import time
import matplotlib.pyplot as plt


def format_time(seconds):
    """
    Форматирование секунд к виду ЧЧ:ММ:СС

      Возвращает строку отформатированного времени
    """
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return "{:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds))


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

    # Создание объекта для анализа тональности
    sia = SentimentIntensityAnalyzer()
    sentiment_labels = []

    # Разбиваем текст на слова
    words = text.split()  # word_tokenize(text)

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


def train_models_and_get_results(models, training_set, testing_set):
    results = {}
    classifier = NLTKClassifier()

    for model_name, model in models.items():
        accuracy = classifier.train_and_evaluate_model(model, training_set, testing_set)
        results[model_name] = accuracy

    return results


def plot_results(results):
    labels = list(results.keys())
    accuracy_values = list(results.values())

    plt.figure(figsize=(8, 6))
    plt.pie(accuracy_values, labels=labels, autopct='%1.1f%%')
    plt.title('Accuracy of Models')
    plt.show()


def plot_results_bar(results):
    labels = list(results.keys())
    accuracy_values = list(results.values())

    # Определение цветов для столбцов гистограммы
    colors = ['skyblue', 'orange', 'green', 'red', 'purple']

    plt.figure(figsize=(10, 6))
    plt.bar(labels, accuracy_values, color=colors)
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Models')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


start_time = time.time()
word_count = 100
data = get_data(word_count)

classifier = NLTKClassifier()

# Создание обучающего и тестового наборов
training_set, testing_set = classifier.create_training_and_testing_sets(data)
print("Время получения данных для тренировки:", format_time(time.time() - start_time))

# Обучение и оценка производительности моделей
models = {
    "Multinomial Naive Bayes": MultinomialNB(),
    "Support Vector Machine": SVC(),
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier()
}

start_time = time.time()

results = train_models_and_get_results(models, training_set, testing_set)
for model_name, accuracy in results.items():
    print(f"{model_name}: Accuracy = {accuracy}")

print("Время выполнения операции тренировки:", format_time(time.time() - start_time))

plot_results(results)
plot_results_bar(results)
