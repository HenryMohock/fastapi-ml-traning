import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Предобработка текста с помощью регулярных выражений
def preprocess_text(text):
    # Удаление символов пунктуации, цифр и лишних пробелов
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


# Тренировка модели по методу BoW или TF_IDF
def fit_model(name_model, X_train_method, y_data_train):

    # Проверка значения параметра name_model
    if name_model not in ['LogisticRegression', 'RidgeClassifier', 'SGDClassifier']:
        raise ValueError(
            "Параметр name_model должен быть одним из: 'LogisticRegression', 'RidgeClassifier', 'SGDClassifier'")

    print('=================================================================')
    print('MODEL: ', name_model)

    if name_model == 'LogisticRegression':
        trained_model = LogisticRegression(max_iter=1000)
        trained_model.fit(X_train_method, y_data_train)

    if name_model == 'RidgeClassifier':
        trained_model = RidgeClassifier()
        trained_model.fit(X_train_method, y_data_train)

    if name_model == 'SGDClassifier':
        trained_model = SGDClassifier(loss='hinge', random_state=42)
        trained_model.fit(X_train_method, y_data_train)

    return trained_model


# Вывод результатов
def output_of_results(object_model, X_train_method, X_test_method, y_train, y_test, name_method, name_model):
    # Проверка значения параметра name_model
    if name_model not in ['LogisticRegression', 'RidgeClassifier', 'SGDClassifier']:
        raise ValueError(
            "Параметр name_model должен быть одним из: 'LogisticRegression', 'RidgeClassifier', 'SGDClassifier'")

    # Проверка значения параметра name_method
    if name_method not in ['BoW', 'TF_IDF']:
        raise ValueError("Параметр name_method должен быть одним из: 'BoW', 'TF_IDF'")

    y_pred_method_train = object_model.predict(X_train_method)
    accuracy_method_train = accuracy_score(y_train, y_pred_method_train)

    y_pred_method_test = object_model.predict(X_test_method)
    accuracy_method_test = accuracy_score(y_test, y_pred_method_test)

    print(f"Method: {name_method}")
    print(f"Accuracy для тренировочных данных: {accuracy_method_train}")
    print(f"Accuracy для тестовых данных: {accuracy_method_test}")

    # Создание DataFrame для гистограмм
    df_train = pd.DataFrame({'prediction': y_pred_method_train, 'actual': y_train})
    df_test = pd.DataFrame({'prediction': y_pred_method_test, 'actual': y_test})

    # Визуализация результатов
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # График для тренировочных данных
    sns.histplot(data=df_train, x='prediction', hue='actual', multiple="dodge", shrink=.8, ax=ax[0])
    ax[0].set_title(name_method+' (train)')
    ax[0].set_xlabel("Sentiment Prediction "+name_model)
    ax[0].set_ylabel("Count")

    # График для тестовых данных
    sns.histplot(data=df_test, x='prediction', hue='actual', multiple="dodge", shrink=.8, ax=ax[1])
    ax[1].set_title(name_method+' (test)')
    ax[1].set_xlabel("Sentiment Prediction "+name_model)
    ax[1].set_ylabel("Count")

    plt.tight_layout()
    plt.show()


# Функция для вывода отчетов:
def get_report(model, X, y_true):
    y_pred = model.predict(X)
    print('Classification Report:')
    print(classification_report(y_true, y_pred, digits=4))


# Загрузка данных
data = pd.read_csv('../../data/IMDB Dataset.csv')

data['review'] = data['review'].apply(preprocess_text)
print(data.head())

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(data['review'], data['sentiment'], test_size=0.2, random_state=42)

# Инициализация и применение метода Bag of Words
bow_vectorizer = CountVectorizer(max_features=5000)
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)

# Инициализация и применение метода TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Обучение линейной модели LogisticRegression на основе классификатора Logistic Regression с использованием Bag of Words
model_bow = fit_model('LogisticRegression', X_train_bow, y_train)
output_of_results(model_bow, X_train_bow, X_test_bow, y_train, y_test,
                  'BoW', 'LogisticRegression')
get_report(model_bow, X_train_bow, y_train)

# Обучение линейной модели LogisticRegression на основе классификатора Logistic Regression с использованием TF-IDF
model_tfidf = fit_model('LogisticRegression', X_train_tfidf, y_train)
output_of_results(model_tfidf, X_train_tfidf, X_test_tfidf, y_train, y_test,
                  'TF_IDF', 'LogisticRegression')
get_report(model_tfidf, X_test_tfidf, y_test)

# Обучение линейной модели RidgeClassifier на основе классификатора RidgeClassifier с использованием Bag of Words
model_bow = fit_model('RidgeClassifier', X_train_bow, y_train)
output_of_results(model_bow, X_train_bow, X_test_bow, y_train, y_test,
                  'BoW', 'RidgeClassifier')
get_report(model_bow, X_train_bow, y_train)

# Обучение линейной модели RidgeClassifier на основе классификатора RidgeClassifier с использованием TF-IDF
model_tfidf = fit_model('RidgeClassifier', X_train_tfidf, y_train)
output_of_results(model_tfidf, X_train_tfidf, X_test_tfidf, y_train, y_test,
                  'TF_IDF', 'RidgeClassifier')
get_report(model_tfidf, X_test_tfidf, y_test)

# Обучение линейной модели SGDClassifier на основе классификатора SGDClassifier с использованием Bag of Words
model_bow = fit_model('SGDClassifier', X_train_bow, y_train)
output_of_results(model_bow, X_train_bow, X_test_bow, y_train, y_test,
                  'BoW', 'SGDClassifier')
get_report(model_bow, X_train_bow, y_train)

# Обучение линейной модели SGDClassifier на основе классификатора SGDClassifier с использованием TF-IDF
model_tfidf = fit_model('SGDClassifier', X_train_tfidf, y_train)
output_of_results(model_tfidf, X_train_tfidf, X_test_tfidf, y_train, y_test,
                  'TF_IDF', 'SGDClassifier')
get_report(model_tfidf, X_test_tfidf, y_test)

print('=================================================================')

