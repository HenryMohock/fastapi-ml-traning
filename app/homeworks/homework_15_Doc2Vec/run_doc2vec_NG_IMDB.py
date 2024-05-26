import os
import re
import pickle
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import multiprocessing
from keras.datasets import imdb
from keras.preprocessing import sequence
from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from collections import Counter
from itertools import islice
from nltk.corpus import stopwords
import time


# Пути для хранения данных
ng_train_path = '../../data/newsgroups_train.pkl'
ng_test_path = '../../data/newsgroups_test.pkl'
imdb_train_texts_path = '../../data/imdb_train_texts.pkl'
imdb_test_texts_path = '../../data/imdb_test_texts.pkl'
combined_texts_path = '../../data/combined_texts.pkl'

# Загрузить список стоп-слов
stop_words = set(stopwords.words('english'))


def format_time(seconds):
    """
    Форматирование секунд к виду ЧЧ:ММ:СС

      Возвращает строку отформатированного времени
    """
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return "{:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds))


def clean_text(text):
    text = re.sub(r'\S+@\S+', '', text)  # Удаление email-адресов
    text = re.sub(r'\n', ' ', text)  # Удаление символов новой строки
    text = re.sub(r"\'", '', text)  # Удаление одинарных кавычек
    return text


def sent_to_words(sentences):
    for sentence in sentences:
        yield gensim.utils.simple_preprocess(str(sentence), deacc=True)  # Токенизация и очистка


# Функция для загрузки данных и сохранения их на диск
def load_and_save_datasets():
    # Загружаем датасет NG
    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')

    # Очистка и токенизация текстов NG
    newsgroups_train_cleaned = [clean_text(doc) for doc in newsgroups_train.data]
    newsgroups_test_cleaned = [clean_text(doc) for doc in newsgroups_test.data]
    newsgroups_train_tokens = list(sent_to_words(newsgroups_train_cleaned))
    newsgroups_test_tokens = list(sent_to_words(newsgroups_test_cleaned))

    # Сохраняем датасет NG
    with open(ng_train_path, 'wb') as f:
        pickle.dump(newsgroups_train_tokens, f)
    with open(ng_test_path, 'wb') as f:
        pickle.dump(newsgroups_test_tokens, f)

    # Загружаем датасет IMDB
    max_features = 5000
    maxlen = 400
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    word_index = imdb.get_word_index()

    # Преобразуем индексы в слова
    index_word = {v: k for k, v in word_index.items()}
    x_train_texts = [' '.join(index_word.get(i - 3, '?') for i in x) for x in x_train]
    x_test_texts = [' '.join(index_word.get(i - 3, '?') for i in x) for x in x_test]

    # Очистка и токенизация текстов IMDB
    x_train_cleaned = [clean_text(doc) for doc in x_train_texts]
    x_test_cleaned = [clean_text(doc) for doc in x_test_texts]
    x_train_tokens = list(sent_to_words(x_train_cleaned))
    x_test_tokens = list(sent_to_words(x_test_cleaned))

    # Сохраняем датасет IMDB
    with open(imdb_train_texts_path, 'wb') as f:
        pickle.dump(x_train_tokens, f)
    with open(imdb_test_texts_path, 'wb') as f:
        pickle.dump(x_test_tokens, f)

    # Объединяем датасеты и сохраняем их
    combined_texts = newsgroups_train_tokens + newsgroups_test_tokens + x_train_tokens + x_test_tokens
    with open(combined_texts_path, 'wb') as f:
        pickle.dump(combined_texts, f)

start_time = time.time()

# Проверяем наличие объединенного датасета на диске и загружаем его
if not os.path.exists(combined_texts_path):
    load_and_save_datasets()

# Загружаем объединенный датасет с диска
with open(combined_texts_path, 'rb') as f:
    combined_texts = pickle.load(f)

# Преобразуем токены обратно в строки для Doc2Vec
combined_texts_str = [' '.join(tokens) for tokens in combined_texts]

# Сохраняем текстовый файл
combined_texts_str_path = '../../data/combined_texts.txt'
with open(combined_texts_str_path, 'w', encoding='utf-8') as f:
    for item in combined_texts_str:
        f.write("%s\n" % item)

# Подготавливаем документы для Doc2Vec
documents = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(combined_texts_str)]

print("Время подготовки документов для Doc2Vec:", format_time(time.time() - start_time))

start_time = time.time()

# Параметры модели
cores = multiprocessing.cpu_count()
model = Doc2Vec(vector_size=100, window=2, min_count=2, workers=cores, epochs=40)

# Строим словарь и тренируем модель
model.build_vocab(documents)
model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

# Сохраняем модель
model_path = "../../models/doc2vec/doc2vec_combined.model"
model.save(model_path)

print("Время тренировки и сохранения модели Doc2Vec:", format_time(time.time() - start_time))

# Для проверки
print(model.infer_vector(["This", "is", "a", "great", "movie"]))


def determine_optimal_clusters(vectors: List, texts: List[str], max_clusters: int = 10) -> Tuple[int, List[str]]:
    sse = []
    silhouette_scores = []
    for num_clusters in range(2, min(max_clusters + 1, len(vectors))):
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(vectors)
        sse.append(kmeans.inertia_)
        if 1 < num_clusters < len(vectors):
            silhouette_scores.append(silhouette_score(vectors, kmeans.labels_))
        else:
            silhouette_scores.append(float('-inf'))  # Для несовместимых значений

    # Метод локтя
    plt.figure(figsize=(10, 5))
    plt.plot(range(2, min(max_clusters + 1, len(vectors))), sse, marker='o')
    plt.title('Метод ліктя для оптимальних кластерів')
    plt.xlabel('Кількість кластерів')
    plt.ylabel('Сума квадратів відстаней')
    plt.show()

    # Метод силуэтного анализа
    plt.figure(figsize=(10, 5))
    plt.plot(range(2, min(max_clusters + 1, len(vectors))), silhouette_scores, marker='o')
    plt.title('Silhouette Score for Optimal Clusters')
    plt.xlabel('Кількість кластерів')
    plt.ylabel('Силуетна оцінка')
    plt.show()

    # Возвращаем количество кластеров с наибольшим силуэтным индексом
    optimal_clusters = range(2, min(max_clusters + 1, len(vectors)))[silhouette_scores.index(max(silhouette_scores))]

    # Кластеризация для получения имен тем
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=0)
    kmeans.fit(vectors)
    labels = kmeans.labels_

    # Получение наиболее частых пар слов для каждого кластера
    # Слова фильтруются по длине (не менее 6 букв) и исключаются стоп-слова.
    # Находятся две наиболее частые пары слов для каждого кластера.
    # Если найдено менее двух подходящих слов, добавляется только имеющиеся слова или пустая строка,
    # если нет подходящих слов.
    topic_words = []
    for i in range(optimal_clusters):
        cluster_texts = [texts[j] for j in range(len(texts)) if labels[j] == i]
        all_words = ' '.join(cluster_texts).split()
        # Фильтруем слова, исключая стоп-слова и слова короче 6 букв
        filtered_words = [word for word in all_words if word.lower() not in stop_words and len(word) >= 6]
        # Получаем две наиболее частые пары слов
        most_common_words = [word for word, count in Counter(filtered_words).most_common(2)]
        if len(most_common_words) == 2:
            topic_words.append(' '.join(most_common_words))
        else:
            topic_words.append(' '.join(most_common_words) if most_common_words else "")

    return optimal_clusters, topic_words


def group_sentences(sentences: List[str], max_clusters: int = 10) -> Dict[str, List[str]]:
    # Загрузить ранее сохраненную модель
    model = Doc2Vec.load(model_path)

    # Получение векторов для предложений
    vectors = [model.infer_vector(sentence.split()) for sentence in sentences]

    # Определение оптимального количества кластеров и их имен
    num_clusters, topic_names = determine_optimal_clusters(vectors, sentences, max_clusters)
    print(f'Optimal number of clusters: {num_clusters}')
    print(f'Topic names: {topic_names}')

    # Кластеризация предложений
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(vectors)
    labels = kmeans.labels_

    # Группировка предложений по темам
    grouped_sentences = {}
    for label, sentence in zip(labels, sentences):
        topic = topic_names[label]
        if topic not in grouped_sentences:
            grouped_sentences[topic] = []
        grouped_sentences[topic].append(sentence)

    return {"grouped_sentences": grouped_sentences}


start_time = time.time()

# Пример использования функции
sentences = [
    "The stock market is down today.",
    "The weather is sunny and warm.",
    "Investors are concerned about the economy.",
    "It might rain tomorrow.",
    "The Federal Reserve raised interest rates.",
    "Had the front panel of my car stereo stolen.",
    "If you do not have this excellent series on disc believe that you should purchase it and put it in your collection.",
    "The the lord of the rings was great adaptation of the story.",
    "My print was perfect.",
    "Is it possible to do wheelie on motorcycle with shaft drive.",
    "Re limiting govt was re employment was re why not concentrate organization at lines in article.",
    "Need to get the specs or at least very verbose interpretation of the specs for QuickTime.",
]

grouped = group_sentences(sentences)
print(grouped)

# Сортировка и вывод результатов
for topic in sorted(grouped["grouped_sentences"].keys()):
    print(f"Cluster: {topic}")
    for sentence in grouped["grouped_sentences"][topic]:
        print(f"  {sentence}")

print("Время группировки предложений моделью Doc2Vec:", format_time(time.time() - start_time))