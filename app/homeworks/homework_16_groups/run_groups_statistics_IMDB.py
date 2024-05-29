from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.tokenize import word_tokenize
from datasets import load_dataset
import string
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt  # для визуализации PCA
import pandas as pd
from sklearn.manifold import TSNE  # для визуализации t-SNE
import numpy as np  # для работы с массивами numpy
from sklearn.feature_extraction import text
from collections import Counter
import random


# Загрузка датасета IMDB
dataset = load_dataset("imdb")

# Получение количества отзывов в датасете
num_reviews = len(dataset['train']['text']) + len(dataset['test']['text'])
print(f"Загальна кількість оглядів в наборі даних IMDb: {num_reviews}")

# Выборка части данных для обучения (например, 1000 отзывов)
texts = dataset['train']['text'][:1000]

# Загрузка стоп-слов и токенизатора
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Слова для исключения из статистики
# Датасет IMDB - это про восхваление и хейт фильмов, то чтобы точнее понять почему предложения попали в кластер
# стоит удалить весь часто встречаемый мусор и самое главное - все слоав близкие к кино и разговорам о кино, т.к.
# они присутствуют практически в каждом предложении.
exclude_words = stop_words.union({
    # мусор
    '/><br', 'the', '&', 'a', 'and', 'to', 'of', 'is', 'in', 'i', 'that', 'this', '<br', 'It', '-', '/>the', "\x96",
    '--', "i've", '/>i',
    # топовые слова присутствующие в каждом кластере
    'movie', 'movie.', 'film', 'one', 'even', 'like', 'would', 'see', 'good', 'story', 'much', 'get', 'really',
    'make', 'could', 'first', 'films', 'bad', 'made', 'it.', 'plot', 'time', 'people', 'two', "he's", 'also',
    'scene', 'watch', 'go', 'movies', 'scenes', 'way', 'character', 'film,', 'think', 'little', "i'm", 'ever',
    'never', 'every', 'better', 'film.', 'characters', 'know', 'acting', 'nothing', 'going', 'watching', 'movie,',
    'many', 'something', 'say', 'however,', 'new', 'back', 'seen', 'well', 'real', 'show', 'still', 'pretty', 'seems',
    'thing', 'want', 'us', 'horror', 'man', "that's", 'great', 'love', 'end', "can't", 'actually', 'another', 'director',
    'actors', 'give', 'saw', 'got', 'look', 'least', 'old', 'makes', 'someone', 'guy'
})


# Функция предварительной обработки текста с удалением исключаемых слов
def preprocess_text(text, exclude_words):
    # Удаление HTML-тегов
    text = BeautifulSoup(text, "html.parser").get_text()
    # Удаление строк вида '<br /><br />'
    text = re.sub(r'<br\s*/><br\s*/>', ' ', text)
    # Приведение к нижнему регистру
    text = text.lower()
    # Удаление пунктуации
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Токенизация и удаление стоп-слов и исключаемых слов
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words and word not in exclude_words]
    return tokens


# Применение конструкции List Comprehension для предварительной обработки к предложениям
# enumerate индексирует текст, а в каждой итерации цикла происходит препроцессинг предложения c помощью preprocess_text
documents = [TaggedDocument(words=preprocess_text(text, exclude_words), tags=[str(i)]) for i, text in enumerate(texts)]
# documents = [LabeledSentence(words=preprocess_text(text), tags=[str(i)]) for i, text in enumerate(texts)]

# Шаг 2: Обучение модели Doc2Vec
model = Doc2Vec(
    documents=documents,
    vector_size=95,     # размер векторного представления (количество измерений в векторе для предложений) (60)
    window=5,           # сколько слов рядом (справа и слева) с текущим словом будет учитываться при обучении (3)
    min_count=2,        # минимальное количество упоминаний слова в коллекции документов для его включения в обучение(2)
    workers=6,          # количество потоков, которые будут использоваться для тренировки модели (6)
    epochs=6,           # количество проходов по корпусу документов во время тренировки
)

# Шаг 3: Описание документов векторами
vectors = [model.dv[str(i)] for i in range(len(texts))]

# Шаг 4: Применение алгоритма кластеризации с использованием алгоритма K-means
num_clusters = 5  # разбиваем данные на 5 кластеров
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(vectors)  # обучение K-means на векторах
labels = kmeans.labels_  # номера кластеров для каждого вектора документа

# Шаг 5: Расчет и вывод в % среднего индекса силуэта для набора данных
silhouette_avg = silhouette_score(vectors, labels)
print()
print(f'Середній бал силуету: {"{:.2%}".format(silhouette_avg)}')
print()

# Шаг 6: Визуализация PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(vectors)

# Создание DataFrame для визуализации
df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
df['Кластер'] = labels

# Визуализация
plt.figure(figsize=(10, 8))
scatter = plt.scatter(df['PC1'], df['PC2'], c=df['Кластер'], cmap='viridis', alpha=0.5)
plt.suptitle('PCA візуалізація кластерів')
plt.xlabel('Основний компонент 1')
plt.ylabel('Основний компонент 2')
plt.legend(handles=scatter.legend_elements()[0], labels=[f'Кластер {i}' for i in range(num_clusters)], loc='upper right')
fig = plt.gcf()
fig.canvas.manager.set_window_title('PCA Visualization of Clusters')
plt.savefig('../../data/PCA_Visualization_of_Clusters')  # сохранение картинки в папке data
plt.show()

# Шаг 7: Визуализация t-SNE
tsne = TSNE(n_components=2, random_state=0)
vectors = np.array([model.dv[str(i)] for i in range(len(texts))])  # Преобразование в numpy array
tsne_components = tsne.fit_transform(vectors)

# Создание DataFrame для визуализации
df = pd.DataFrame(data=tsne_components, columns=['Компонент 1', 'Компонент 2'])
df['Кластер'] = labels

# Визуализация
plt.figure(figsize=(10, 8))
scatter = plt.scatter(df['Компонент 1'], df['Компонент 2'], c=df['Кластер'], cmap='viridis', alpha=0.5)
plt.title('t-SNE Візуалізація кластерів')
plt.xlabel('Основний компонент 1')
plt.ylabel('Основний компонент 2')
plt.legend(handles=scatter.legend_elements()[0], labels=[f'Кластер {i}' for i in range(num_clusters)], loc='upper right')
fig = plt.gcf()
fig.canvas.manager.set_window_title('t-SNE Visualization of Clusters')
plt.savefig('../../data/t-SNE_Visualization_of_Clusters')  # сохранение картинки в папке data
plt.show()

# Шаг 8: Группировка документов по кластерам
clusters = {i: [] for i in range(num_clusters)}
for i, label in enumerate(labels):
    clusters[label].append(texts[i])


# Шаг 9: Тематическое моделирование и именование кластеров
def get_topic_names(cluster_texts, num_topics=1, num_words=5):
    # Добавляем 'br' к стандартному списку стоп-слов
    stop_words = list(text.ENGLISH_STOP_WORDS) + ['br']  # убираем стоп-слова и 'br' из имен кластеров

    # Создаем CountVectorizer с обновленным списком стоп-слов
    vectorizer = CountVectorizer(max_df=0.95, min_df=0.01, stop_words=stop_words)  # min_df=2
    doc_term_matrix = vectorizer.fit_transform(cluster_texts)

    # обучение модели LDA с извлеченными темами и их распределениями по документам.
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
    lda.fit(doc_term_matrix)

    # извлечение ключевых слов для названия тем в список
    topic_keywords = []
    for topic in lda.components_:
        topic_keywords.append([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-num_words:]])

    return topic_keywords


# Названия тем для каждого кластера
cluster_names = {}
for cluster, docs in clusters.items():
    topic_keywords = get_topic_names(docs)
    cluster_names[cluster] = ' '.join(topic_keywords[0])

# Вывод результатов кластеризации с названиями тем
for cluster, docs in clusters.items():
    print(f"Кластер {cluster} ({cluster_names[cluster]}):")
    for doc in docs[:5]:  # Вывод первых 5 документов для каждого кластера
        print(f" --> {doc[:300]}...")  # Вывод первых 300 символов каждого документа для краткости
    print()


# Шаг 10: Расчет дополнительных статистик для кластеров

# Рассчитаем среднюю и медианную длину документов в каждом кластере
avg_doc_lengths = {cluster: np.mean([len(doc.split()) for doc in docs]) for cluster, docs in clusters.items()}
median_doc_lengths = {cluster: np.median([len(doc.split()) for doc in docs]) for cluster, docs in clusters.items()}

# Распределение длины документов по символам
char_lengths = {cluster: [len(doc) for doc in docs] for cluster, docs in clusters.items()}

# Частота наиболее часто встречающихся слов
all_words = ' '.join([doc for docs in clusters.values() for doc in docs])
word_counts = Counter(all_words.split())


# Функция для извлечения топ-N слов без указанных исключений
def get_top_n_words_excluding(docs, exclude_words, n=10):
    # Приводим каждый документ к нижнему регистру
    docs_lower = [doc.lower() for doc in docs]
    all_words = ' '.join(docs_lower)
    word_counts = Counter(all_words.split())
    filtered_word_counts = {word: count for word, count in word_counts.items() if word not in exclude_words}
    return Counter(filtered_word_counts).most_common(n)


# Обновленный расчет частоты наиболее часто встречающихся слов без исключенных
top_words_per_cluster = {
    cluster: get_top_n_words_excluding(docs, exclude_words, 10) for cluster, docs in clusters.items()}

# Доля уникальных слов
unique_word_ratio = {cluster: len(set(' '.join(docs).split())) / len(' '.join(docs).split()) for cluster, docs in clusters.items()}


# Функция для извлечения топ-N слов без указанных исключений
def get_top_n_words(docs, exclude_words, n=5):
    # Приводим каждый документ к нижнему регистру
    docs_lower = [doc.lower() for doc in docs]

    all_words = ' '.join(docs_lower)
    word_counts = Counter(all_words.split())
    filtered_word_counts = {word: count for word, count in word_counts.items() if word not in exclude_words}
    return [word for word, _ in Counter(filtered_word_counts).most_common(n)]


# Обновленный список топ-N ключевых слов кластера без исключенных
top_n_keywords = {cluster: get_top_n_words(docs, exclude_words) for cluster, docs in clusters.items()}

# Выведем статистики для каждого кластера
print('Кількісні статистики для кожного кластера')
for cluster, docs in clusters.items():
    print(f"Статистика для кластера {cluster} ({cluster_names[cluster]}):")
    print(f"Количество документов: {len(docs)}")
    print(f"Средняя длина документов: {avg_doc_lengths[cluster]:.2f} слов")
    print(f"Медіанна довжина документів: {median_doc_lengths[cluster]:.2f} слов")
    print(f"Розподіл кількості слів: {word_counts[cluster]}")
    print(f"Розподіл довжини документів за символами: {char_lengths[cluster]}")
    print(f"Частка унікальних слів: {unique_word_ratio[cluster]:.2f}")
    print()


print('Частотні характеристики кластерів')
print()
for cluster, docs in clusters.items():
    print(f"Частота слів, що найчастіше зустрічаються: {top_words_per_cluster[cluster]}")

print()
for cluster, docs in clusters.items():
    print(f"Топ-N ключових слів: {top_n_keywords[cluster]}")


# Шаг 11: Создание случайных предложений из N-топа слов.

# Функция для создания предложения из списка слов
def create_sentence(words):
    # Базовые структуры предложений для разнообразия
    structures = [
        "The {0} was {1} and {2}, making it the {3} {4}.",
        "In a {0} of {1}, the {2} {3} the {4}.",
        "It is {0} to {1} a {2} that is {3} and {4}.",
        "A {0} {1} can {2} with {3}, but {4}.",
        "To {0} a {1}, you need {2}, {3}, and {4}."
    ]
    # Выбор случайной структуры
    structure = random.choice(structures)
    # Формирование предложения
    sentence = structure.format(*words)
    return sentence


# Создание предложений для каждого списка
print()
for cluster, docs in clusters.items():
    sentence = create_sentence(top_n_keywords[cluster])
    print("Sentence 1:", sentence)




