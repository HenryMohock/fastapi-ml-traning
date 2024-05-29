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


# Шаг 1: Загрузка датасета IMDB и предварительная обработка
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


# Функция предварительной обработки текста
def preprocess_text(text):
    # Удаление HTML-тегов
    text = BeautifulSoup(text, "html.parser").get_text()
    # Удаление строк вида '<br /><br />'
    text = re.sub(r'<br\s*/><br\s*/>', ' ', text)
    # Приведение к нижнему регистру
    text = text.lower()
    # Удаление пунктуации
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Токенизация и удаление стоп-слов
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return tokens


# Применение конструкции List Comprehension для предварительной обработки к предложениям
# enumerate индексирует текст, а в каждой итерации цикла происходит препроцессинг предложения c помощью preprocess_text
documents = [TaggedDocument(words=preprocess_text(text), tags=[str(i)]) for i, text in enumerate(texts)]

# Шаг 2: Обучение модели Doc2Vec
model = Doc2Vec(
    documents=documents,
    vector_size=95,     # размер векторного представления (количество измерений в векторе для предложений) (60)
    window=5,           # сколько слов рядом (справа и слева) с текущим словом будет учитываться при обучении (3)
    min_count=2,        # минимальное количество упоминаний слова в коллекции документов для его включения в обучение(2)
    workers=6,          # количество потоков, которые будут использоваться для тренировки модели (6)

    epochs=6,           # количество проходов по корпусу документов во время тренировки
    #dm=0,               # алгоритмы: 1 для Distributed Memory (DM) и 0 для Distributed Bag of Words (DBOW)
    #dbow_words=1,       # если установлено в 1, модель также будет обучать вектора слов во время тренировки DBOW (только если dm=0)
    #alpha=0.025,        # начальная скорость обучения (0.025)
    #min_alpha=0.0001,   # минимальная скорость обучения. Она уменьшается с каждым шагом обучения (0.0001)
    #negative=7,         # если больше 0, будет использоваться негативное сэмплирование, с указанным количеством "негативных" слов. (5)
    #hs=0,               # если установлено в 1, будет использоваться Softmax иерархического кодирования; если в 0, используется негативное сэмплирование
    #sample=0.001        # порог для конфигурации случайного down-sampling высокочастотных слов (0.001)
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
