import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt


# Чтение данных из файла combined.json
# Поля: "id", "title", "contents", "date", "topics", "components"
df = pd.read_json("../../data/combined.json", lines=True, encoding='utf-8')

# Выбор нужных полей для кластеризации (например, 'contents')
data = df['contents']

# Определение количества кластеров (можно выбрать любое подходящее значение)
num_clusters = 5

# Создание модели для Bag-of-Words
vectorizer_bow = CountVectorizer(max_df=0.5, max_features=10000, stop_words='english')
pipeline_bow = make_pipeline(vectorizer_bow, Normalizer())
X_bow = pipeline_bow.fit_transform(data)

# Создание модели для Term Frequency - Inverse Document Frequency (TF-IDF)
vectorizer_tf = TfidfVectorizer(max_df=0.5, max_features=10000, stop_words='english')
pipeline_tf = make_pipeline(vectorizer_tf, Normalizer())
X_tf = pipeline_tf.fit_transform(data)

# ===================FIT==================================

# Обучение модели KMeans для Bag-of-Words
kmeans_bow = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100, n_init=1)
kmeans_bow.fit(X_bow)

# Обучение модели KMeans для TF-IDF
kmeans_tf = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100, n_init=1)
kmeans_tf.fit(X_tf)

# ===================RESULTS==============================

# Вывод результатов для Bag-of-Words
print("Top terms per cluster for Bag-of-Words:")
order_centroids_bow = kmeans_bow.cluster_centers_.argsort()[:, ::-1]
terms_bow = vectorizer_bow.get_feature_names_out()
for i in range(num_clusters):
    print("Cluster %d:" % i),
    for ind in order_centroids_bow[i, :10]:
        print(' %s' % terms_bow[ind]),
    print()

# Вывод результатов для TF-IDF
print("\nTop terms per cluster for TF-IDF:")
order_centroids_tf = kmeans_tf.cluster_centers_.argsort()[:, ::-1]
terms_tf = vectorizer_tf.get_feature_names_out()
for i in range(num_clusters):
    print("Cluster %d:" % i),
    for ind in order_centroids_tf[i, :10]:
        print(' %s' % terms_tf[ind]),
    print()

# ===============METRICS=======================

# Вычисление метрик для Bag-of-Words
silhouette_score_bow = silhouette_score(X_bow, kmeans_bow.labels_)
inertia_bow = kmeans_bow.inertia_

# Вычисление метрик для TF-IDF
silhouette_score_tf = silhouette_score(X_tf, kmeans_tf.labels_)
inertia_tf = kmeans_tf.inertia_

# Вывод метрик
print("\nSilhouette Score for Bag-of-Words:", silhouette_score_bow)
print("Inertia for Bag-of-Words:", inertia_bow)

print("\nSilhouette Score for TF-IDF:", silhouette_score_tf)
print("Inertia for TF-IDF:", inertia_tf)

# ===============VISUAL========================

# Применение PCA к уменьшению размерности
pca = PCA(n_components=2)

# Применение PCA к данным Bag-of-Words
reduced_bow = pca.fit_transform(X_bow.toarray())

# Применение PCA к данным TF-IDF
reduced_tf = pca.fit_transform(X_tf.toarray())

# Визуализация результатов для Bag-of-Words
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(reduced_bow[:, 0], reduced_bow[:, 1], c=kmeans_bow.labels_, cmap='viridis', alpha=0.5)
plt.title('KMeans Clustering with Bag-of-Words')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Визуализация результатов для TF-IDF
plt.subplot(1, 2, 2)
plt.scatter(reduced_tf[:, 0], reduced_tf[:, 1], c=kmeans_tf.labels_, cmap='viridis', alpha=0.5)
plt.title('KMeans Clustering with TF-IDF')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.tight_layout()
plt.show()


