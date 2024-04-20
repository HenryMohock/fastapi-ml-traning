import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import Birch
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Чтение данных из файла combined.json
# Поля: "id", "title", "contents", "date", "topics", "components"
df = pd.read_json("combined.json", lines=True, encoding='utf-8')

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

# Обучение модели Birch для Bag-of-Words
birch_bow = Birch(n_clusters=num_clusters)
birch_bow.fit(X_bow)

# Обучение модели Birch для TF-IDF
birch_tf = Birch(n_clusters=num_clusters)
birch_tf.fit(X_tf)

# Вывод результатов для Bag-of-Words
print("Top terms per cluster for Bag-of-Words:")
for i in range(num_clusters):
    print("Cluster %d:" % i),
    cluster_center = birch_bow.subcluster_centers_[i]
    top_terms_indices = cluster_center.argsort()[-10:][::-1]
    terms_bow = vectorizer_bow.get_feature_names_out()
    top_terms = [terms_bow[index] for index in top_terms_indices]
    print(' '.join(top_terms))

# Вывод результатов для TF-IDF
print("\nTop terms per cluster for TF-IDF:")
for i in range(num_clusters):
    print("Cluster %d:" % i),
    cluster_center = birch_tf.subcluster_centers_[i]
    top_terms_indices = cluster_center.argsort()[-10:][::-1]
    terms_tf = vectorizer_tf.get_feature_names_out()
    top_terms = [terms_tf[index] for index in top_terms_indices]
    print(' '.join(top_terms))

# Вычисление метрик для Bag-of-Words
silhouette_score_bow = silhouette_score(X_bow.toarray(), birch_bow.labels_)
# Вычисление метрик для TF-IDF
silhouette_score_tf = silhouette_score(X_tf.toarray(), birch_tf.labels_)

# Вывод метрик
print("\nSilhouette Score for Bag-of-Words:", silhouette_score_bow)
print("Silhouette Score for TF-IDF:", silhouette_score_tf)

# Применение PCA к уменьшению размерности
pca = PCA(n_components=2)

# Применение PCA к данным Bag-of-Words
reduced_bow = pca.fit_transform(X_bow.toarray())

# Применение PCA к данным TF-IDF
reduced_tf = pca.fit_transform(X_tf.toarray())

# Визуализация результатов для Bag-of-Words
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(reduced_bow[:, 0], reduced_bow[:, 1], c=birch_bow.labels_, cmap='viridis', alpha=0.5)
plt.title('Birch Clustering with Bag-of-Words')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Визуализация результатов для TF-IDF
plt.subplot(1, 2, 2)
plt.scatter(reduced_tf[:, 0], reduced_tf[:, 1], c=birch_tf.labels_, cmap='viridis', alpha=0.5)
plt.title('Birch Clustering with TF-IDF')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.tight_layout()
plt.show()
