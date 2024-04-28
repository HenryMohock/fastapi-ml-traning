import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

# Чтение данных из файла combined.json
df = pd.read_json("../../data/combined.json", lines=True, encoding='utf-8')

# Выбор нужных полей для кластеризации (например, 'contents')
data = df['contents']

# Создание модели для Bag-of-Words
vectorizer_bow = CountVectorizer(max_df=0.5, max_features=10000, stop_words='english')
pipeline_bow = make_pipeline(vectorizer_bow, Normalizer())
X_bow = pipeline_bow.fit_transform(data)

# Создание модели для Term Frequency - Inverse Document Frequency (TF-IDF)
vectorizer_tf = TfidfVectorizer(max_df=0.5, max_features=10000, stop_words='english')
pipeline_tf = make_pipeline(vectorizer_tf, Normalizer())
X_tf = pipeline_tf.fit_transform(data)

# Обучение модели DBSCAN для Bag-of-Words
dbscan_bow = DBSCAN(eps=0.7, min_samples=5)
dbscan_labels_bow = dbscan_bow.fit_predict(X_bow)

# Проверка наличия более чем одного кластера
if len(set(dbscan_labels_bow)) > 1:
    silhouette_bow = silhouette_score(X_bow, dbscan_labels_bow)
    davies_bouldin_bow = davies_bouldin_score(X_bow.toarray(), dbscan_labels_bow)
else:
    silhouette_bow = None
    davies_bouldin_bow = None

# Обучение модели DBSCAN для TF-IDF
dbscan_tf = DBSCAN(eps=0.7, min_samples=5)
dbscan_labels_tf = dbscan_tf.fit_predict(X_tf)

# Проверка наличия более чем одного кластера
if len(set(dbscan_labels_tf)) > 1:
    silhouette_tf = silhouette_score(X_tf, dbscan_labels_tf)
    davies_bouldin_tf = davies_bouldin_score(X_tf.toarray(), dbscan_labels_tf)
else:
    silhouette_tf = None
    davies_bouldin_tf = None

# Вывод результатов для Bag-of-Words
print("Results for Bag-of-Words:")
print("Number of clusters:", len(set(dbscan_labels_bow)) - (1 if -1 in dbscan_labels_bow else 0))
print("Silhouette Score:", silhouette_bow)
print("Davies-Bouldin Index:", davies_bouldin_bow)

# Вывод результатов для TF-IDF
print("\nResults for TF-IDF:")
print("Number of clusters:", len(set(dbscan_labels_tf)) - (1 if -1 in dbscan_labels_tf else 0))
print("Silhouette Score:", silhouette_tf)
print("Davies-Bouldin Index:", davies_bouldin_tf)

# Применение PCA для визуализации результатов
pca = PCA(n_components=2)

# Стандартизация данных перед применением PCA
scaler = StandardScaler()
X_bow_scaled = scaler.fit_transform(X_bow.toarray())
X_tf_scaled = scaler.fit_transform(X_tf.toarray())

# Уменьшение размерности с PCA для Bag-of-Words
reduced_bow = pca.fit_transform(X_bow_scaled)

# Уменьшение размерности с PCA для TF-IDF
reduced_tf = pca.fit_transform(X_tf_scaled)

# Визуализация результатов для Bag-of-Words
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(reduced_bow[:, 0], reduced_bow[:, 1], c=dbscan_labels_bow, cmap='viridis', alpha=0.5)
plt.title('DBSCAN Clustering with Bag-of-Words')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Визуализация результатов для TF-IDF
plt.subplot(1, 2, 2)
plt.scatter(reduced_tf[:, 0], reduced_tf[:, 1], c=dbscan_labels_tf, cmap='viridis', alpha=0.5)
plt.title('DBSCAN Clustering with TF-IDF')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.tight_layout()
plt.show()
