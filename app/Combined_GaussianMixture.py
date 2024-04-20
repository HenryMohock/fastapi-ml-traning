import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Чтение данных из файла JSON
df = pd.read_json("combined.json", lines=True, encoding='utf-8')

# Выбор нужных полей для кластеризации (например, 'contents')
data = df['contents']

# Определение количества кластеров
num_clusters = 3

# Создание модели для Bag-of-Words
vectorizer_bow = CountVectorizer(max_df=0.5, max_features=10000, stop_words='english')
pipeline_bow = make_pipeline(vectorizer_bow, Normalizer())
X_bow = pipeline_bow.fit_transform(data)

# Создание модели для Term Frequency - Inverse Document Frequency (TF-IDF)
vectorizer_tf = TfidfVectorizer(max_df=0.5, max_features=10000, stop_words='english')
pipeline_tf = make_pipeline(vectorizer_tf, Normalizer())
X_tf = pipeline_tf.fit_transform(data)

# Обучение модели GaussianMixture для Bag-of-Words
gmm_bow = GaussianMixture(n_components=num_clusters, random_state=42, n_init=1, covariance_type='diag')
gmm_bow.fit(X_bow.toarray())

# Обучение модели GaussianMixture для TF-IDF
gmm_tf = GaussianMixture(n_components=num_clusters, random_state=42, n_init=1, covariance_type='diag')
gmm_tf.fit(X_tf.toarray())

# Визуализация результатов на PCA-reduced данных
def visualize_clusters(X, labels, title):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.toarray())
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

visualize_clusters(X_bow, gmm_bow.predict(X_bow.toarray()), 'GaussianMixture Clustering with Bag-of-Words')
visualize_clusters(X_tf, gmm_tf.predict(X_tf.toarray()), 'GaussianMixture Clustering with TF-IDF')
