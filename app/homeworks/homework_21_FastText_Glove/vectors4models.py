from gensim.models import Word2Vec
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.models import FastText
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns


def load_glove_model(file_path):
    """
    Загрузка предобученной модели GloVe из файла.

    Args:
    file_path (str): Путь к файлу с предобученной моделью GloVe.

    Returns:
    dict: Словарь, где ключ - слово, значение - векторное представление слова.
    """
    glove_model = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            glove_model[word] = vector
    return glove_model


def get_sentence_vector_w2v(model, sentence):
    """
    Преобразование предложения в векторное представление с использованием модели Word2Vec.

    Args:
    model (gensim.models.Word2Vec): Обученная модель Word2Vec.
    sentence (str): Предложение для векторизации.

    Returns:
    numpy.ndarray: Средний вектор слов предложения.
    """
    words = sentence.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(word_vectors, axis=0)


def get_sentence_vector_ft(model, sentence):
    """
    Преобразование предложения в векторное представление с использованием модели FastText.

    Args:
    model (gensim.models.FastText): Обученная модель FastText.
    sentence (str): Предложение для векторизации.

    Returns:
    numpy.ndarray: Средний вектор слов предложения.
    """
    words = sentence.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(word_vectors, axis=0)


def get_sentence_vector_glove(model, sentence):
    """
    Преобразование предложения в векторное представление с использованием модели GloVe.

    Args:
    model (dict): Словарь с предобученными векторами GloVe.
    sentence (str): Предложение для векторизации.

    Returns:
    numpy.ndarray: Средний вектор слов предложения.
    """
    words = sentence.split()
    word_vectors = [model[word] for word in words if word in model]
    return np.mean(word_vectors, axis=0)


def compare_vectors(vectors):
    """
    Вычисление косинусной схожести между векторами предложений.

    Args:
    vectors (list): Список векторов предложений.

    Returns:
    numpy.ndarray: Матрица косинусной схожести.
    """
    similarity_matrix = cosine_similarity(vectors)
    return similarity_matrix


sentences = [
    "1. The quick brown fox jumps over the lazy dog.",                         # Быстрая коричневая лиса прыгает через ленивую собаку.
    "1. The dogs lazily glanced at the red fox running out of the forest.",    # Собаки лениво поглядывали на выбежавшую из леса рыжую лису.
    "2. Never judge a book by its cover.",                                     # Не суди книгу по ее обложке.
    "2. The cover of this book looked more valuable than its contents.",       # Обложка этой книги выглядела ценнее ее содержания.
    "3. He walked thousands of miles.",                                        # Он прошел пешком тысяи миль
    "3. A journey of a thousand miles begins with a single step.",             # Путь в тысячу миль начинается с одного шага.
    "4. To be or not to be, that is the question.",                            # Быть или не быть, вот в чем вопрос.
    "4. He had to ask himself a difficult question.",                          # Ему пришлось задать себе сложный вопрос.
    "5. The ring on her little finger was not gold, but it sparkled brightly.",  # Кольцо на ее мизинце не было золотым, но ярко блестело
    "5. All that glitters is not gold."                                        # Не все, что блестит, золото.
]


# Обучение модели Word2Vec
word2vec_model = Word2Vec(sentences=[s.split() for s in sentences], vector_size=100, window=5, min_count=1, workers=4)

# Обучение модели Doc2Vec
documents = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(sentences)]
doc2vec_model = Doc2Vec(documents, vector_size=100, window=5, min_count=1, workers=4)

# Обучение модели FastText
fasttext_model = FastText(sentences=[s.split() for s in sentences], vector_size=100, window=5, min_count=1, workers=4, sg=1)

# Объявление модели Glove
# https://www.kaggle.com/datasets/sawarn69/glove6b100dtxt?resource=download
glove_model = load_glove_model('../../data/glove.6B.100d.txt')

sentence_vectors_w2v = [get_sentence_vector_w2v(word2vec_model, sentence) for sentence in sentences]
sentence_vectors_d2v = [doc2vec_model.infer_vector(sentence.split()) for sentence in sentences]
sentence_vectors_ft = [get_sentence_vector_ft(fasttext_model, sentence) for sentence in sentences]
sentence_vectors_glove = [get_sentence_vector_glove(glove_model, sentence) for sentence in sentences]

similarity_w2v = compare_vectors(sentence_vectors_w2v)
similarity_d2v = compare_vectors(sentence_vectors_d2v)
similarity_ft = compare_vectors(sentence_vectors_ft)
similarity_glove = compare_vectors(sentence_vectors_glove)

print()
print("Косинусна подібність - Word2Vec")
print(similarity_w2v)

print()
print("Косинусна подібність - Doc2Vec")
print(similarity_d2v)

print()
print("Косинусна подібність - FastText")
print(similarity_ft)

print()
print("Косинусна подібність - GloVe")
print(similarity_glove)


# Создание графиков
plt.figure(figsize=(16, 12))

# Word2Vec
plt.subplot(2, 2, 1)
sns.heatmap(similarity_w2v, annot=True, cmap='coolwarm', xticklabels=sentences, yticklabels=sentences)
plt.title('Cosine Similarity - Word2Vec')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Doc2Vec
plt.subplot(2, 2, 2)
sns.heatmap(similarity_d2v, annot=True, cmap='coolwarm', xticklabels=sentences, yticklabels=sentences)
plt.title('Cosine Similarity - Doc2Vec')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# FastText
plt.subplot(2, 2, 3)
sns.heatmap(similarity_ft, annot=True, cmap='coolwarm', xticklabels=sentences, yticklabels=sentences)
plt.title('Cosine Similarity - FastText')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# GloVe
plt.subplot(2, 2, 4)
sns.heatmap(similarity_glove, annot=True, cmap='coolwarm', xticklabels=sentences, yticklabels=sentences)
plt.title('Cosine Similarity - GloVe')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()


# Вычисление средней косинусной схожести для каждой модели
avg_similarity_w2v = np.mean(similarity_w2v[np.triu_indices_from(similarity_w2v, k=1)])
avg_similarity_d2v = np.mean(similarity_d2v[np.triu_indices_from(similarity_d2v, k=1)])
avg_similarity_ft = np.mean(similarity_ft[np.triu_indices_from(similarity_ft, k=1)])
avg_similarity_glove = np.mean(similarity_glove[np.triu_indices_from(similarity_glove, k=1)])

# Проверка неотрицательности значений
avg_similarity_w2v = max(avg_similarity_w2v, 0)
avg_similarity_d2v = max(avg_similarity_d2v, 0)
avg_similarity_ft = max(avg_similarity_ft, 0)
avg_similarity_glove = max(avg_similarity_glove, 0)

# Данные для круговой диаграммы
labels = ['Word2Vec', 'Doc2Vec', 'FastText', 'GloVe']
sizes = [avg_similarity_w2v, avg_similarity_d2v, avg_similarity_ft, avg_similarity_glove]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0.1, 0, 0, 0)  # Выделить первый кусок (Word2Vec)

# Создание круговой диаграммы
plt.figure(figsize=(8, 8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')  # Обеспечить круговую форму
plt.title('Average Cosine Similarity by Model')
plt.show()

