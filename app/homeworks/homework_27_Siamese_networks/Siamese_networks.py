import os

# Отключение оптимизаций TF OneDNN
# (чтобы не добавлять в свойства Environment variables модуля)
# Почему-то эта штука срабатывает только в самом начале кода сразу после импорта os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import re
import time
import numpy as np
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.layers import Input, Embedding, LSTM, Dropout, Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt


# Загрузка ресурсов NLTK
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))  # Загрузка стоп-слов
lemmatizer = WordNetLemmatizer()  # Инициализация лемматизатора


# Функция предобработки текста
def preprocess_text(text):
    """Предобрабатывает текст: приводит к нижнему регистру, токенизирует, лемматизирует и удаляет стоп-слова."""
    # Удаление не-буквенных символов
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Приведение к нижнему регистру
    text = text.lower()
    # Токенизация
    tokens = word_tokenize(text)
    # Лемматизация и удаление стоп-слов
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)


# Создание базовой нейронной сети
def create_base_network(input_shape, vocab_size, embedding_dim):
    """Создает базовую нейронную сеть для обработки входных данных."""
    input = Input(shape=input_shape)
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, trainable=True)(input)
    x = LSTM(embedding_dim)(x)
    x = Dropout(0.5)(x)
    x = Dense(embedding_dim, kernel_regularizer=l2(0.01))(x)
    return Model(input, x)


def triplet_loss(y_true, y_pred, margin=0.1):
    """Вычисляет triplet loss."""
    anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]

    # Нормализация L2 эмбеддингов
    anchor = K.l2_normalize(anchor, axis=-1)
    positive = K.l2_normalize(positive, axis=-1)
    negative = K.l2_normalize(negative, axis=-1)

    # Вычисляем косинусные расстояния между эмбеддингами
    positive_distance = 1 - K.sum(anchor * positive, axis=-1, keepdims=True)
    negative_distance = 1 - K.sum(anchor * negative, axis=-1, keepdims=True)

    # Используем y_true для определения весов
    weights = K.cast(K.equal(y_true, 1), dtype='float32') + 1

    # Вычисляем triplet loss с учетом весов
    triplet_loss = K.mean(weights * K.maximum(positive_distance - negative_distance + margin, 0), axis=0)

    return triplet_loss


# Функция форматирования времени
def format_time(seconds):
    """Возвращает время в секундах в формате ЧЧ:ММ:СС."""
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return "{:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds))


# Функция форматирования текущего времени
def format_current_time():
    """Возвращает текущее время в формате ЧЧ:ММ:СС."""
    current_time_seconds = time.time()
    struct_time = time.localtime(current_time_seconds)
    formatted_time = time.strftime("%H:%M:%S", struct_time)
    return formatted_time


# Генерация синтетических данных для обучения triplet loss
def generate_triplets(num_triplets, input_shape, vocab_size):
    """Генерирует тройки (anchor, positive, negative) для обучения triplet loss."""
    anchors = np.random.randint(1, vocab_size, size=(num_triplets, *input_shape))
    positives = np.random.randint(1, vocab_size, size=(num_triplets, *input_shape))
    negatives = np.random.randint(1, vocab_size, size=(num_triplets, *input_shape))
    return [anchors, positives, negatives], np.zeros((num_triplets, 1))


# Разделение предложений на слова
def get_words(sentence):
    return set(sentence.lower().split())


# Гиперпараметры
input_shape = (10,)
embedding_dim = 50

# Примеры предложений
print('\nДанные для обучения:')
sentences = [
    "How are you doing today?",
    "I had a cup of coffee",
    "What is the weather like today?",
    "Is it going to rain today?",
    "What time is it now?"
]
print(sentences)

# Предобработка предложений
preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]

# Токенизация
tokenizer = Tokenizer(num_words=None)
tokenizer.fit_on_texts(preprocessed_sentences)
sequences = tokenizer.texts_to_sequences(preprocessed_sentences)
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

# Выравнивание последовательностей
padded_sequences = pad_sequences(sequences, maxlen=input_shape[0])

# Создание базовой нейронной сети
# ===============================
base_network = create_base_network(input_shape, vocab_size, embedding_dim)

# Входы для triplet loss
anchor_input = Input(shape=input_shape, name='anchor_input')
positive_input = Input(shape=input_shape, name='positive_input')
negative_input = Input(shape=input_shape, name='negative_input')

# Обработка входов через базовую нейронную сеть
encoded_anchor = base_network(anchor_input)
encoded_positive = base_network(positive_input)
encoded_negative = base_network(negative_input)

# Конкатенация выходов для вычисления triplet loss
triplet_loss_inputs = concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=1)

# Определение модели
# ==================
model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=triplet_loss_inputs)

# Настройка оптимизатора
initial_learning_rate = 0.0001  # 0.01 - 0.0001
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

# Компиляция модели с функцией потерь triplet loss
# ==================================================
model.compile(loss=triplet_loss, optimizer=optimizer)

# Вывод структуры модели
model.summary()

# Сохранение визуализации модели в файл
plot_model(model, to_file='../../data/siamese_network.png', show_shapes=True, show_layer_names=True)

# Генерация тренировочных троек
num_triplets = 100000
triplets, _ = generate_triplets(num_triplets, input_shape, vocab_size)

# Планировщик learning rate
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=7, min_lr=0.00001)

# Колбэк для сохранения истории обучения
csv_logger = CSVLogger('../../data/training.log')

print(f'\nОбучение началось в: {format_current_time()}')
start_time = time.time()

epochs = 10
batch_size = 256  # 512

# Обучение модели на тренировочных тройках
# ========================================
history = model.fit(triplets, np.zeros((num_triplets, 1)), epochs=epochs, batch_size=batch_size, callbacks=[reduce_lr, csv_logger])

print(f'Время обучения: {format_time(time.time() - start_time)}\n')

# Оценка модели на тестовых данных
num_triplets_test = 10000
triplets_test, _ = generate_triplets(num_triplets_test, input_shape, vocab_size)
loss = model.evaluate(triplets_test, np.zeros((num_triplets_test, 1)))
print(f"\nТестовая ошибка: {loss}")

# Использование модели
# ====================

# Пример использования модели для оценки сходства предложений
test_sentences = [
    "How are you doing today?",
    "What is the weather like today?",
    "Is it going to rain today?",
    "I had a cup of coffee",
    "What time is it now?"
]

# Предобработка тестовых предложений
preprocessed_test_sentences = [preprocess_text(sentence) for sentence in test_sentences]
test_sequences = tokenizer.texts_to_sequences(preprocessed_test_sentences)
padded_test_sequences = pad_sequences(test_sequences, maxlen=input_shape[0])

# Получение эмбеддингов для тестовых предложений
embeddings = base_network.predict(padded_test_sequences)

# Вычисление сходства между предложениями
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(embeddings)
print("\nМатрица сходства:")
print(similarity_matrix)

print("\nСходство каждого предложения с другими:")
for i, sentence in enumerate(test_sentences):
    for j, other_sentence in enumerate(test_sentences):
        if i != j:
            similarity = similarity_matrix[i, j] * 100
            print(f"Предложение '{sentence}' с предложением '{other_sentence}': {similarity:.2f}%")

print("\nСходство предложений самих с собой:")
for i, sentence in enumerate(test_sentences):
    similarity = similarity_matrix[i, i] * 100
    print(f"Предложение '{sentence}' с самим собой: {similarity:.2f}%")


# Создание списка слов для каждого предложения
word_sets = [get_words(sentence) for sentence in test_sentences]

# Цикл для сравнения предложений на основе общей similarity_matrix и вывода общих слов
print("\nСходство предложений по словам:")
for i, words_i in enumerate(word_sets):
    for j, words_j in enumerate(word_sets):
        if i != j and similarity_matrix[i, j] > 0:  # Использование similarity_matrix
            common_words = words_i.intersection(words_j)
            if common_words:
                print(f"Общие слова между предложением {i+1} и {j+1}: {', '.join(common_words)}")

# Визуализация истории обучения
plt.plot(history.history['loss'])
plt.title('Model Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper right')
plt.savefig('../../data/training_loss.png')
plt.show()
