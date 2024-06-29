import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from PIL import Image
import time


def format_time(seconds):
    """
    Форматирование секунд к виду ЧЧ:ММ:СС

      Возвращает строку отформатированного времени
    """
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return "{:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds))


def format_current_time():
    """
       Возвращает текущее время в формате ЧЧ:ММ:СС.

       Returns:
           str: Строка времени в формате ЧЧ:ММ:СС.
    """
    # Получаем текущее время в секундах с начала эпохи
    current_time_seconds = time.time()

    # Преобразуем время в struct_time
    struct_time = time.localtime(current_time_seconds)

    # Форматируем время в строку ЧЧ:ММ:СС
    formatted_time = time.strftime("%H:%M:%S", struct_time)

    return formatted_time


# Параметры для загрузки датасета
max_features = 20000  # Количество наиболее часто встречающихся слов
maxlen = 200  # Ограничение на количество слов в отзыве

# Загрузка датасета IMDB
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Приведение всех отзывов к одной длине
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# Построение модели
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(128)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Определение оптимизатора
optimizer = Adam(learning_rate=0.001)

# Компиляция модели
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# Callbacks для уменьшения скорости обучения
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.001)

# Обучение модели
batch_size = 32
epochs = 5

print()
print(f'Початок тренування моделі: {format_current_time()}')
start_time = time.time()
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    callbacks=[reduce_lr])
print(f'Час тренування моделі: {format_time(time.time() - start_time)}')
print()

# Оценка модели
print()
print('Оцінка моделі:')
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print(f"Тестові втрати: {score}")
print(f"Тест на точність: {acc}")

# Сохранение модели вместе с оптимизатором
model_path = '../../models/imdb_rnn_model_with_optimizer.keras'
model.save(model_path, include_optimizer=True)

# Визуализация
plt.figure(figsize=(12, 5))

# График точности
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# График потерь
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Сохранение графика
visualization_path = '../../data/training_history_rnn_model.png'
plt.tight_layout()
plt.savefig(visualization_path)
plt.show()

# Визуализация структуры модели с использованием plot_model напрямую и отображение картинки
plot_model(model, to_file='../../data/rnn_model_structure.png', show_shapes=True, show_layer_names=True)

# Отображение изображения с помощью Pillow
image = Image.open('../../data/rnn_model_structure.png')
image.show()

# Пример использования модели
# ============================================

# Тестирование на новых данных
test_reviews = [
    "The movie was fantastic! I really enjoyed it.",
    "The movie was terrible. I hated it."
]
print()
print('Дані для тестування моделі:')
print(test_reviews)

word_index = imdb.get_word_index()

# Подготовка текстов для предсказания
sequences = [
    pad_sequences([[word_index.get(w, 0) for w in review.lower().split() if word_index.get(w, 0) < max_features]], maxlen=maxlen)
    for review in test_reviews
]

# Прогноз
print()
print('Прогноз')
for review, sequence in zip(test_reviews, sequences):
    prediction = model.predict(sequence)
    sentiment = "позитивний" if prediction[0][0] > 0.5 else "негативний"
    print(f"Огляд: {review}\nПрогнозування (ймовірність позитивної рецензії): {prediction[0][0]}\nСентимент: {sentiment}\n")

