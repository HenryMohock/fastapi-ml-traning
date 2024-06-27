import os
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.datasets import imdb
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam  # , RMSprop, Nadam
from tensorflow.keras.utils import model_to_dot
import matplotlib.pyplot as plt
import graphviz
from PIL import Image


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
print(f'Version Tensorflow: {tf.__version__}')

# Задание гиперпараметров
max_features = 20000  # Размер словаря
max_len = 400  # Ограничение на длину последовательностей (отзывов)
embedding_dims = 128  # Размерность эмбеддинга
filters = 128  # Количество фильтров в сверточном слое
kernel_size = 5  # Размер ядра свертки
hidden_dims = 128  # Размер скрытого слоя
batch_size = 32
epochs = 7  # 5, 6, 10

# Загрузка и подготовка данных
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

# Построение модели
model = Sequential()
model.add(Embedding(max_features, embedding_dims, input_length=max_len))
model.add(Dropout(0.2))
model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1, kernel_regularizer=l2(0.01)))
model.add(GlobalMaxPooling1D())
model.add(Dense(hidden_dims, kernel_regularizer=l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Компиляция модели
optimizer = Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Обучение модели
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), verbose=1)

# Сохранение модели
model.save('../../data/imdb_cnn_model.keras')

# Загрузка модели
loaded_model = tf.keras.models.load_model('../../data/imdb_cnn_model.keras')

# Использование модели для предсказаний
predictions = loaded_model.predict(x_test)

# Оценка точности модели
loss, accuracy = loaded_model.evaluate(x_test, y_test, verbose=1)
print()
print(f'Тестові втрати: {loss}')
print(f'Тест на точність: {accuracy}')

# Визуализация графиков обучения
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
plt.savefig('../../data/training_history_cnn_model.png')

# Визуализация структуры модели с использованием graphviz напрямую и отображение картинки
dot_model = model_to_dot(model, show_shapes=True, show_layer_names=True).to_string()
graph = graphviz.Source(dot_model)

# Сохранение в формате PNG
graph.render('../../data/model_structure', format='png')

# Отображение изображения с помощью Pillow
image = Image.open('../../data/model_structure.png')
image.show()

# Пример использования сохраненной модели
# ========================================

# Пример новых данных для предсказания
print()
print('Дані для передбачення:')
new_data = ["The movie was fantastic! I really enjoyed it.",
            "The movie was awful and boring. I would not recommend it to anyone."]
print(new_data)
print()

# Загрузка словаря IMDB для преобразования текстов
word_index = imdb.get_word_index()


# Преобразование новых данных в числовые последовательности
def encode_text(text):
    tokens = text.lower().split()
    encoded = [word_index.get(word, 2) for word in tokens]  # 2 - это индекс для "unknown" слов
    return encoded


encoded_data = [encode_text(review) for review in new_data]

# Дополнение или обрезка последовательностей до нужной длины
padded_data = sequence.pad_sequences(encoded_data, maxlen=max_len)

# Выполнение предсказания
predictions = loaded_model.predict(padded_data)

# Интерпретация результатов
print()
for i, prediction in enumerate(predictions):
    if prediction[0] > 0.5:
        sentiment = 'Позитивний'
    else:
        sentiment = 'Негативний'

    print(f"Огляд: {new_data[i]}")
    print(f"Сентимент: {sentiment} (Впевненість: {prediction[0]:.2f})\n")
