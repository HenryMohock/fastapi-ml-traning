import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Шаг 1: Импортируем необходимые библиотеки
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow as tf
print(f'Version Tensorflow: {tf.__version__}')

# Шаг 2: Загрузка и предобработка данных
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Преобразование формы данных
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Нормализация данных
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Преобразование меток в категориальный формат
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# Шаг 3: Визуализация данных
def plot_samples(samples, labels, predictions=None):
    """
        Визуализирует выборку изображений с соответствующими метками и (опционально) предсказаниями.

        Аргументы:
        samples (ndarray): Набор изображений для отображения, размерностью (N, 28, 28) или (N, 28, 28, 1).
        labels (ndarray): Набор истинных меток для изображений, размерностью (N, 10).
        predictions (ndarray, optional): Набор предсказанных меток для изображений, размерностью (N, 10).
                                         Если не передано, отображаются только истинные метки.

        Функция:
        - Создает фигуру размера 10x10.
        - Отображает первые 25 изображений из набора samples в сетке 5x5.
        - Убирает отметки на осях и сетку для каждого изображения.
        - Отображает изображение в черно-белом цветовом пространстве.
        - Если предсказания не переданы, подписывает каждое изображение его истинной меткой.
        - Если предсказания переданы, подписывает каждое изображение истинной и предсказанной меткой.
          Цвет подписи - синий, если предсказание верное, и красный, если предсказание неверное.
        - Показывает полученную фигуру.
    """
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(samples[i].reshape(28, 28), cmap=plt.cm.binary)
        if predictions is None:
            plt.xlabel(np.argmax(labels[i]))
        else:
            color = 'blue' if np.argmax(labels[i]) == np.argmax(predictions[i]) else 'red'
            plt.xlabel(f"True: {np.argmax(labels[i])}\nPred: {np.argmax(predictions[i])}", color=color)
    plt.show()


# Визуализация первых 25 изображений из тренировочного набора данных
plot_samples(x_train, y_train)

# Шаг 4: Построение модели
model = Sequential()
model.add(Input(shape=(28, 28, 1)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Шаг 5: Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Шаг 6: Обучение модели
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))

# Шаг 7: Оценка модели
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print()
print(f'Тестові втрати:   {test_loss}')
print(f'Тест на точність: {test_accuracy}')
print()

# Шаг 8: Визуализация предсказаний
predictions = model.predict(x_test)
plot_samples(x_test, y_test, predictions)
