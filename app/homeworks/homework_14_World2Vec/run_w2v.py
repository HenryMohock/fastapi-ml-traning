import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
from Preparing_text import Preliminary_preparation_data
import re
import time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


def format_time(seconds):
    """
    Форматирование секунд к виду ЧЧ:ММ:СС

      Возвращает строку отформатированного времени
    """
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return "{:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds))


def get_data(word_count: int = 0, path: str = ''):
    # Параметры текста
    encoding = 'utf8'
    lang = 'english'

    # Путь к файлу CSV
    if path == '':
        file_path = "../../data/Bible_NIV.txt"
    else:
        file_path = path

    # Регулярное выражение для удаления прочих символов
    regular_expression = r'blah|_|fffc|br|oz|aam'

    # Порог встречаемости слов
    threshold = 5

    # Метод обрезки слов (lemmatization или stemming или ничего '')
    method = 'lemmatization'

    # Инициализация экземпляра Preliminary_preparation_data
    pre_processor = Preliminary_preparation_data(file_path=file_path, regular_expression=regular_expression,
                                                 threshold=threshold, method=method, encoding=encoding, lang=lang)

    # Получаем текст из Preliminary_preparation_data
    text = pre_processor.get_text
    if word_count != 0:
        text = " ".join(text.split()[:word_count])

    # Получаем текст из Preliminary_preparation_data
    roh_text = pre_processor.roh_text
    if word_count != 0:
        roh_text = " ".join(roh_text.split()[:word_count])

    return text, roh_text


def apply_regex(text: str):
    """
    Обработка текста регулярными выражениями.

    согласно переданному регулярному выражению изменяется текст
    """
    text = re.sub(r'(\d)([^\d\s])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text)  # замена нескольких пробелов одним
    text = re.sub(r'\d+', '', text)  # числа
    text = re.sub(r'\s*GENESIS\s*', '', text)
    text = re.sub(r'"', '', text)  # двойные кавычки
    text = re.sub(r'-', '', text)  # тире
    text = re.sub(r'—', '', text)  # длинное тире
    text = re.sub(r"['“”]", '', text)  # двойные кавычки
    text = re.sub(r"['‘’]", '', text)  # одинарные кавычки
    text = re.sub(r'[^\S\n]+', ' ', text)  # пробелы
    return text.lower()


def plot_tsne(model, words):
    labels = []
    tokens = []

    for word in words:
        tokens.append(model.wv[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=5, n_components=2, init='pca', n_iter=2500, random_state=23)

    # Преобразуем список векторов tokens в массив NumPy
    tokens = np.array(tokens)

    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


# Загружаем данные
file_path = "../../data/NIV_Bible.txt"
text, roh_text = get_data(word_count=0, path=file_path)

# Предобработка текста: регулярные выражения, токенизация
text = apply_regex(roh_text)
nltk.download('punkt')
sentences = sent_tokenize(text)
print(f"Количество предложений: {len(sentences)}")
tokenized_sentences = [word_tokenize(sentence) for sentence in sentences if sentence]
print(f"Количество токенизированных предложений: {len(tokenized_sentences)}")

if not tokenized_sentences:
    print("Корпус данных пуст или не удалось токенизировать текст.")
    exit()

# Обучение модели Word2Vec
try:
    start_time = time.time()
    model = Word2Vec(
        sentences=tokenized_sentences,  # данные для обучения
        vector_size=300,                # размерность векторов. 300 считается оптимальным. Пробовал иные значения. Таке.
        window=5,                       # максимальное расстояние в словах между текущим и прогнозируемым словом
        min_count=1,                    # минимальное количество вхождений слова в корпусе
        workers=4,                      # количество потоков параллельной обработки данных
        compute_loss=True,              # указание модели вычислять потери во время обучения
        sg=1,                           # алгоритмы обучения 0 - CBOW, 1 - Skip-gram
        epochs=30                      # количество итераций (эпох) обучения над данными (параметр заметно работает)
    )
    print("Модель успешно обучена.")
    print("- Время обучения модели: ", format_time(time.time() - start_time))
except Exception as e:
    print(f"Ошибка при обучении модели: {e}")
    exit()

# Получение значений потерь при обучении (т.к. compute_loss=True)
print()
training_loss = model.get_latest_training_loss()
print(f"Значение потерь при обучении: {training_loss}")

# Сохранение модели
model_path = "../../models/word2vec/word2vec.model"
try:
    model.save(model_path)
    print(f"Модель сохранена по пути: {model_path}")
except Exception as e:
    print(f"Ошибка при сохранении модели: {e}")
    exit()

# Загрузка модели
try:
    model = Word2Vec.load(model_path)
    print("Модель успешно загружена.")
except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")
    exit()

# Извлечение словаря модели:
print()
print('Словарь модели (первые 40 слов):')
for index, word in enumerate(model.wv.index_to_key):
    if index == 40:
        break
    print(f"word #{index}/{len(model.wv.index_to_key)} is {word}")

# Использование модели
word = 'lord'
if word in model.wv:

    # Получить вектор слова
    word_vector = model.wv[word]
    print()
    print(f"Вектор слова '{word}': {word_vector}")

    # Найти похожие слова
    similar_words = model.wv.most_similar(word)
    print()
    print(f"Слова, похожие на '{word}': {similar_words}")

else:
    print(f"Слово '{word}' не найдено в модели.")


# Варианты определения схожести слов
# Имена бога в Библии
print()
print('Схожесть имен Бога в Библии:')
pairs = [
    ('god', 'lord'),
    ('god', 'jacob'),
    ('god', 'father'),
    ('god', 'abel'),
    ('god', 'noah'),
    ('god', 'japheth'),
    ('god', 'boris'),
]

for word1, word2 in pairs:
    if word1 in model.wv and word2 in model.wv:
        similarity = model.wv.similarity(word1, word2) * 100  # Преобразование в проценты
        print(f"Схожесть между '{word1}' и '{word2}': {similarity:.3f} %")
    else:
        print(f"Одно или оба слова отсутствуют в словаре: '{word1}', '{word2}'")

# Вывод 5 самых похожих на тройку слов
print()
print('Вывод 5 самых похожих на пару-тройку слов')
for word in ['lord', 'god', 'father']:
    if word in model.wv:
        similar_words = model.wv.most_similar(word, topn=5)  # Найти 5 самых похожих слов
        print(f"5 слов, похожих на '{word}': {similar_words}")
    else:
        print(f"Слово '{word}' не найдено в модели.")

# Проверка слов, которые не принадлежат последовательности
print()
words = ['fire', 'water', 'land', 'sea', 'earth', 'light']
print(f'Проверка слов, которые не принадлежат последовательности:\n {words}')
try:
    odd_one_out = model.wv.doesnt_match(words)
    print(f"Слово, которое не принадлежит последовательности: {odd_one_out}")
except KeyError as e:
    print(f"Ошибка: одно или несколько слов отсутствуют в словаре: {e}")


# Визуализация векторных представлений слов

# Список слов для визуализации
words_to_visualize = ['god', 'lord', 'jacob',
                      'father', 'abel', 'noah',
                      'has', 'fire', 'water',
                      'land', 'sea', 'earth', 'light']
plot_tsne(model, words_to_visualize)

