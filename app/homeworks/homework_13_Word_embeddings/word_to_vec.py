import gensim.downloader as api
import numpy as np
import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def format_time(seconds):
    """
    Форматирование секунд к виду ЧЧ:ММ:СС

      Возвращает строку отформатированного времени
    """
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return "{:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds))


# ================== ЗАГРУЗКА МОДЕЛИ ============================

start_time = time.time()
name_model = 'word2vec-google-news-300'
model = api.load(name_model)
print('Модель загружена')
print(f'Класс модели: {type(model)}')
print("- Время загрузки модели: ", format_time(time.time() - start_time))
print()

# ================== ПОИСК ПОХОЖИХ СЛОВ ============================

# Выбираем 4 слова
words = ['king', 'computer', 'music', 'car']
print(f'Выбранные слова: {words}')

print('Находим слова, наиболее похожие на выбранные')
print()
for word in words:
    similar_words = model.most_similar(word, topn=5)
    print(f"Слова, наиболее похожие на '{word}':")
    for similar_word, similarity in similar_words:
        print(f"  {similar_word} (сходство: {similarity * 100:.2f}%)")
    print()

# ================== РАСЧЕТ ПОДОБИЯ ПАР СЛОВ ============================

print('Список пар слов для вычисления сходства')
word_pairs = [
    ('car', 'automobile'),
    ('plane', 'airplane'),
    ('dog', 'cat'),
    ('hotel', 'motel')
]
print(word_pairs)
print()

print('Вычисление сходства для каждой пары:')
for word1, word2 in word_pairs:
    similarity = model.similarity(word1, word2)
    # print(f'Similarity between "{word1}" and "{word2}": {similarity:.4f}')
    print(f'Сходство между "{word1}" and "{word2}": {similarity * 100:.2f}%')

print()
print('Извлечение векторных представлений для слов:')
words = [word for pair in word_pairs for word in pair]
sample_vectors = np.array([model[word] for word in words])
print(sample_vectors.shape)  # 8 слов, 300 измерений
print()

pca = PCA()
pca.fit(sample_vectors)
cumulative_variance_explained = np.cumsum(pca.explained_variance_ratio_)*100

# Построение графика:
plt.figure(num='Кумулятивна дисперсія')
plt.plot(range(1, len(cumulative_variance_explained) + 1), cumulative_variance_explained, '-o')
plt.xlabel("Кількість головних компонентів")
plt.ylabel("Пояснення кумулятивної дисперсії (%)")
plt.title("Кумулятивна дисперсія, пояснена головними компонентами")
plt.show()

result = pca.transform(sample_vectors)

# Построение графика:
plt.figure(num='Новий векторний простір')
plt.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    if i % 2 == 0:
        plt.annotate(words[i], xy=(result[i, 0], result[i, 1]), xytext=(0, 15), textcoords='offset points', ha='right',
                     va='top')
    else:
        plt.annotate(words[i], xy=(result[i, 0], result[i, 1]), xytext=(0, -15), textcoords='offset points', ha='left',
                     va='bottom')

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Новий векторний простір")
plt.ylim(-2.5, None)
plt.xlim(None, 2.5)
plt.show()

# ================== СЕМАНТИЧЕСКИЕ ПРЕОБРАЗОВАНИЯ ============================

print('Семантическое преобразование: король - мужчина + женщина')
result = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=5)

# Вывод результатов
print('Ближайшие слова к "король" - "мужчина" + "женщина":')
for word, similarity in result:
    print(f'{word}: {similarity * 100:.2f}%')
print()

print('Семантическое преобразование: страна - столица')
countries_capitals = [
    ('France', 'Paris'),
    ('Germany', 'Berlin'),
    ('Japan', 'Tokyo'),
    ('Ukraine', 'Kyiv'),
    ('Brazil', 'Brasília')
]
print()

print('Проверка ближайших слов для каждой пары')
print()
for country, capital in countries_capitals:
    if country in model and capital in model:
        result = model.most_similar(positive=[capital, country], topn=3)
        print(f'Столица {capital} ближайшие слова к стране {country}:')
        for word, similarity in result:
            print(f'{word}: {similarity * 100:.2f}%')
        print()
    else:
        print(f'Для страны {country} или столицы {capital} нет достаточных данных в модели.\n')

print('Семантическое преобразование: единственное число - множественное число')
word_singular_plural = [
    'car',
    'dog',
    'house',
    'cat',
    'child'
]
print()

print('Проверка ближайших слов для каждого слова')
print()
for word in word_singular_plural:
    plural_word = model.most_similar(positive=[word, 'plural'], topn=1)
    print(f'Единственное число "{word}" ближайшее слово к множественному числу:')
    print(f'{plural_word[0][0]}: {plural_word[0][1] * 100:.2f}%')
    print()
