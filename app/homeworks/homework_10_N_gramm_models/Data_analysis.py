'''
Виды анализа текста:

1. Частотный анализ: Определение наиболее часто встречающихся n-грамм в тексте или корпусе текстов.
Этот анализ может помочь выявить ключевые словосочетания, термины или фразы.

2. Синтаксический анализ: Использование n-грамм для анализа синтаксической структуры предложений.
Это может включать выявление шаблонов словосочетаний, анализ зависимостей между словами или определение
частей речи в контексте.

3. Контекстный анализ: Использование n-грамм для анализа контекста слов или фраз. Это может включать в себя
выявление контекста вокруг определенного слова или фразы для понимания его смысла или значения.

4. Семантический анализ: Анализ n-грамм с целью выявления семантических отношений между словами или фразами.
Это может включать в себя определение синонимов, антонимов, гипонимов или гиперонимов.

5. Морфологический анализ: Использование n-грамм для анализа морфологических особенностей языка,
таких как флективная или деривационная морфология. Это может включать в себя анализ форм слова или
выявление морфологических паттернов.
'''

from nltk import ngrams
from app.homeworks.homework_10_N_gramm_models.Preparing_text import Preliminary_preparation_data
from collections import Counter
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import sys


def get_data(word_count: int = 0):
    # Параметры текста
    encoding = 'utf8'
    lang = 'english'

    # Путь к файлу CSV
    file_path = "../../data/Bible_NIV.txt"

    # Регулярное выражение для удаления прочих символов
    regular_expression = r'blah|_|fffc|br|oz'

    # Порог встречаемости слов
    threshold = 5

    # Метод обрезки слов (lemmatization или стемминг)
    method = 'lemmatization'

    # Инициализация экземпляра CsvPreProcessing
    csv_processor = Preliminary_preparation_data(file_path=file_path, regular_expression=regular_expression,
                                                 threshold=threshold, method=method, encoding=encoding, lang=lang)

    # Получаем текст из CsvPreProcessing
    text = csv_processor.get_text
    if word_count != 0:
        text = " ".join(text.split()[:word_count])

    return text


def top_words(sentence, excluded_words=[], min_word_length=4, top_n=50):
    # Разбить предложение на слова, учитывая разделитель пробела
    words = sentence.split()

    # Исключить слова из списка исключенных слов и слова короче min_word_length
    # words = [word for word in words if word.lower() not in excluded_words]
    words = [word for word in words if word.lower() not in excluded_words and len(word) > min_word_length]

    # Подсчитать частоту встречаемости слов
    word_counter = Counter(words)

    # Вернуть список наиболее встречаемых слов
    return word_counter.most_common(top_n)


def n_grams(sentence, n=2):
    grams = ngrams(sentence.split(), n)
    return grams


def most_common_ngrams(ngrams, top_n=10):
    ngrams_list = list(ngrams)  # Преобразование объекта zip в список кортежей
    counter = Counter(ngrams_list)
    most_common = counter.most_common(top_n)
    return most_common


def analyze_context(sentence, top_words, context_window=2):
    result = {}
    for word, _ in top_words:
        word_bigrams = []
        for i in range(len(sentence)):
            if sentence[i] == word:
                start_index = max(0, i - context_window)
                end_index = min(len(sentence), i + context_window + 1)
                context = sentence[start_index:i] + sentence[i + 1:end_index]
                word_bigrams.append(context)
        result[word] = word_bigrams
    return result


def syntactic_analysis(sentence, word_count=100000):
    # Загрузка модели языка для английского языка
    nlp = spacy.load("en_core_web_sm")

    # Разбиваем предложение на пакеты по word_count слов
    batches = [sentence[i:i+word_count] for i in range(0, len(sentence), word_count)]

    # Список для хранения результатов анализа
    analysis_results = []

    # Анализируем каждый пакет
    for batch in batches:
        # Анализ с использованием SpaCy
        doc = nlp(batch)
        # Получаем зависимости между токенами
        for token in doc:
            analysis_results.append({
                "token_text": token.text,       # текст токена
                "dependency": token.dep_,       # зависимость
                "head_text": token.head.text    # текст заголовка
            })

    return analysis_results


def semantic_analysis_ngrams(bigrams):
    glove_path = "../../data/glove.6B.100d.txt"
    glove_embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.split()
            word = parts[0]
            embedding = np.array([float(val) for val in parts[1:]])
            glove_embeddings[word] = embedding
    # Вычисление векторного представления для каждой n-граммы
    ngram_vectors = []
    for ngram in bigrams:
        ngram_vector = np.mean([glove_embeddings[word] for word in ngram if word in glove_embeddings], axis=0)
        ngram_vectors.append(ngram_vector)

    # Вычисление матрицы семантической схожести
    similarity_matrix = cosine_similarity(ngram_vectors)

    return similarity_matrix


def morphological_analysis(sentence, block_size=100000):
    # Загрузка модели SpaCy для английского языка
    nlp = spacy.load("en_core_web_sm")

    # Разбиение предложения на блоки по заданному размеру
    words = sentence.split()
    word_blocks = [words[i:i + block_size] for i in range(0, len(words), block_size)]

    # Список для хранения результатов анализа
    analysis_results = []

    # Обработка каждого блока
    for block in word_blocks:
        # Склеиваем слова обратно в предложение
        block_sentence = " ".join(block)
        # Анализ предложения с помощью SpaCy
        doc = nlp(block_sentence)

        # Получение морфологической информации о каждом токене
        for token in doc:
            analysis_results.append({
                "token_text": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "tag": token.tag_,
                "dependency": token.dep_,
                "shape": token.shape_,
                "is_alpha": token.is_alpha,
                "is_stop": token.is_stop
            })

    return analysis_results


def save_results_to_json(data, file_path):
    if isinstance(data, np.ndarray):
        # Для массива numpy
        with open(file_path, 'w') as file:
            json.dump(data.tolist(), file)
    elif isinstance(data, list) and len(data) != 0 and isinstance(data[0], tuple):
        # Для данных в формате списка кортежей
        json_data = [{'source': pair[0], 'target': pair[1]} for pair in data]
        with open(file_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)
    elif isinstance(data, zip):
        # Для данных в формате zip
        data_list = list(data)
        json_data = [{'source': pair[0], 'target': pair[1]} for pair in data_list]
        with open(file_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)
    elif isinstance(data, str):
        # Для данных в формате строки
        with open(file_path, 'w') as file:
            file.write(data)
    elif isinstance(data, list) and not data:
        # Для пустого списка
        with open(file_path, 'w') as file:
            file.write('[]')
    else:
        # Для других данных (например, sentence)
        with open(file_path, 'w') as file:
            json.dump(data, file)


def save_results_to_txt(data, file_path):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(data)
        # print("Предложение успешно сохранено в файл по пути:", file_path)
    except Exception as e:
        print("Возникла ошибка при сохранении предложения в файл:", str(e))


def save_ngrams_to_json(ngrams, file_path):
    try:
        with open(file_path, 'w') as file:
            json.dump(ngrams, file)
        # print(f"ngrams сохранены в файл JSON по пути: {file_path}")
    except Exception as e:
        print(f"Ошибка при сохранении ngrams в файл JSON: {e}")


# ==================================================================
#      ПОДГОТОВКА ТЕКСТА
# ==================================================================
sentence = get_data(word_count=0)  # word_count=0 для всего файла, 1000, 50, 1000 и т.п. для части слов
save_results_to_txt(sentence, f"../../data/{'Bible_NIV_prepared_data'}.txt")

# ==================================================================
#      НАИБОЛЕЕ ЧАСТО ВСТРЕЧАЕМЫЕ СЛОВА
# ==================================================================
excluded_words = ["i", "to", "be", "one", "get", "n", "mr", "u"]
top_n = 50
print(f"Наиболее встречаемые {top_n} слов")
result_top_words = top_words(sentence, excluded_words, min_word_length=4, top_n=top_n)
print(result_top_words)
save_results_to_json(result_top_words, f"../../data/{'Bible_NIV_top_words'}.json")

# ==================================================================
#      ИЗВЛЕЧЕНИЕ N-ГРАММ
# ==================================================================
n = 2
print(f'Извлечение {n}-grams')
bigrams = n_grams(sentence, n)
size = sys.getsizeof(bigrams)
print(f"Размер объекта zip: {size} байт")
save_ngrams_to_json(list(list(bigrams)), f"../../data/{'Bible_NIV_ngrams'}.json")
size = sys.getsizeof(bigrams)
print(f"Размер объекта zip: {size} байт")

# ==================================================================
#      1. ЧАСТОТНЫЙ АНАЛИЗ
# ==================================================================
n = 3
print(f"Наииболее часто встречающиеся {n}-граммы в тексте")
ngrams = n_grams(sentence, n)
top_n = 10
most_common_bigrams = most_common_ngrams(ngrams, top_n)
print(most_common_bigrams)
save_results_to_json(most_common_bigrams, f"../../data/{'Bible_NIV_frequency_analysis'}.json")

# ==================================================================
#      2. СИНТАКСИЧЕСКИЙ АНАЛИЗ
# ==================================================================
print("Результаты синтаксического анализа:")
analysis_results = syntactic_analysis(sentence)
for result_syntactic in analysis_results:
    print(result_syntactic)
save_results_to_json(analysis_results, f"../../data/{'Bible_NIV_parsing'}.json")

# ==================================================================
#      3. КОНТЕКСТНЫЙ АНАЛИЗ
# ==================================================================
print("Контекст для каждого слова из наиболее встречаемых в н-граммах:")
context_analysis = analyze_context(sentence.split(), result_top_words)
for word, bigrams in context_analysis.items():
    print(f"Слово: {word}")
    for context in bigrams:
        print(f"Контекст: {' '.join(context)}")
    print()
save_results_to_json(context_analysis, f"../../data/{'Bible_NIV_context_analysis'}.json")

# ==================================================================
#      4. СЕМАНТИЧЕСКИЙ АНАЛИЗ
# ==================================================================
print("Результаты семантического анализа n-грамм:")
similarity_matrix = semantic_analysis_ngrams(bigrams)
print(similarity_matrix)
save_results_to_json(similarity_matrix, f"../../data/{'Bible_NIV_semantic_analysis'}.json")

# ==================================================================
#      5. МОРФОЛОГИЧЕСКИЙ АНАЛИЗ
# ==================================================================
print("Результаты морфологического анализа:")
analysis_results = morphological_analysis(sentence)
# Вывод результатов анализа
for result in analysis_results:
    print(result)
save_results_to_json(analysis_results, f"../../data/{'Bible_NIV_morphological_analysis'}.json")








