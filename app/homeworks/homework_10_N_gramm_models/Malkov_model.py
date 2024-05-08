from app.homeworks.homework_10_N_gramm_models.Preparing_text import Preliminary_preparation_data
from collections import Counter
from collections import defaultdict
import json
import time
from nltk.util import ngrams
import random
import asyncio
import TranslatorX


def get_data(word_count: int = 0):
    # Параметры текста
    encoding = 'utf8'
    lang = 'english'

    # Путь к файлу CSV
    file_path = "../../data/Bible_NIV.txt"

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


def format_time(seconds):
    """
    Форматирование секунд к виду ЧЧ:ММ:СС

      Возвращает строку отформатированного времени
    """
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return "{:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds))


def create_transition_matrix_model(sentence):
    # Разбиваем предложение на слова
    words = sentence.split()

    # Создаем словарь для хранения счетчиков переходов
    transition_matrix = defaultdict(lambda: defaultdict(float))
    # Создаем модель Маркова
    markov_model = defaultdict(lambda: defaultdict(int))

    # Проходим по корпусу текста и считаем переходы
    for i in range(len(words) - 1):
        current_word, next_word = words[i], words[i + 1]
        markov_model[current_word][next_word] += 1

    # Нормализуем счетчики, чтобы получить вероятности
    for current_word, next_words in markov_model.items():
        total_transitions = sum(next_words.values())
        for next_word, count in next_words.items():
            transition_matrix[current_word][next_word] = count / total_transitions

    return transition_matrix, markov_model


def generate_text(start_word, transition_matrix, length=10):
    current_word = start_word
    generated_text = [current_word]

    for _ in range(length - 1):
        if current_word in transition_matrix:
            next_word = random.choices(
                list(transition_matrix[current_word].keys()),
                weights=list(transition_matrix[current_word].values())
            )[0]
        else:
            # Если для текущего слова нет доступных суффиксов,
            # выбираем случайное слово из всех доступных префиксов
            next_word = random.choice(list(transition_matrix.keys()))
        generated_text.append(next_word)
        current_word = next_word

    return " ".join(generated_text)


def train_markov_model_and_generate_sentence(sentence, length=10):
    # Разбиваем предложение на слова
    words = sentence.split()

    # Создаем модель Маркова
    markov_model = defaultdict(lambda: defaultdict(int))

    # Проходим по предложению и считаем переходы
    for i in range(len(words) - 1):
        current_word, next_word = words[i], words[i + 1]
        markov_model[current_word][next_word] += 1

    # Создаем пустой список для сгенерированного предложения
    generated_sentence = []

    # Генерируем предложение из 10 слов
    for _ in range(length):
        # Если текущее слово не встречается в модели Маркова, выбираем случайное слово из предложения
        if not generated_sentence:
            current_word = random.choice(words)
        else:
            # Иначе выбираем следующее слово на основе вероятностей из модели Маркова
            current_word = random.choices(
                list(markov_model[current_word].keys()),
                weights=list(markov_model[current_word].values())
            )[0]

        # Добавляем слово в сгенерированное предложение
        generated_sentence.append(current_word)

    # Собираем предложение из списка слов
    generated_sentence = ' '.join(generated_sentence)

    return generated_sentence


async def translate_to_ukrainian_asinc(text):
    trans = TranslatorX.Translator()
    trans_text = trans.Translate(text=text, to_lang='uk')
    return trans_text


def translate_to_ukrainian(text):
    return asyncio.run(translate_to_ukrainian_asinc(text))


print('Підготовка корпусів тексту')
print('===================================================================')
start_time = time.time()
sentence, corpus = get_data(word_count=0)  # word_count=0 для всего файла, 1000, 50, 1000 и т.п. для части слов
save_results_to_txt(sentence, f"../../data/{'Bible_NIV_prepared_data'}.txt")
print(f"Оброблений текст: {sentence}")
print(f"Не оброблений текст: {corpus}")
print("- Час підготовки тексту:", format_time(time.time() - start_time))


print()
print('Генерація n-граам')
print('===================================================================')
start_time = time.time()
n = 3
n_grams = list(n_grams(sentence=sentence, n=n))
# Выводим результат
print(n_grams)
print("- Час підготовки n-граам:", format_time(time.time() - start_time))


print()
print('Побудова матриці переходу та моделі Маркова')
print('===================================================================')
start_time = time.time()
transition_matrix, markov_model = create_transition_matrix_model(sentence)
for current_word, next_words in transition_matrix.items():
    print(f'Word: {current_word}')
    for next_word, probability in next_words.items():
        print(f'\tNext word: {next_word}, Probability: {probability}')
print("- Час побудови Матриці переходів:", format_time(time.time() - start_time))


print()
print('Генерація речень зі словом "god"')
print('===================================================================')
start_time = time.time()
start_word = 'god'
number_sentence = 10
for nn in range(number_sentence):
    new_text = generate_text(start_word=start_word, transition_matrix=transition_matrix, length=10)
    print(f"Нове речення № {nn+1}: {translate_to_ukrainian(new_text)}")
print("- Час генерація речень зі словом 'god'':", format_time(time.time() - start_time))


print()
print('Генерація кількох нових речень із 10 слів')
print('===================================================================')
start_time = time.time()
number_sentence = 10
for nn in range(number_sentence):
    new_text = train_markov_model_and_generate_sentence(sentence)
    print(f"Нове речення № {nn+1}: {translate_to_ukrainian(new_text)}")
print("- Час генерація кількох нових речень із 10 слів:", format_time(time.time() - start_time))


print()
print('Генерація кількох нових речень починаючи з 10 популярних слів')
print('===================================================================')
start_time = time.time()
top_words_list = top_words(sentence=sentence, top_n=5)
number_sentence = 5
print('Найбільш популярні слова^')
print(top_words_list)
for start_word, count in top_words_list:
    print(f'Речення зі слова {start_word}:')
    for nn in range(number_sentence):
        new_text = generate_text(start_word=start_word, transition_matrix=transition_matrix, length=10)
        print(f"Нове речення № {nn + 1}: {translate_to_ukrainian(new_text)}")
print("- Час генерації кількох нових речень починаючи з 10 популярних слів:", format_time(time.time() - start_time))


