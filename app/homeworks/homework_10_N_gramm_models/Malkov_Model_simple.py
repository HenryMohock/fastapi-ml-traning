import nltk
from nltk.util import ngrams
from app.homeworks.homework_10_N_gramm_models.Preparing_text import Preliminary_preparation_data
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


def n_grams(sentence, n=2):
    grams = ngrams(sentence.split(), n)
    return grams


# Создание матрицы переходов по n-граммам
def create_transition_matrix(n_grams):
    transitions = {}
    for n_gram in n_grams:
        prefix = tuple(n_gram[:-1])
        suffix = n_gram[-1]
        if prefix in transitions:
            transitions[prefix].append(suffix)
        else:
            transitions[prefix] = [suffix]
    return transitions


async def translate_to_ukrainian_asinc(text):
    trans = TranslatorX.Translator()
    trans_text = trans.Translate(text=text, to_lang='uk')
    return trans_text


def translate_to_ukrainian(text):
    return asyncio.run(translate_to_ukrainian_asinc(text))


# Обучение модели Маркова
def generate_sentence(transition_matrix, n, length=15):
    current = random.choice(list(transition_matrix.keys()))
    sentence = list(current)

    while len(sentence) < length:
        if current in transition_matrix:
            next_word = random.choice(transition_matrix[current])
            sentence.append(next_word)
            current = tuple(sentence[-n:])
        else:
            break

    return ' '.join(sentence)


# Предложение для тестирования
sentence, corpus = get_data(word_count=0)

# Создание n-грамм текста
n = 10  # Размер n-грамм равен размеру предложения
n_grams = n_grams(sentence=sentence, n=n)
transition_matrix = create_transition_matrix(n_grams)
# Создание 10 предложений из 15 слов
print()
print('Створення 10 речень із 15 слів:')
for _ in range(10):
    generated_sentence = generate_sentence(transition_matrix=transition_matrix, n=n)
    translate_generated_sentence = translate_to_ukrainian(generated_sentence)
    print(generated_sentence, ' (', translate_generated_sentence, ')')
