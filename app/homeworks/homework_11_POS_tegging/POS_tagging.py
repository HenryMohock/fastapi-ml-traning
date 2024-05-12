import nltk
import re
import nltk
from nltk.tag import BrillTaggerTrainer
from nltk.tag import UnigramTagger
from app.homeworks.homework_10_N_gramm_models.Preparing_text import Preliminary_preparation_data
from nltk.tag import CRFTagger
import matplotlib.pyplot as plt
import time
from sklearn.metrics import classification_report
import pandas as pd


def format_time(seconds):
    """
    Форматирование секунд к виду ЧЧ:ММ:СС

      Возвращает строку отформатированного времени
    """
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return "{:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds))


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


def pos_tagging_sentences(sentences):
    # Создание пустого списка для объединенных размеченных предложений
    pos_tagged_sentences = []
    # Применение POS-теггера к каждому предложению
    for sentence in sentences:
        # Токенизация предложения
        tokens = nltk.word_tokenize(sentence)
        # Применение POS-теггера
        tagged_sentence = nltk.pos_tag(tokens)
        # Добавление размеченного предложения к объединенному списку
        pos_tagged_sentences.extend([tagged_sentence])
        # Вывод результатов
    return pos_tagged_sentences


def apply_regex_tokenize(text):
    """
    Обработка текста регулярными выражениями.

    согласно переданному регулярному выражению изменяется текст
    """
    text = re.sub(r'(\d)([^\d\s])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text)  # замена нескольких пробелов одним
    #text = re.sub(r'([A-Za-z]\.\" )|([A-Za-z]\. )', r'\1\2\n', text)  # каждое новое предложение с новой строки
    text = re.sub(r'\d+', '', text)  # числа
    #text = re.sub(r'\s*\n\s*', '\n', text)  # пробелы
    text = re.sub(r'\s*GENESIS\s*', '', text)
    text = re.sub(r'"', '', text)  # двойные кавычки
    text = re.sub(r'-', '', text)  # тире
    text = re.sub(r'—', '', text)  # длинное тире
    text = re.sub(r"['“”]", '', text)  # двойные кавычки
    text = re.sub(r"['‘’]", '', text)  # одинарные кавычки
    #text = re.sub(r'[^\w\s]', '', text)  # все знаки препинания
    text = re.sub(r'[^\S\n]+', ' ', text)  # пробелы
    text = nltk.sent_tokenize(text)
    # text = [item.lower() for item in text]
    return text


def plot_distribution_parts_speech(tagged_sentences):
    # Создание словаря для подсчета частей речи
    pos_counts = {}
    for sentence in tagged_sentences:
        for word, pos in sentence:
            if pos in pos_counts:
                pos_counts[pos] += 1
            else:
                pos_counts[pos] = 1

    # Выделение частей речи с долей менее 2% и суммирование их в одну категорию "Другие"
    total_count = sum(pos_counts.values())
    for pos, count in list(pos_counts.items()):
        if count / total_count < 0.02:
            pos_counts['Other'] = pos_counts.get('Другие', 0) + count
            del pos_counts[pos]

    # Построение круговой диаграммы
    plt.figure(figsize=(10, 6))
    plt.pie(pos_counts.values(), labels=pos_counts.keys(), autopct='%1.1f%%', startangle=140)
    plt.title('Распределение частей речи в тексте')
    plt.axis('equal')
    plt.show()


def plot_model_accuracy_histogram(accuracy_models):
    # Извлечение названий моделей и их точности
    model_names = [model[0] for model in accuracy_models]
    accuracies = [model[1] for model in accuracy_models]

    # Построение гистограммы
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, accuracies, color='skyblue')
    plt.title('Точность различных моделей')
    plt.xlabel('Модели')
    plt.ylabel('Точность')
    plt.ylim(0.75, 0.95)  # Установка предела оси y от 0 до 1 (точность в диапазоне от 0 до 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Добавление сетки на оси y
    plt.xticks(rotation=45)  # Поворот подписей оси x на 45 градусов для лучшей читаемости
    plt.tight_layout()  # Автоматическое наложение подписей
    plt.show()


def plot_accuracy_in_time(train_sets, test_sets, templates):
    # Списки для хранения точности каждой модели
    brill_accuracies = []
    mapped_accuracies = []
    crf_accuracies = []
    hmm_accuracies = []

    # Обучение моделей и запись точности на каждом этапе
    start_time = time.time()
    for i in range(10, train_size, 10):  # Измените размер шага по вашему усмотрению
        # Обучение модели BRILL
        initial_tagger = UnigramTagger(train_sets[:i])
        trainer = BrillTaggerTrainer(initial_tagger=initial_tagger, templates=templates)
        brill_tagger = trainer.train(train_sents=train_sets[:i], max_rules=10)
        brill_accuracy = brill_tagger.accuracy(test_sets)
        brill_accuracies.append(brill_accuracy)

        # Обучение модели BRILL с маппингом
        initial_tagger = UnigramTagger(train_sets[:i])
        trainer = BrillTaggerTrainer(initial_tagger=initial_tagger, templates=templates)
        brill_tagger = trainer.train(train_sents=train_sets[:i], max_rules=10)
        mapped_accuracy = brill_tagger.accuracy(test_sets)
        mapped_accuracies.append(mapped_accuracy)

        # Обучение модели CRF
        crf_tagger = CRFTagger()
        crf_tagger.train(train_sets[:i], 'model.crf.tagger')
        crf_accuracy = crf_tagger.accuracy(test_sets)
        crf_accuracies.append(crf_accuracy)

        # Обучение модели HMM
        hmm_tagger = nltk.HiddenMarkovModelTagger.train(train_sets[:i])
        hmm_accuracy = hmm_tagger.accuracy(test_sets)
        hmm_accuracies.append(hmm_accuracy)

    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(range(10, train_size, 10), brill_accuracies, label='BRILL', marker='o', markersize=2)
    plt.plot(range(10, train_size, 10), mapped_accuracies, label='BRILL + MAPPING', marker='o', markersize=2)
    plt.plot(range(10, train_size, 10), crf_accuracies, label='CRF', marker='o', markersize=2)
    plt.plot(range(10, train_size, 10), hmm_accuracies, label='HMM', marker='o', markersize=2)

    plt.title('Точность моделей во времени')
    plt.xlabel('Объем обучающего набора данных')
    plt.ylabel('Точность')
    plt.legend()
    plt.grid(True)
    print("- Час підготовки графіку Точності моделей у часі:", format_time(time.time() - start_time))
    plt.show()


def model_brill(train_sets):
    # Создаем шаблоны для модели Brill
    templates = nltk.tag.brill.brill24()

    # Тренировка UnigramTagger
    initial_tagger = UnigramTagger(train_sets)

    # Обучение модели Brill
    trainer = BrillTaggerTrainer(initial_tagger=initial_tagger, templates=templates)
    brill_tagger = trainer.train(train_sents=train_sets, max_rules=10)

    return brill_tagger, templates


def model_brill_mapping(train_sets):
    # Создаем шаблоны для модели Brill
    templates = nltk.tag.brill.brill24()

    # Обучаем модель Brill
    initial_tagger = UnigramTagger(train_sets)
    trainer = BrillTaggerTrainer(initial_tagger=initial_tagger, templates=templates)
    brill_tagger = trainer.train(train_sents=train_sets, max_rules=10)

    # Создание маппинга
    tag_mapping = {'NN': 'NOUN', 'VB': 'VERB', 'JJ': 'ADJ', 'RB': 'ADV', 'PRP': 'PRON', 'DT': 'DET'}

    # Преобразование разметки данных с использованием маппинга
    mapped_train_sets = [[(word, tag_mapping.get(tag, tag)) for word, tag in sent] for sent in train_sets]
    mapped_test_sets = [[(word, tag_mapping.get(tag, tag)) for word, tag in sent] for sent in test_sets]

    # Обучение модели Brill с маппингом
    mapped_trainer = BrillTaggerTrainer(initial_tagger=initial_tagger, templates=templates)
    mapped_brill_tagger = mapped_trainer.train(train_sents=mapped_train_sets, max_rules=10)

    return mapped_brill_tagger, mapped_test_sets


def model_cfr(train_sets):
    # Создание объекта модели
    crf_tagger = CRFTagger()
    # Обучение модели
    crf_tagger.train(train_sets, 'model.crf.tagger')
    crf_test_sets = crf_tagger.tag_sents(sentences)

    return crf_tagger


def model_hmm(train_sets):
    # Обучение модели HMM
    hmm_tagger = nltk.HiddenMarkovModelTagger.train(train_sets)
    return hmm_tagger


def model_accuracy(model_tagger, accuracy_models, name_model, test_sets):
    # Оценка точности модели
    if name_model == 'Brill' or name_model == 'CRF' or name_model == 'HMM':
        accuracy = model_tagger.accuracy(test_sets)
        print(f"{name_model} Accuracy:", accuracy)
        accuracy_models.append([f'{name_model} model', accuracy])
    if name_model == 'Brill mapping':
        accuracy = model_tagger.accuracy(test_sets)
        print(f"{name_model} Accuracy:", accuracy)
        accuracy_models.append([f'{name_model} model', accuracy])


def classification_report_model_simple(test_sets, model_tagger, name_model):
    # Построение классификационного отчета
    correct_labels = [tag for sentences in test_sets for word, tag in sentences]
    predicted_labels = []
    for sentences in test_sets:
        predicted_labels += [tag for _, tag in model_tagger.tag([word for word, _ in sentences])]

    print()
    print(f'{name_model} correct labels (len {len(correct_labels)}): {correct_labels}')
    print(f'{name_model} predicted labels (len {len(predicted_labels)}): {predicted_labels}')
    print()
    print(f'Классификационный отчет по {name_model}:')
    # Инициализация счетчиков
    total_matches = 0
    total_mismatches = 0

    # Цикл сравнения данных и подсчет совпадений и несовпадений
    for correct_label, predicted_label in zip(correct_labels, predicted_labels):
        if correct_label == predicted_label:
            total_matches += 1
        else:
            total_mismatches += 1

    # Подсчет процентного соотношения
    total_items = len(correct_labels)
    match_percentage = (total_matches / total_items) * 100
    mismatch_percentage = (total_mismatches / total_items) * 100

    # Вывод результатов
    print(f"Количество совпадений: {total_matches} ({match_percentage:.2f}% от общего числа)")
    print(f"Количество несовпадений: {total_mismatches} ({mismatch_percentage:.2f}% от общего числа)")


def classification_report_model(test_sets, model_tagger, name_model):
    # Построение классификационного отчета
    correct_labels = [tag for sentences in test_sets for word, tag in sentences]
    predicted_labels = []
    for sentences in test_sets:
        predicted_labels += [tag for _, tag in model_tagger.tag([word for word, _ in sentences])]

    print()
    print(f'{name_model} correct labels (len {len(correct_labels)}): {correct_labels}')
    print(f'{name_model} predicted labels (len {len(predicted_labels)}): {predicted_labels}')
    print()
    print(f'Классификационный отчет по {name_model}:')
    # Создаем DataFrame для хранения результатов
    results_df = pd.DataFrame(columns=['correct', 'predict', 'precision', 'recall', 'f1-score', 'support'])

    # Рассчитываем метрики для каждой пары меток
    for correct, predict in zip(correct_labels, predicted_labels):
        precision = 1 if correct == predict else 0
        recall = 1 if correct == predict else 0
        f1_score = 1 if correct == predict else 0
        support = 1

        # Добавляем результаты в DataFrame
        results_df = results_df._append({'correct': correct,
                                        'predict': predict,
                                        'precision': precision,
                                        'recall': recall,
                                        'f1-score': f1_score,
                                        'support': support}, ignore_index=True)

    # Вычисляем итоги
    accuracy = results_df['f1-score'].mean()
    macro_avg = results_df.agg({'precision': 'mean', 'recall': 'mean', 'f1-score': 'mean', 'support': 'sum'})
    weighted_avg = results_df.agg({'precision': 'mean', 'recall': 'mean', 'f1-score': 'mean', 'support': 'sum'})


    # Добавляем итоги в DataFrame
    results_df = results_df._append({'correct': 'Accuracy',
                                    'predict': accuracy,
                                    'precision': macro_avg['precision'],
                                    'recall': macro_avg['recall'],
                                    'f1-score': macro_avg['f1-score'],
                                    'support': weighted_avg['support']}, ignore_index=True)

    results_df = results_df._append({'correct': 'Macro avg',
                                    'predict': '',
                                    'precision': macro_avg['precision'],
                                    'recall': macro_avg['recall'],
                                    'f1-score': macro_avg['f1-score'],
                                    'support': weighted_avg['support']}, ignore_index=True)

    results_df = results_df._append({'correct': 'Weighted avg',
                                    'predict': '',
                                    'precision': weighted_avg['precision'],
                                    'recall': weighted_avg['recall'],
                                    'f1-score': weighted_avg['f1-score'],
                                    'support': weighted_avg['support']}, ignore_index=True)

    print(results_df)


text, roh_text = get_data(word_count=0)

# Обработка регулярными выражениями и разделение текста на отдельные предложения
sentences = apply_regex_tokenize(roh_text)

# Применение POS-теггера к каждому предложению
tagged_sentences = pos_tagging_sentences(sentences)

# Разделение на обучающий и тестовый наборы данных
train_size = int(0.8 * len(tagged_sentences))
train_sets = tagged_sentences[:train_size]
test_sets = tagged_sentences[train_size:]
print(f'Кількість речень: {len(tagged_sentences)}')
print(f'Кількість тренувальних речень: {train_size}')
print(f'Кількість тестувальних речень: {len(tagged_sentences) - train_size}')
print()

accuracy_models = []
model_training_time = []

# ============= BRILL ============================

name_model = 'Brill'
start_time = time.time()
brill_tagger, templates = model_brill(train_sets)
model_accuracy(brill_tagger, accuracy_models, 'Brill', test_sets)
model_training_time.append(f'{name_model}: {format_time(time.time() - start_time)}')

# Построение классификационного отчета
classification_report_model(test_sets=test_sets,model_tagger=brill_tagger,name_model=name_model)

# ============ BRILL + MAPPING ==========================

name_model = 'Brill mapping'
start_time = time.time()
mapped_brill_tagger, mapped_test_sets = model_brill_mapping(train_sets)
model_accuracy(mapped_brill_tagger, accuracy_models, 'Brill mapping', mapped_test_sets)
model_training_time.append(f'{name_model}: {format_time(time.time() - start_time)}')

# Построение классификационного отчета
classification_report_model(test_sets=mapped_test_sets,model_tagger=mapped_brill_tagger,name_model=name_model)

# =============== CRF ===================

name_model = 'CRF'
start_time = time.time()
crf_tagger = model_cfr(train_sets)
model_accuracy(crf_tagger, accuracy_models, 'CRF', test_sets)
model_training_time.append(f'{name_model}: {format_time(time.time() - start_time)}')

# Построение классификационного отчета
correct_labels = [tag for sentences in test_sets for word, tag in sentences]
predicted_labels = []
for sentences in test_sets:
    predicted_labels += [tag for _, tag in crf_tagger.tag([word for word, _ in sentences])]

print()
print(f'{name_model} correct labels (len {len(correct_labels)}): {correct_labels}')
print(f'{name_model} predicted labels (len {len(predicted_labels)}): {predicted_labels}')
print()
print(f'Классификационный отчет по {name_model}')
print(classification_report(y_true=correct_labels, y_pred=predicted_labels, zero_division=0.0))
print()

# ============ H M M ==========================

name_model = 'HMM'
start_time = time.time()
hmm_tagger = model_hmm(train_sets)
model_accuracy(crf_tagger, accuracy_models, 'HMM', test_sets)
model_training_time.append(f'{name_model}: {format_time(time.time() - start_time)}')

# Построение классификационного отчета
correct_labels = [tag for sentences in test_sets for word, tag in sentences]
predicted_labels = []
for sentences in test_sets:
    predicted_labels += [tag for _, tag in hmm_tagger.tag([word for word, _ in sentences])]

print()
print(f'{name_model} correct labels (len {len(correct_labels)}): {correct_labels}')
print(f'{name_model} predicted labels (len {len(predicted_labels)}): {predicted_labels}')
print()
print(f'Классификационный отчет по {name_model}')
print(classification_report(y_true=correct_labels, y_pred=predicted_labels, zero_division=0.0))
print()

# ================= Время тренировки моделей =============
print()
print('Время тренировки моделей:')
print(model_training_time)

# ================= Точности всех моделей ================
print()
print('Accuracy models:')
print(accuracy_models)

# ========== ГРАФИКИ =======================

# 1. Распределение частей речи в тексте:
plot_distribution_parts_speech(tagged_sentences)

# 2. Гистограмма точностей моделей
plot_model_accuracy_histogram(accuracy_models)

# 3. Точность во времени
plot_accuracy_in_time(train_sets, test_sets, templates)
