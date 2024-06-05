import spacy
from spacy.tokens import DocBin
import pandas as pd
import random
import signal


# Регистрация обработчика сигнала прерывания
signal.signal(signal.SIGINT, signal.default_int_handler)

#  Подготовка данных и разметка новых сущностей:
#  =============================================

# Параметры текста
encoding = 'utf8'
lang = 'english'
len_text = 65521550  # 65521550  # 200000

# Путь к файлу CSV
file_path = "../../data/IMDB Dataset.csv"

# Имя извлекаемой колонки файла CSV
column_name = 'review'

# Загрузка данных из файла CSV
data = pd.read_csv(file_path, encoding=encoding)

# Извлечение колонки column_name и объединение ее в текстовую переменную (внутреннее свойство)
text = ' '.join(data[column_name].astype(str))[:len_text]

# Создание подвыборки для обучения
subset_size = 100000  # Используйте меньшее количество данных для примера
text_subset = text[:subset_size]

# Загрузка модели SpaCy
nlp = spacy.load("en_core_web_sm")

# Пример разметки данных вручную
TRAIN_DATA = [
    # ORG (Organization) - организация, компания, учреждение
    ("Apple is looking at buying U.K. startup for $1 billion", {"entities": [(0, 5, "ORG")]}),
    # GPE (Geo-Political Entity) - геополитическая сущность, страна/город/регион
    ("San Francisco considers banning sidewalk delivery robots", {"entities": [(0, 13, "GPE")]}),
    # NORP (Nationalities or Religious or Political Groups) - национальность, религиозная или политическая группа
    ("The Republican Party is one of the two major contemporary political parties", {"entities": [(4, 14, "NORP")]}),
    # CARDINAL - числительное
    ("The party was founded in 1854 by anti-slavery activists", {"entities": [(25, 30, "CARDINAL")]}),
    ("About 3 people are missing", {"entities": [(6, 7, "CARDINAL")]}),
    # PERSON (Person) - персона, человек
    ("John is having visions that cause him to black out", {"entities": [(0, 4, "PERSON")]}),
    # DATE (Date) - дата
    ("This happened on August 30, 1999", {"entities": [(17, 31, "DATE")]}),
    ("This happened on 12-25-1992", {"entities": [(17, 26, "DATE")]}),
    ("This happened on 2023-05-15", {"entities": [(17, 26, "DATE")]}),
    # TIME (Time) - время
    ("This happened at 10:00 AM", {"entities": [(17, 24, "TIME")]}),
    ("This happened at 14:30", {"entities": [(17, 21, "TIME")]}),
    # MONEY (Money) - деньги, валюта
    ("These pants cost $10", {"entities": [(17, 17, "MONEY")]}),
    ("This phone costs €200", {"entities": [(17, 17, "MONEY")]}),
    # PERCENT (Percentage) - процент
    ("Our profits increased by 50% or more", {"entities": [(27, 28, "PERCENT")]}),
    # LOC (Location) - местоположение
    ("Today we walked in the park among the trees", {"entities": [(23, 27, "LOC")]}),
    # PRODUCT (Product) - продукт, товар
    ("I bought an iPhone today with a great camera", {"entities": [(12, 18, "PRODUCT")]}),
    # EVENT (Event) - событие
    ("Chernomorets Odesa won the FIFA World Cup", {"entities": [(32, 40, "EVENT")]}),
]

# Создание DocBin для обучения
db = DocBin()
for text, annot in TRAIN_DATA:
    doc = nlp.make_doc(text)
    ents = []
    for start, end, label in annot["entities"]:
        span = doc.char_span(start, end, label=label)
        if span is None:
            print(f"Skipping entity [{start}, {end}, {label}] in {text}")
        else:
            ents.append(span)
    doc.ents = ents
    db.add(doc)

# Сохранение размеченных данных
db.to_disk("./train.spacy")


#  Обучение NER-модели:
#  ====================

import spacy
from spacy.training import Example
from spacy.tokens import DocBin

# Загрузка размеченных данных
db = DocBin().from_disk("./train.spacy")
docs = list(db.get_docs(nlp.vocab))

# Создание примеров для обучения
examples = [Example.from_dict(doc, {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]}) for doc in docs]

# Создание новой модели
nlp_blank = spacy.blank("en")
ner = nlp_blank.create_pipe("ner")
nlp_blank.add_pipe("ner", last=True)

# Добавление меток
for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Обучение модели
optimizer = nlp_blank.begin_training()
for i in range(10):
    random.shuffle(examples)
    for example in examples:
        nlp_blank.update([example], sgd=optimizer)

# Сохранение модели
nlp_blank.to_disk("./ner_model")
