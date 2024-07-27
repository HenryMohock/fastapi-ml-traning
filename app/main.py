from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from textdistance import (hamming, mlipns, levenshtein, damerau_levenshtein, jaro_winkler, strcmp95, needleman_wunsch,
                          gotoh, smith_waterman,
                          jaccard, sorensen, tversky, overlap, tanimoto, cosine,monge_elkan, bag,
                          lcsseq, lcsstr, ratcliff_obershelp,
                          arith_ncd, rle_ncd, bwtrle_ncd,
                          sqrt_ncd, entropy_ncd,
                          bz2_ncd, lzma_ncd, zlib_ncd,
                          mra, editex,
                          prefix, postfix, length, identity, matrix)

from app.api.api import api_router
from app.api.heartbeat import heartbeat_router
from app.core.config import settings
from app.core.event_handler import start_app_handler, stop_app_handler
from fastapi.staticfiles import StaticFiles  # для публікації index.html

from app.homeworks.homework_09_classifier.Classifier import NLTKClassifier, CsvPreProcessing
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from pydantic import BaseModel
from gensim.models.doc2vec import Doc2Vec
from sklearn.cluster import KMeans
import os
from sklearn.metrics import silhouette_score
from collections import Counter
from typing import List, Dict, Tuple
from nltk.corpus import stopwords

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import spacy

# Імпорт необхідних бібліотек для моделі NLLB
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

import speech_recognition as sr
from translate import Translator


app = FastAPI(title=settings.PROJECT_NAME)

app.include_router(heartbeat_router)
app.include_router(api_router, prefix=settings.API_V1_STR, tags=["ML API"])

app.add_event_handler("startup", start_app_handler(app, settings.MODEL_PATH))
app.add_event_handler("shutdown", stop_app_handler(app))

# =========================================================
# Монтування статичних файлів (наприклад, HTML, CSS, JS)
# Додано, щоб опублікувати на localhost сторінки index.html
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}
# ==========================================================


# POST REQUEST 1 (calculate_similarity)
# ==========================================================
# ==========================================================

def calculate_method(method, line1, line2):
    # Edit based
    if method == 'hamming':
        similarity = hamming.normalized_similarity(line1, line2)
        return similarity
    elif method == 'mlipns':
        similarity = mlipns.normalized_similarity(line1, line2)
        return similarity
    elif method == 'levenshtein':
        similarity = levenshtein.normalized_similarity(line1, line2)
        return similarity
    elif method == 'damerau_levenshtein':
        similarity = damerau_levenshtein.normalized_similarity(line1, line2)
        return similarity
    elif method == 'jaro_winkler':
        similarity = jaro_winkler.normalized_similarity(line1, line2)
        return similarity
    elif method == 'strcmp95':
        similarity = strcmp95.normalized_similarity(line1, line2)
        return similarity
    elif method == 'needleman_wunsch':
        similarity = needleman_wunsch.normalized_similarity(line1, line2)
        return similarity
    elif method == 'gotoh':
        similarity = gotoh.normalized_similarity(line1, line2)
        return similarity
    elif method == 'smith_waterman':
        similarity = smith_waterman.normalized_similarity(line1, line2)
        return similarity
    # Token based
    elif method == 'jaccard':
        similarity = jaccard.normalized_similarity(line1, line2)
        return similarity
    elif method == 'sorensen':
        similarity = sorensen.normalized_similarity(line1, line2)
        return similarity
    elif method == 'tversky':
        similarity = tversky.normalized_similarity(line1, line2)
        return similarity
    elif method == 'overlap':
        similarity = overlap.normalized_similarity(line1, line2)
        return similarity
    elif method == 'tanimoto':
        similarity = tanimoto.normalized_similarity(line1, line2)
        return similarity
    elif method == 'cosine':
        similarity = cosine.normalized_similarity(line1, line2)
        return similarity
    elif method == 'tversky':
        similarity = tversky.normalized_similarity(line1, line2)
        return similarity
    elif method == 'monge_elkan':
        similarity = monge_elkan.normalized_similarity(line1, line2)
        return similarity
    elif method == 'bag':
        similarity = bag.normalized_similarity(line1, line2)
        return similarity
    # Sequence based
    elif method == 'lcsseq':
        similarity = lcsseq.normalized_similarity(line1, line2)
        return similarity
    elif method == 'lcsstr':
        similarity = lcsstr.normalized_similarity(line1, line2)
        return similarity
    elif method == 'ratcliff_obershelp':
        similarity = ratcliff_obershelp.normalized_similarity(line1, line2)
        return similarity
    # Classic compression algorithms
    elif method == 'arith_ncd':
        similarity = arith_ncd.normalized_similarity(line1, line2)
        return similarity
    elif method == 'rle_ncd':
        similarity = rle_ncd.normalized_similarity(line1, line2)
        return similarity
    elif method == 'bwtrle_ncd':
        similarity = bwtrle_ncd.normalized_similarity(line1, line2)
        return similarity
    # Normal compression algorithms
    elif method == 'sqrt_ncd':
        similarity = sqrt_ncd.normalized_similarity(line1, line2)
        return similarity
    elif method == 'entropy_ncd':
        similarity = entropy_ncd.normalized_similarity(line1, line2)
        return similarity
    # Work in progress algorithms that compare two strings as array of bits
    elif method == 'bz2_ncd':
        similarity = bz2_ncd.normalized_similarity(line1, line2)
        return similarity
    elif method == 'lzma_ncd':
        similarity = lzma_ncd.normalized_similarity(line1, line2)
        return similarity
    elif method == 'zlib_ncd':
        similarity = zlib_ncd.normalized_similarity(line1, line2)
        return similarity
    # Phonetic
    elif method == 'mra':
        similarity = mra.normalized_similarity(line1, line2)
        return similarity
    elif method == 'editex':
        similarity = editex.normalized_similarity(line1, line2)
        return similarity
    # Simple
    elif method == 'prefix':
        similarity = prefix.normalized_similarity(line1, line2)
        return similarity
    elif method == 'postfix':
        similarity = postfix.normalized_similarity(line1, line2)
        return similarity
    elif method == 'length':
        similarity = length.normalized_similarity(line1, line2)
        return similarity
    elif method == 'identity':
        similarity = identity.normalized_similarity(line1, line2)
        return similarity
    elif method == 'matrix':
        similarity = matrix.normalized_similarity(line1, line2)
        return similarity
    else:
        return None


def calculate_similarity(data):
    method = data.get('method')
    line1 = data.get('line1')
    line2 = data.get('line2')
    return calculate_method(method, line1, line2)


@app.post("/calculate_similarity/")
async def calculate_similarity_endpoint(data: dict):
    similarity = calculate_similarity(data)
    if similarity is None:
        raise HTTPException(status_code=400, detail="Invalid method specified")

    response_data = {
        'method': data['method'],
        'line1': data['line1'],
        'line2': data['line2'],
        'similarity': similarity
    }
    return response_data


# POST REQUEST 2 (train_classify_models)
# ==========================================================
# ==========================================================

def get_models(name_model):
    if name_model == 'MultinomialNB':
        models = {"Multinomial Naive Bayes": MultinomialNB()}
    elif name_model == 'SVC':
        models = {"Support Vector Machine": SVC(kernel='poly')}
    elif name_model == 'LogisticRegression':
        models = {"Logistic Regression": LogisticRegression()}
    elif name_model == 'RandomForestClassifier':
        models = {"Random Forest": RandomForestClassifier()}
    else:
        models = {
            "Multinomial Naive Bayes": MultinomialNB(),
            "Support Vector Machine": SVC(),
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier()
        }
    return models


def get_data(word_count: int = 0):
    # Параметры текста
    encoding = 'utf8'
    lang = 'english'

    # Шлях до файлу CSV
    file_path = "../../data/IMDB Dataset.csv"

    # Ім'я вилученої колонки файлу CSV
    column_name = 'review'

    # Регулярний вираз для видалення інших символів
    regular_expression = r'blah|_|fffc|br|oz'

    # Поріг зустрічаємості слів
    threshold = 6

    # Метод обрізки слів (лемматизація або стеммінг)
    method = 'lemmatization'

    # Ініціалізація екземпляру CsvPreProcessing
    csv_processor = CsvPreProcessing(file_path, column_name, regular_expression, threshold, method, encoding, lang)

    # Получаем текст з CsvPreProcessing
    text = csv_processor.get_text
    if word_count != 0:
        text = " ".join(text.split()[:word_count])

    # Створення об'єкта для аналізу тональності
    sia = SentimentIntensityAnalyzer()
    sentiment_labels = []

    # Розбиваємо текст на слова
    words = text.split()

    # Аналізуємо настрій кожного слова і додаємо метку в список
    for word in words:
        sentiment_score = sia.polarity_scores(word)
        # Определяем метку настроения на базі compound score
        if sentiment_score['compound'] >= 0.05:
            label = 'positive'
        elif sentiment_score['compound'] <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        sentiment_labels.append((word, label))

    return sentiment_labels


def train_classify_models(data):
    name_model = data.get('model')
    training_set = data.get('training_set')
    testing_set = data.get('testing_set')
    models = get_models(name_model)

    results = {}
    classifier = NLTKClassifier()

    for model_name, model in models.items():
        accuracy = classifier.train_and_evaluate_model(model, training_set, testing_set)
        results[model_name] = accuracy

    return results


@app.post("/train_classify_models/")
async def train_classify_models_endpoint(data: dict):
    results = train_classify_models(data)
    if results is None:
        raise HTTPException(status_code=400, detail="Вказано недійсний метод")

    return results


# POST REQUEST 3 (group_sentences)
# ==========================================================
# ==========================================================

# Завантажити список стоп-слів
stop_words = set(stopwords.words('english'))

# Завантажимо раніше навчену модель
model_path = 'models/doc2vec/doc2vec_combined.model'
try:
    if os.path.isfile(model_path):
        model = Doc2Vec.load(model_path)
    else:
        print("Файл модели не найден")
        exit()
except Exception as e:
    print(f'Произошла ошибка: {e}')


class SentencesRequest(BaseModel):
    sentences: List[str]
    max_clusters: int = 10


def determine_optimal_clusters(vectors: List, texts: List[str], max_clusters: int = 10) -> Tuple[int, List[str]]:
    try:
        sse = []
        silhouette_scores = []
        for num_clusters in range(2, min(max_clusters + 1, len(vectors))):
            kmeans = KMeans(n_clusters=num_clusters, random_state=0)
            kmeans.fit(vectors)
            sse.append(kmeans.inertia_)
            if 1 < num_clusters < len(vectors):
                silhouette_scores.append(silhouette_score(vectors, kmeans.labels_))
            else:
                silhouette_scores.append(float('-inf'))  # Для несовместимых значений

        # Повертаємо кількість кластерів із найбільшим силуетним індексом
        optimal_clusters = range(2, min(max_clusters + 1, len(vectors)))[silhouette_scores.index(max(silhouette_scores))]

        # Кластеризація для отримання імен тем
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=0)
        kmeans.fit(vectors)
        labels = kmeans.labels_

        # Отримання найчастіших пар слів кожного кластера
        # Слова фільтруються за довжиною (не менше 6 літер) та виключаються стоп-слова.
        # Знаходяться дві найчастіші пари слів кожного кластера.
        # Якщо знайдено менше двох відповідних слів, додається лише наявні слова або порожній рядок,
        # якщо немає відповідних слів.
        topic_words = []
        for i in range(optimal_clusters):
            cluster_texts = [texts[j] for j in range(len(texts)) if labels[j] == i]
            all_words = ' '.join(cluster_texts).split()
            # Фільтруємо слова, виключаючи стоп-слова та слова коротше 6 букв
            filtered_words = [word for word in all_words if word.lower() not in stop_words and len(word) >= 6]
            # Отримуємо дві найчастіші пари слів
            most_common_words = [word for word, count in Counter(filtered_words).most_common(2)]
            if len(most_common_words) == 2:
                topic_words.append(' '.join(most_common_words))
            else:
                topic_words.append(' '.join(most_common_words) if most_common_words else "")

        return optimal_clusters, topic_words
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in determine_optimal_clusters: {e}")


@app.post("/group_sentences/")
async def group_sentences(request: SentencesRequest):
    try:
        # Завантажити раніше збережену модель
        model = Doc2Vec.load(model_path)

        # Получение векторов для предложений
        vectors = [model.infer_vector(sentence.split()) for sentence in request.sentences]

        # Визначення оптимальної кількості кластерів та їх імен
        num_clusters, topic_names = determine_optimal_clusters(vectors, request.sentences, request.max_clusters)

        # Кластеризація пропозицій
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(vectors)
        labels = kmeans.labels_

        # Угруповання пропозицій за темами
        grouped_sentences = {}
        for label, sentence in zip(labels, request.sentences):
            topic = topic_names[label]
            if topic not in grouped_sentences:
                grouped_sentences[topic] = []
            grouped_sentences[topic].append(sentence)
        return grouped_sentences
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid parameter group_sentences: {e}")


# POST REQUEST 4 (nltk_preprocessing, spacy_preprocessing)
# ==========================================================
# ==========================================================

def spacy_preprocessing_text(text):
    # Завантаження моделі SpaCy
    nlp = spacy.load("en_core_web_sm")

    # Функція обробки частини тексту
    def process_text_chunk(chunk):
        doc = nlp(chunk)
        tokens = [token.text for token in doc if not token.is_stop and token.is_alpha]
        return tokens

    # Функція для лематизації частини тексту
    def lemmatize_text_chunk(chunk):
        doc = nlp(chunk)
        return [token.lemma_ for token in doc]

    # Обробка тексту за допомогою регулярного виразу
    regex_patterns = r'blah|_|fffc|br|oz|< />|<br />|\d+|\W+'
    text = re.sub(regex_patterns, ' ', text)

    # Розбиваємо текст на частини по 100 000 символів
    chunk_size = 100000
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    # Обробляємо кожну частину та об'єднуємо результати
    all_tokens = []
    for chunk in chunks:
        all_tokens.extend(process_text_chunk(chunk))

    # Підрахунок частоти слів
    word_freq = Counter(all_tokens)

    # Визначення рідкісних слів (слова, що зустрічаються менше ніж 5 разів)
    rare_words = {word for word, count in word_freq.items() if count < 5}

    # Видалення рідкісних слів
    filtered_tokens = [word for word in all_tokens if word not in rare_words]

    # Розбиваємо текст на частини по 100000 символів для лематизації
    filtered_text = ' '.join(filtered_tokens)
    filtered_chunks = [filtered_text[i:i + chunk_size] for i in range(0, len(filtered_text), chunk_size)]

    # Лематизація кожної частини та об'єднання результатів
    lemmatized_tokens = []
    for chunk in filtered_chunks:
        lemmatized_tokens.extend(lemmatize_text_chunk(chunk))

    # Об'єднання оброблених токенів у рядок
    cleaned_text = ' '.join(lemmatized_tokens)

    # Повернення обробленого тексту
    return cleaned_text


def nltk_preprocessing_text(text):
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    # Перетворення тексту на нижній регістр
    text = text.lower()

    # Токенізація тексту з використанням регулярного виразу
    regex_patterns = r'blah|_|fffc|br|oz|< />|<br />|\d|\w+'
    tokenizer = RegexpTokenizer(regex_patterns)

    # Токенізація тексту
    tokens = word_tokenize(text)

    # Завантаження стоп-слів
    stop_words = set(stopwords.words('english'))

    # Видалення стоп-слів
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]

    # Підрахунок частоти слів
    word_freq = Counter(tokens)

    # Визначення рідкісних слів (слова, що зустрічаються менше ніж 5 разів)
    rare_words = {word for word, count in word_freq.items() if count < 5}

    # Видалення рідкісних слів
    tokens = [word for word in tokens if word not in rare_words]

    # Лематизація
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Об'єднання оброблених токенів у рядок
    cleaned_text = ' '.join(tokens)

    # Повернення обробленого тексту
    return cleaned_text


class DataRequest(BaseModel):
    data: str


@app.post("/nltk_preprocessing/")
async def nltk_preprocessing(request: DataRequest):
    response_data = {'nltk_cleaned_text': nltk_preprocessing_text(request.data)}
    return response_data


@app.post("/spacy_preprocessing/")
async def spacy_preprocessing(request: DataRequest):
    response_data = {'spacy_cleaned_text': spacy_preprocessing_text(request.data)}
    return response_data

# POST REQUEST 5 (translator_nllb, перекладач із використанням моделі NLLB)
# ==========================================================
# ==========================================================


class ChatbotRequest(BaseModel):
    input_text: str
    source_language = "eng_Latn"
    target_language: str = "ukr_Cyrl"  # Мова цільової моделі


# Завантаження моделі та токенізатора NLLB
tokenizer_nllb = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model_nllb = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")


@app.post("/translator_nllb/")
async def chatbot_endpoint(request: ChatbotRequest):
    try:
        input_text = request.input_text
        sentences = input_text.split('.')
        target_language = request.target_language
        source_language = request.source_language

        tokenizer_nllb.src_lang = source_language
        tokenizer_nllb.tgt_lang = target_language

        # Перекладаємо текст
        translated_sentences = []
        max_length = 512
        generation_config = GenerationConfig(max_length=max_length)  # Налаштовуємо параметри генерації

        for sentence in sentences:
            inputs = tokenizer_nllb(sentence, return_tensors="pt", max_length=max_length, truncation=True, padding=True)
            inputs = {k: v.to(model_nllb.device) for k, v in inputs.items()}
            outputs = model_nllb.generate(
                **inputs,
                max_length=max_length,
                forced_bos_token_id=tokenizer_nllb.lang_code_to_id[target_language],
                generation_config=generation_config)
            translated_sentence = tokenizer_nllb.decode(outputs[0], skip_special_tokens=True)
            translated_sentences.append(translated_sentence)

        response_text = translated_sentences[0]

        return {"response_text": response_text, "target_language": target_language}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


# POST REQUEST 6 (розпізнавання аудіо англійської мови та переклад її на українську)
# ==========================================================
# ==========================================================


# Функція для перетворення аудіо на текст
def audio_to_text(audio_file_path):
    """
    Функція audio_to_text(audio_file_path) приймає шлях до аудіофайлу та перетворює аудіо в текст,

    використовуючи сервіс Google для розпізнавання мовлення.

    Параметри:
        audio_file_path (str): Шлях до аудіофайлу.

    Повертає:
        str: Розпізнаний текст з аудіофайлу або повідомлення про помилку.

    Функція працює таким чином:
    1. Створює об'єкт розпізнавача мовлення.
    2. Відкриває аудіофайл за заданим шляхом.
    3. Записує аудіодані з файлу.
    4. Виконує розпізнавання мовлення за допомогою сервісу Google.
    5. Якщо розпізнавання вдалося, повертає текст.
    6. Якщо аудіо не вдалося розпізнати, повертає повідомлення "Не вдалося розпізнати аудіо".
    7. Якщо сталася помилка сервісу розпізнавання мовлення, повертає повідомлення про помилку з детальною інформацією.
    """

    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio_data, language="en-EN")
            return text
        except sr.UnknownValueError:
            return "Не вдалося розпізнати аудіо"
        except sr.RequestError as e:
            return f"Помилка сервісу розпізнавання мовлення; {e}"


# Функція для перекладу тексту
def translate_text(text, dest_language="uk"):
    """
    Функція translate_text(text, dest_language="uk") приймає текст і перекладає його на вказану мову.

    Параметри:
        text (str): Текст, який необхідно перекласти.
        dest_language (str, опціонально): Цільова мова перекладу. За замовчуванням "uk" (українська).

    Повертає:
        str: Перекладений текст або повідомлення про помилку.

    Функція працює таким чином:
    1. Створює об'єкт перекладача з вказаною цільовою мовою.
    2. Виконує переклад заданого тексту.
    3. Якщо переклад успішний, повертає перекладений текст.
    4. Якщо сталася помилка під час перекладу, повертає повідомлення "Помилка перекладу" з детальною інформацією про помилку.
    """

    translator = Translator(to_lang=dest_language)
    try:
        translation = translator.translate(text)
        return translation
    except Exception as e:
        return f"Помилка перекладу: {e}"


@app.post("/speech_to_text/")
async def speech_to_text(file: UploadFile = File(...)):  # process_audio
    try:
        # Зберігаємо завантажений файл
        with open(f"temp_{file.filename}", "wb") as f:
            f.write(await file.read())

        # Преобразуем аудіо в текст
        original_text = audio_to_text(f"temp_{file.filename}")

        # Переводим текст на украинский язык
        translated_text = translate_text(original_text)

        # Видаляємо тимчасовий файл
        os.remove(f"temp_{file.filename}")

        return JSONResponse(content={
            "original_text": original_text,
            "translated_text": translated_text
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================================
# ==========================================================

if __name__ == "__main__":
    # Використовуйте це лише для налагодження
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="debug")
