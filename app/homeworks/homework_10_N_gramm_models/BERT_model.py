from transformers import GPT2Tokenizer, GPT2LMHeadModel
from app.homeworks.homework_10_N_gramm_models.Preparing_text import Preliminary_preparation_data
import asyncio
import TranslatorX
from transformers import BertTokenizer, BertForMaskedLM
import torch

# Получение данных
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


async def translate_to_ukrainian_asinc(text):
    trans = TranslatorX.Translator()
    trans_text = trans.Translate(text=text, to_lang='uk')
    return trans_text


def translate_to_ukrainian(text):
    return asyncio.run(translate_to_ukrainian_asinc(text))


# Получение данных
sentence, corpus = get_data(word_count=900)


# Загрузка предобученного токенизатора и модели BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

# Токенизация предложения
tokenized_text = tokenizer.tokenize(sentence)
# Обрезка последовательности токенов до 512 элементов
tokenized_text = tokenized_text[:510]

# Добавление токена начала последовательности ([CLS]) и токена конца последовательности ([SEP])
tokenized_text = ["[CLS]"] + tokenized_text + ["[SEP]"]

# Преобразование токенов в индексы
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Преобразование в тензор
tokens_tensor = torch.tensor([indexed_tokens])

# Генерация текста с помощью модели
with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]

# Получение индекса токена для предсказания следующего слова
predicted_index = torch.argmax(predictions[0, -1, :]).item()

# Получение предсказанного слова
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]

# Добавление предсказанного слова к предложению
generated_text = tokenized_text + [predicted_token]
result_sentence = " ".join(generated_text)
result_sentence = result_sentence.replace('##', '\n')
print()
print(result_sentence)
print()
print(translate_to_ukrainian(result_sentence))

