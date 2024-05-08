from transformers import GPT2Tokenizer, GPT2LMHeadModel
from app.homeworks.homework_10_N_gramm_models.Preparing_text import Preliminary_preparation_data
import asyncio
import TranslatorX

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


# Загрузка предобученного токенизатора и модели GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Получение данных
sentence, corpus = get_data(word_count=900)

# Получение n-грамм из предложения
ngrams = tokenizer.encode(sentence, return_tensors="pt")

# Генерация текста с помощью модели
output = model.generate(input_ids=ngrams, max_length=ngrams.size(1) + 15,
                        num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

# Декодирование сгенерированного текста
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
generated_text_words = generated_text.split()[:500]
result_sentence = ' '.join(generated_text_words)
print()
print(result_sentence)
print()
print(translate_to_ukrainian(result_sentence))
