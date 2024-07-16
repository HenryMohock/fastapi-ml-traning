import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Імпортуємо необхідні бібліотеки
import time
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, GenerationConfig
from datasets import Dataset, DatasetDict
import torch
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration

# Функция форматирования времени
def format_time(seconds):
    """Повертає час в секундах у форматі ЧЧ:ММ:СС."""
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return "{:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds))


# Функция форматирования текущего времени
def format_current_time():
    """Повертає поточний час у форматі ЧЧ:ММ:СС."""
    current_time_seconds = time.time()
    struct_time = time.localtime(current_time_seconds)
    formatted_time = time.strftime("%H:%M:%S", struct_time)
    return formatted_time


# Шлях до файлу з текстом
file_path = '../../data/Bible_NIV.txt'
output_path = '../../data/Bible_NIV_UA.txt'

# Читаємо файл з текстом
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# Розбиваємо текст на речення
sentences = text.split('.')

# Використовуємо меншу частину даних для прискорення навчання
sentences = sentences[:10]

# Розділяємо дані на навчальний та валідаційний набори
train_sentences = sentences[:8]
eval_sentences = sentences[8:]

# Завантажуємо токенізатор та модель NLLB
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

# Встановлюємо мови для токенізатора
source_lang = "eng_Latn"
target_lang = "ukr_Cyrl"

tokenizer.src_lang = source_lang
tokenizer.tgt_lang = target_lang

# Функція для попередньої обробки даних
def preprocess_function(examples):
    """Попереднє оброблення даних для моделі Seq2Seq."""
    inputs = [example for example in examples['text']]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')
    labels = tokenizer(text_target=inputs, max_length=512, truncation=True, padding='max_length')
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Створюємо датасет з речень
train_dataset = Dataset.from_dict({"text": train_sentences})
eval_dataset = Dataset.from_dict({"text": eval_sentences})

train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# Додаємо обробник для колаборації даних
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Налаштовуємо параметри тренування
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=64,  # Збільшено розмір батчу
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,  # Зменшено кількість епох
    predict_with_generate=True,
    fp16=torch.cuda.is_available()
)

# Ініціалізуємо акселератор
dataloader_config = DataLoaderConfiguration(
    dispatch_batches=None,
    split_batches=False,
    even_batches=True,
    use_seedable_sampler=True
)
accelerator = Accelerator(dataloader_config=dataloader_config)

# Створюємо тренера
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Додаємо валідаційний датасет
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Використовуємо акселератор для тренування
trainer.accelerator = accelerator

print(f'\nНавчання почалось з: {format_current_time()}')
start_time = time.time()

# Тренуємо модель
# =================
trainer.train()
# =================

print(f'Час навчання: {format_time(time.time() - start_time)}\n')

# Зберігаємо дотреновану модель
model.save_pretrained("../../models/NLLB/fine_tuned_nllb_model")
tokenizer.save_pretrained("../../models/NLLB/fine_tuned_nllb_tokenizer")

# Перекладаємо текст та зберігаємо переклад у файл
translated_sentences = []
max_length = 512
generation_config = GenerationConfig(max_length=max_length)  # Налаштовуємо параметри генерації

for sentence in sentences:
    inputs = tokenizer(sentence, return_tensors="pt", max_length=max_length, truncation=True, padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
        generation_config=generation_config)
    translated_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    translated_sentences.append(translated_sentence)

# Зберігаємо перекладений текст у файл
with open(output_path, 'w', encoding='utf-8') as output_file:
    output_file.write('. '.join(translated_sentences))
