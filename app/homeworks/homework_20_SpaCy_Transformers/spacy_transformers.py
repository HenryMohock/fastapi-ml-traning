import spacy
from scipy.spatial.distance import cosine
import numpy as np

# Загрузка большой модели spaCy с векторными представлениями слов
nlp = spacy.load("en_core_web_lg")

# Вывод компонентов пайплайна
print()
print("Компоненти у процесі pipeline:\n", nlp.pipe_names)
print('де:')
for name, component in nlp.pipeline:
    print(name, component)

# Определение пар слов, которые пишутся одинаково, но имеют разный контекст
word_pairs = [
    ("bank", "bank"),  # финансовое учреждение vs. берег реки
    ("bat", "bat"),    # летучая мышь vs. спортивный инвентарь
    ("lead", "lead"),  # вести, руководить vs. металл свинец
    ("bark", "bark")   # лай собаки vs. кора дерева
]

# Определение предложений для каждого слова в парах для предоставления контекста
sentences = [
    "I need to deposit money in the bank.",        # Мне нужно положить деньги в банк.
    "The river bank was flooded after the storm.", # Берег реки был затоплен после шторма.
    "A bat flew out of the cave at dusk.",         # Летучая мышь вылетела из пещеры на закате.
    "He swung the bat and hit a home run.",        # Он размахнулся битой и сделал хоум-ран.
    "She will lead the team to victory.",          # Она приведет команду к победе.
    "The pipe is made of lead.",                   # Труба сделана из свинца.
    "The dog's bark was loud and clear.",          # Лай собаки был громким и отчетливым.
    "The bark of the tree is very rough."          # Кора дерева очень шершавая.
]

# Обработка предложений с помощью пайплайна spaCy и извлечение эмбеддингов
docs = list(nlp.pipe(sentences))
embeddings = [doc.vector for doc in docs]

# Вычисление косинусного расстояния для каждой пары предложений в парах слов
similarities = {}
for i in range(0, len(sentences), 2):
    pair = word_pairs[i // 2]
    # Проверка, что векторные представления не пустые
    if np.any(embeddings[i]) and np.any(embeddings[i+1]):
        similarity = 1 - cosine(embeddings[i], embeddings[i+1])
        # Преобразование схожести в проценты и округление до 2 знаков
        similarity_percent = round(similarity * 100, 2)
    else:
        similarity_percent = None  # Установить None, если одно из векторных представлений пустое
    similarities[pair] = f'{similarity_percent} %'

# Вывод результатов
print()
print('Подібність:')
print(similarities)

# Пояснение
print()
print('Пояснення:\n')
explanation = (
    '61.57% для "bank": Це вказує на те, що модель знайшла значну схожість між "банком" як фінансову установу\n'
    'та "берег річки". Обидва контексти можуть бути пов`язані через концепцію зберігання, захисту чи кордону.\n'
    '\n'
    '64.64% для "bat": Це показує помірну схожість між "кажан" і "біта". Ці контексти можуть бути схожі\n'
    'через фізичних властивостей об`єктів (швидкість польоту), хоча вони істотно різняться за призначенням.\n'
    '\n'
    '52.98% для "lead": Це найнижча схожість, що логічно, так як "керувати" (діяти) та "свинець" (метал) мають\n'
    'найменший концептуальний перетин серед усіх пар.\n'
    '\n'
    '78.22% для "bark": Це найвища схожість, і вона цілком логічна, так як гавкіт собаки та кора дерева можуть бути\n'
    'концептуально пов`язані через ідею зовнішнього покриву чи оболонки (захисний покрив живих істот).\n'
)
print(explanation)



