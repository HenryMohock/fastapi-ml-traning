#  Проверка точности модели:
#  =========================
import spacy


# Загрузка обученной модели
nlp_trained = spacy.load("./ner_model")

# Тестовые данные
test_text = """
3 people reported a 5% increase in the revenue of Company Apple. The deal was finalized in New York. 
The American and British teams collaborated with Company Microsoft. Japanese and Chinese representatives also participated 
in the meeting, along with members from the EU and NATO. This is San Francisco.
In by the incredible Lionel Messi, La Albiceleste were crowned world champions for a third time on Sunday on FIFA World Cup
"""

# Применение модели. Визуальный контроль
doc = nlp_trained(test_text)
print(doc)
for ent in doc.ents:
    print(ent.text, ent.label_)
