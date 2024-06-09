import spacy
from spacy.tokens import Doc, Span, Token
from spacy.language import Language


class AdditionalComponents:
    """
    Класс AdditionalComponents предназначен для определения и добавления дополнительных компонентов

    обработки текста в пайплайн SpaCy.

    Attributes:
    ----------
    nlp : spacy.language.Language
        Объект языковой модели SpaCy, в который будут добавлены компоненты.
    words : list
        Список слов и биграмм для подсчета их вхождений в тексте.

    Methods:
    -------
    __init__(self, nlp, words):
        Инициализирует класс с объектом языковой модели SpaCy и устанавливает расширения.

    setup_extensions(self):
        Устанавливает расширения для токенов, спанов и документа.

    length_component(self, doc):
        Компонент для вычисления длины каждого токена в документе.

    sentence_length_component(self, doc):
        Компонент для вычисления длины каждого предложения в документе.

    document_length_component(self, doc):
        Компонент для вычисления общей длины документа.

    sentence_count_component(self, doc):
        Компонент для подсчета количества предложений в документе.

    word_count_component(self, doc):
        Компонент для подсчета количества встречающихся слов и биграмм в документе.
    """

    def __init__(self, nlp, words):
        """
        Инициализирует класс AdditionalComponents.

        Parameters:
        -----------
        nlp : spacy.language.Language
            Объект языковой модели SpaCy, в который будут добавлены компоненты.
        words : list
            Список слов и биграмм для подсчета их вхождений в тексте.
        """
        self.nlp = nlp
        self.words = words
        self.setup_extensions()

    def setup_extensions(self):
        """
        Устанавливает расширения для токенов, спанов и документа.

        .
        """
        Token.set_extension('length', default=None)
        Span.set_extension('length', default=None)
        Doc.set_extension('sentences', default=[])
        Doc.set_extension('total_length', default=None)
        Doc.set_extension('sentence_count', default=None)
        Doc.set_extension('word_counts', default=None)

    @Language.component('length_component')
    def length_component(self, doc):
        """
        Компонент для вычисления длины каждого токена в документе.

        Args:
        doc (Doc): Объект документа SpaCy.

        Returns:
        Doc: Обработанный объект документа SpaCy.
        """
        for token in doc:
            token._.length = len(token.text)
        return doc

    @Language.component('sentence_length_component')
    def sentence_length_component(self, doc):
        """
        Компонент для вычисления длины каждого предложения в документе.

        Args:
        doc (Doc): Объект документа SpaCy.

        Returns:
        Doc: Обработанный объект документа SpaCy.
        """
        sentences = []
        sentence_start = 0
        for i, token in enumerate(doc):
            if token.text == '.':
                sentence_end = i
                sentence_length = sum(len(tok.text) for tok in doc[sentence_start:sentence_end])
                span = Span(doc, sentence_start, sentence_end + 1)
                span._.length = sentence_length
                sentences.append(span)
                sentence_start = sentence_end + 1
        doc._.sentences = sentences
        return doc

    @Language.component('document_length_component')
    def document_length_component(self, doc):
        """
        Компонент для вычисления общей длины документа.

        Args:
        doc (Doc): Объект документа SpaCy.

        Returns:
        Doc: Обработанный объект документа SpaCy.
        """
        total_length = sum(len(token.text) for token in doc)
        doc._.total_length = total_length
        return doc

    @Language.component('sentence_count_component')
    def sentence_count_component(self, doc):
        """
        Компонент для подсчета количества предложений в документе.

        Args:
        doc (Doc): Объект документа SpaCy.

        Returns:
        Doc: Обработанный объект документа SpaCy.
        """
        sentence_count = sum(1 for token in doc if token.text == '.')
        doc._.sentence_count = sentence_count
        return doc

    @Language.component('word_count_component')
    def word_count_component(self, doc):
        """
        Компонент для подсчета количества встречающихся слов и биграмм в документе.

        Args:
        doc (Doc): Объект документа SpaCy.

        Returns:
        Doc: Обработанный объект документа SpaCy.
        """
        word_counts = {word: 0 for word in self.words}
        for token in doc:
            token_text = token.text
            if token_text in self.words:
                word_counts[token_text] += 1
        for i in range(len(doc) - 1):
            bigram = f"{doc[i].text} {doc[i+1].text}"
            if bigram in self.words:
                word_counts[bigram] += 1
        doc._.word_counts = word_counts
        return doc

# Инициализация модели spaCy и компонентов
nlp = spacy.load('en_core_web_sm')
words_to_count = ["SpaceX", "Flight 4", "Starship", "Elon Musk"]
additional_components = AdditionalComponents(nlp, words_to_count)


# Добавление компонентов в пайплайн с использованием декораторов @Language.component
@Language.component('length_component')
def length_component(doc):
    return additional_components.length_component(doc)


@Language.component('sentence_length_component')
def sentence_length_component(doc):
    return additional_components.sentence_length_component(doc)


@Language.component('document_length_component')
def document_length_component(doc):
    return additional_components.document_length_component(doc)


@Language.component('sentence_count_component')
def sentence_count_component(doc):
    return additional_components.sentence_count_component(doc)


@Language.component('word_count_component')
def word_count_component(doc):
    return additional_components.word_count_component(doc)


# Добавление компонентов в пайплайн
nlp.add_pipe('length_component', last=True)
nlp.add_pipe('sentence_length_component', last=True)
nlp.add_pipe('document_length_component', last=True)
nlp.add_pipe('sentence_count_component', last=True)
nlp.add_pipe('word_count_component', last=True)

# Загрузка данных из текстового файла
with open('../../data/doc_pipeline.txt', 'r') as file:
    text = file.read()

# Анализ текста
doc = nlp(text)

# Вывод результатов
print()
for token in doc:
    print(f"Length: {token._.length},   Token: {token.text}")

print()
for sent in doc._.sentences:
    print(f"Length: {sent._.length},    Sentence: {sent.text}")

print()
print(f"Total document length: {doc._.total_length} characters")
print()
print(f"Total sentence count: {doc._.sentence_count} sentences")
print()
print(f"Word counts: {doc._.word_counts}")
