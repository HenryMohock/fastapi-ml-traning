import requests


# Процедура выводящая результат POST запроса
def post_request(url, methods, line1, line2):
    for method in methods:
        data = {
        'method': method,
        'line1': line1,
        'line2': line2
        }
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print(response.json())
        else:
            print('Error:', response.text)

# Функция получающая текст запроса
def get_url(server, post_method):
    slash = '/'
    return f"{server}{post_method}{slash}"

# Функция возвращает содержимое текстового файла
def get_text_of_string(name_file):
    with open(name_file, 'r') as file:
        return file.read()


# Адрес сервера
server = 'http://127.0.0.1:8000/'
# Метод на сервере возвращающий результат запроса
post_method = 'calculate_similarity'

# Строки для сравнения
line_1 = get_text_of_string('line1.txt')  # 'This is a test string'
line_2 = get_text_of_string('line1.txt')  # 'This is another test string'

# Создание массива методов на основе Edit based
methods_edit_based = ["hamming", "mlipns", "levenshtein", "damerau_levenshtein",
                      "strcmp95", "needleman_wunsch", "gotoh", "smith_waterman"]

# Создание массива методов на основе токенов (Token based)
methods_token_based = ["jaccard", "sorensen", "tversky", "overlap",
                       "tanimoto", "cosine", "monge_elkan", "bag"]

# Создание массива методов на основе последовательностей (Sequence based)
methods_sequence_based = ["lcsseq", "lcsstr", "ratcliff_obershelp"]

# Создание массива методов на основе классических алгоритмов сжатия (Classic compression algorithms)
methods_classic_compression = ["arith_ncd", "rle_ncd", "bwtrle_ncd"]

# Создание массива методов на основе обычных алгоритмов сжатия (Normal compression algorithms)
methods_normal_compression = ["sqrt_ncd", "entropy_ncd"]

# Создание массива методов которые сравнивают две строки как массив битов
# (Work in progress algorithms that compare two strings as array of bits)
methods_compare_bits = ["bz2_ncd", "lzma_ncd", 'zlib_ncd']

# Создание массива фонетических методов (Phonetic)
methods_phonetic = ["mra", "mra"]

# Создание массива простых методов (Simple)
methods_simple = ["prefix", "postfix", 'length', 'identity', 'matrix']

# Получаем строку запроса
url = get_url(server, post_method)
print(url)

# Посылаем запросы серверу

print("Алгоритмы на основе Edit based: ")
post_request(url, methods_edit_based, line_1, line_2)

print("Алгоритмы на основе токенов (Token based): ")
post_request(url, methods_token_based, line_1, line_2)

print("Алгоритмы на основе последовательностей (Sequence based): ")
post_request(url, methods_sequence_based, line_1, line_2)

print("Алгоритмы на основе классических алгоритмов сжатия (Classic compression algorithms): ")
post_request(url, methods_classic_compression, line_1, line_2)

print("Алгоритмы на основе обычных алгоритмов сжатия (Normal compression algorithms): ")
post_request(url, methods_normal_compression, line_1, line_2)

print("Алгоритмы сравнения двух строк как массива битов (Work in progress algorithms that compare two strings as array of bits): ")
post_request(url, methods_compare_bits, line_1, line_2)

print("Алгоритмы на основе фонетических методов (Phonetic): ")
post_request(url, methods_phonetic, line_1, line_2)

print("Алгоритмы на основе простых методов (Simple): ")
post_request(url, methods_simple, line_1, line_2)















