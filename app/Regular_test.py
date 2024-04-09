import re

print('1: ==============================================')
# 1)
# Write a Python program that matches a word containing 'z',
# not the start or end of the word
# ===========================================================
def match_word_with_z(word):
    pattern = r'\b[^z\s]+\w*z\w*[^z\s]+\b'
    return bool(re.search(pattern, word))


# Тестирование функции
words = ["apple", "amazing", "bazooka", "zebra", "haze", "zoo", "blaze", "fizz", "lazy", "puzzle"]

for word in words:
    if match_word_with_z(word):
        print(f"{word} відповідає шаблону.")
    else:
        print(f"{word} не відповідає шаблону.")

print('2: ==============================================')
# 2)
# Write a Python program to remove leading zeros from an IP address
# =================================================================
def remove_leading_zeros(ip_address):
    # Используем регулярное выражение для удаления ведущих нулей из каждой части IP-адреса
    cleaned_ip = re.sub(r'\b0+(\d)', r'\1', ip_address)
    return cleaned_ip


# Тестирование функции
ip_addresses = ["192.168.001.001", "010.010.010.010", "0.0.0.1", "255.255.255.255"]

for ip in ip_addresses:
    cleaned_ip = remove_leading_zeros(ip)
    print(f"Исходный IP: {ip}, Очищенный IP: {cleaned_ip}")

print('3: ==============================================')
# 3)
# Write a Python program to find the occurrence and position of substrings within a string
# (використайте атрбути знайденої групи)
# ========================================================================================
def find_substring_position(main_string, substring):
    # Ищем вхождение подстроки в строку
    match = re.search(substring, main_string)

    if match:
        # Возвращаем позицию начала вхождения и само вхождение
        return match.start(), match.group()
    else:
        return None, None


# Тестирование функции
main_string = "Hello, world! This is a test string."
substring = "world"

position, found_substring = find_substring_position(main_string, substring)

if position is not None:
    print(f"Подстрока '{found_substring}' найдена в позиции {position} в строке '{main_string}'.")
else:
    print(f"Подстрока '{substring}' не найдена в строке '{main_string}'.")

print('4: ==============================================')
# 4)
# Write a Python program to convert a date of yyyy-mm-dd format to dd-mm-yyyy format
# ==================================================================================
def transform_date(date_string):
    # Используем регулярное выражение для поиска даты в формате yyyy-mm-dd
    pattern = r'(\d{4})-(\d{2})-(\d{2})'

    # Преобразуем дату в формат dd-mm-yyyy
    transformed_date = re.sub(pattern, r'\3-\2-\1', date_string)

    return transformed_date


# Тестирование функции
date_strings = ["2024-04-09", "2023-12-25", "2025-01-01"]

for date_str in date_strings:
    transformed_date = transform_date(date_str)
    print(f"Исходная дата: {date_str}, Преобразованная дата: {transformed_date}")

print('5: ==============================================')
# 5)
# Write a Python program to find all three, four, and five character words in a string
# ====================================================================================
def find_words(string):
    # Используем регулярное выражение для поиска трех-, четырех- и пяти-символьных слов
    pattern = r'\b\w{3,5}\b'

    # Находим все совпадения в строке
    words = re.findall(pattern, string)

    return words


# Тестирование функции
text = "Python is a powerful programming language. It is used for web development, data science, and artificial intelligence."
found_words = find_words(text)

print("Найденные слова:", found_words)

print('6: ==============================================')
# 6)
# Write a Python program to convert a camel-case string to a snake-case string
# ==============================================================================
def camel_to_snake(camel_string):
    # Используем регулярное выражение для поиска заглавных букв в camel-case строке
    snake_string = re.sub(r'([A-Z])', r'_\1', camel_string)

    # Преобразуем строку в нижний регистр и удаляем возможный первый символ '_'
    snake_string = snake_string.lower().lstrip('_')

    return snake_string


# Тестирование функции
camel_strings = \
    ["ABitterAndBloody",
     "WarInUkraineHasDevastatedTheCountry",
     "FurtherIsolatedRussiaFromTheWest",
     "AndFueledEconomicInsecurityAroundTheWorld"
     ]

for camel_str in camel_strings:
    snake_str = camel_to_snake(camel_str)
    print(f"Исходная строка: {camel_str}, Преобразованная строка: {snake_str}")

print('7: ==============================================')
# 7)
# Write a Python program to find all adverbs and their positions in a given sentence
# ====================================================================================
def find_adverbs_positions(sentence):
    # Используем регулярное выражение для поиска наречий
    pattern = r'\b\w+ly\b'

    # Находим все совпадения в предложении
    matches = re.finditer(pattern, sentence)

    # Собираем позиции и найденные наречия в словарь
    adverbs = {match.group(): match.start() for match in matches}

    return adverbs


# Тестирование функции
sentence = "She sings beautifully and runs quickly."

adverbs_positions = find_adverbs_positions(sentence)

for adverb, position in adverbs_positions.items():
    print(f"Наречие '{adverb}' найдено в позиции {position} в предложении '{sentence}'.")

print('8: ==============================================')
# 8)
# Write a Python program to concatenate the consecutive numbers in a given string.
# =================================================================================

def concatenate_consecutive_numbers(input_string):
    # Use regular expression to find consecutive numbers
    pattern = r'(\d+)(?:\s)(\d+)'

    # Find all matches
    matches = re.findall(pattern, input_string)

    # Concatenate consecutive numbers
    for match in matches:
        concatenated = match[0] + match[1]
        input_string = input_string.replace(f"{match[0]} {match[1]}", concatenated)

    return input_string


# Test the function
original_string = "Enter at 1 20 Kearny Street. The security desk can direct you to floor 1 6. Please have your identification ready."
concatenated_string = concatenate_consecutive_numbers(original_string)

print("Исходная строка:")
print(original_string)
print("\nПосле объединения последовательных чисел в указанной строке:")
print(concatenated_string)

print('==============================================')
