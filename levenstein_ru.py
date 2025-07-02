import json
import re
from razdel import tokenize


def levens(s1, s2):
    if len(s1) > len(s2):
        return levens(s2, s1)

    prev_row = range(len(s2) + 1)
    for i in range(1, len(s1) + 1):
        res_row = [i]
        for j in range(1, len(s2) + 1):
            insert, delete, change = prev_row[j] + 1, res_row[j - 1] + 1, prev_row[j - 1] + (s1[i - 1] != s2[j - 1])
            res_row.append(min(insert, delete, change))

        prev_row = res_row

    return prev_row[-1]


def load_dictionary(dictionary_path):
    try:
        with open(dictionary_path, 'r', encoding='utf-8') as f:
            return [line.strip().split()[0] for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Ошибка: Словарь {dictionary_path} не найден.")
        exit(1)


def correct_text(text, dictionary, max_distance=3):
    try:
        text = text.encode('utf-8').decode('utf-8')
    except UnicodeEncodeError:
        pass

    tokens = list(tokenize(text))
    corrected_tokens = []
    corrections = []

    for token in tokens:
        word = token.text
        if re.match(r'^[а-яА-ЯёЁ-]+$', word):
            min_distance = float('inf')
            best_match = word
            for dict_word in dictionary:
                distance = levens(word.lower(), dict_word.lower())
                if distance < min_distance:
                    min_distance = distance
                    best_match = dict_word
            if min_distance <= max_distance and best_match != word:
                if word[0].isupper():
                    best_match = best_match.capitalize()
                corrections.append(f"{word} -> {best_match}")
                corrected_tokens.append(best_match)
            else:
                corrected_tokens.append(word)
        else:
            corrected_tokens.append(word)

    corrected_text = ""
    last_stop = 0
    for token, corrected in zip(tokens, corrected_tokens):
        corrected_text += text[last_stop:token.start] + corrected
        last_stop = token.stop
    corrected_text += text[last_stop:]

    if corrections:
        print("Исправления:")
        for correction in corrections:
            print(correction)
    else:
        print("Исправления не найдены.")

    return corrected_text


dictionary_path = "ru-100k.txt"
dictionary = load_dictionary(dictionary_path)

input_file = "input_ru.json"
output_file = "output_ru.json"

try:
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'data' in data and 'text' in data['data']:
        original_text = data['data']['text']
        corrected_text = correct_text(original_text, dictionary, max_distance=3)
        data['data']['text'] = corrected_text

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"\nТекст исправлен:\nОригинал: {original_text}\nИсправленный: {corrected_text}")
        print("Результат сохранён в", output_file)
    else:
        print("Ошибка: Ключ ['data']['text'] не найден в файле.")

except FileNotFoundError:
    print(f"Ошибка: Файл {input_file} не найден.")
except json.JSONDecodeError:
    print("Ошибка: Неверный формат JSON в файле.")
except Exception as e:
    print(f"Произошла ошибка: {str(e)}")
