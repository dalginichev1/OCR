import json
import re
from symspellpy import SymSpell, Verbosity
from nltk.tokenize import word_tokenize
import nltk

try:
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    print(f"Ошибка при загрузке punkt_tab: {e}")
    exit(1)

sym_spell = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)

dictionary_path = "eng.txt"
if not sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1, encoding='utf-8'):
    print("Ошибка: Не удалось загрузить словарь. Проверьте путь к файлу", dictionary_path)
    exit(1)


def correct_text(text):
    try:
        text = text.encode('utf-8').decode('utf-8')
    except UnicodeEncodeError:
        pass

    tokens = word_tokenize(text)
    corrected_tokens = []
    corrections = []

    for word in tokens:
        if re.match(r'^[a-zA-Z-]+$', word):
            suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=3)
            if suggestions:
                corrected_word = suggestions[0].term
                if corrected_word != word:
                    corrections.append(f"{word} -> {corrected_word}")
                corrected_tokens.append(corrected_word)
            else:
                corrected_tokens.append(word)
        else:
            corrected_tokens.append(word)

    corrected_text = ""
    last_pos = 0
    for word in tokens:
        start = text.find(word, last_pos)
        if start == -1:
            corrected_text += corrected_tokens.pop(0)
            continue
        corrected_text += text[last_pos:start] + corrected_tokens.pop(0)
        last_pos = start + len(word)
    corrected_text += text[last_pos:]

    if corrections:
        print("Исправления:")
        for correction in corrections:
            print(correction)
    else:
        print("Исправления не найдены.")

    return corrected_text


input_file = "input.json"
output_file = "output.json"

try:
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'data' in data and 'text' in data['data']:
        original_text = data['data']['text']
        corrected_text = correct_text(original_text)
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
