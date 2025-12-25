def ru_to_lat(text: str) -> str:
    mapping = {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d",
        "е": "e", "ё": "yo", "ж": "zh", "з": "z", "и": "i",
        "й": "y", "к": "k", "л": "l", "м": "m", "н": "n",
        "о": "o", "п": "p", "р": "r", "с": "s", "т": "t",
        "у": "u", "ф": "f", "х": "kh", "ц": "ts", "ч": "ch",
        "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "",
        "э": "e", "ю": "yu", "я": "ya",
    }

    out = []
    for ch in text:
        lower = ch.lower()
        if lower in mapping:
            lat = mapping[lower]
            if ch.isupper() and lat:
                lat = lat[0].upper() + lat[1:]
            out.append(lat)
        else:
            out.append(ch)
    return "".join(out)


def lat_to_ru(text: str) -> str:
    # greedy longest-match decoder
    seq_map = [
        ("shch", "щ"),
        ("sch", "щ"),
        ("yo", "ё"),
        ("yu", "ю"),
        ("ya", "я"),
        ("kh", "х"),
        ("ts", "ц"),
        ("ch", "ч"),
        ("sh", "ш"),
        ("zh", "ж"),
    ]
    single_map = {
        "a": "а", "b": "б", "v": "в", "g": "г", "d": "д",
        "e": "е", "z": "з", "i": "и", "y": "ы",
        "k": "к", "l": "л", "m": "м", "n": "н",
        "o": "о", "p": "п", "r": "р", "s": "с", "t": "т",
        "u": "у", "f": "ф",
    }

    i = 0
    res = []
    while i < len(text):
        chunk = text[i:]
        matched = False
        for seq, ru in seq_map:
            if chunk.lower().startswith(seq):
                is_upper = chunk[0].isupper()
                res.append(ru.upper() if is_upper else ru)
                i += len(seq)
                matched = True
                break
        if matched:
            continue
        ch = text[i]
        lower = ch.lower()
        if lower in single_map:
            ru = single_map[lower]
            res.append(ru.upper() if ch.isupper() else ru)
        else:
            res.append(ch)
        i += 1
    return "".join(res)
