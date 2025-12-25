import argparse
import random
import re
from pathlib import Path


BAD_WORDS = [
    "бля",
    "бляд",
    "пизд",
    "хуй",
    "хуё",
    "хуя",
    "сука",
    "суч",
    "говн",
    "мудак",
    "мудил",
    "дроч",
    "жоп",
    "залуп",
    "шлюх",
    "проститут",
    "трах",
    "секс",
]

CLOSERS = [
    "Вот и вся история.",
    "Так и вышло.",
    "С тех пор он туда не ходит.",
    "С тех пор все стало ясно.",
    "Вот такой поворот.",
]

FALLBACKS = [
    "... Потом один говорит: \"Ну и дела\". Другой отвечает: \"Вот так и живем\".",
    "... А в конце все только переглянулись и рассмеялись.",
    "... И тут все стало на свои места. Такой вот поворот.",
    "... С тех пор он всем это рассказывает и каждый раз улыбается.",
    "... И только бармен тихо усмехнулся: \"Классика\".",
]

LOOKALIKES = str.maketrans(
    {
        "A": "А",
        "B": "В",
        "C": "С",
        "E": "Е",
        "H": "Н",
        "K": "К",
        "M": "М",
        "O": "О",
        "P": "Р",
        "T": "Т",
        "X": "Х",
        "Y": "У",
        "a": "а",
        "c": "с",
        "e": "е",
        "h": "н",
        "k": "к",
        "m": "м",
        "o": "о",
        "p": "р",
        "t": "т",
        "x": "х",
        "y": "у",
    }
)


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_prefix(line: str) -> tuple[str, str]:
    if "..." in line:
        prefix, rest = line.split("...", 1)
        return prefix.strip(), "... " + rest.strip()
    return line.strip(), ""


def has_bad_words(text: str) -> bool:
    low = text.lower()
    return any(b in low for b in BAD_WORDS)


def strip_qa(text: str) -> str:
    cut = re.split(r"\b(?:вопрос|ответ)\b\s*[:\-—]", text, flags=re.IGNORECASE)
    return cut[0].strip()


def normalize_punct(text: str) -> str:
    text = re.sub(r"[!?]{2,}", lambda m: m.group(0)[0], text)
    text = re.sub(r"\.\s*\.\s*\.", "...", text)
    text = re.sub(r"\.\s*\.", ".", text)
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    return normalize_spaces(text)


def clean_response(text: str, min_sent: int, max_sent: int) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    text = text.translate(LOOKALIKES)
    text = strip_qa(text)
    text = normalize_punct(text)

    if not text.startswith("..."):
        text = "... " + text.lstrip(" .,-")

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    sentences = [s for s in sentences if not has_bad_words(s)]
    sentences = [s for s in sentences if not re.search(r"[A-Za-z]", s)]
    sentences = [s for s in sentences if not re.search(r"\d", s)]

    if len(sentences) > max_sent:
        sentences = sentences[:max_sent]

    while len(sentences) < min_sent and len(sentences) < max_sent:
        sentences.append(random.choice(CLOSERS))

    if not sentences:
        sentences = ["... " + random.choice(CLOSERS)]

    result = " ".join(sentences).strip()
    if not result.startswith("..."):
        result = "... " + result.lstrip(" .,-")
    if result[-1] not in ".!?":
        result += "."
    return result


def passes_quality(text: str) -> bool:
    if re.search(r"[A-Za-z]", text):
        return False
    if re.search(r"\d", text):
        return False
    if has_bad_words(text):
        return False
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text) if s]
    if not (1 <= len(sents) <= 3):
        return False
    if len(text) < 60:
        return False
    return True


def fallback_response(prefix: str) -> str:
    idx = abs(hash(prefix)) % len(FALLBACKS)
    return FALLBACKS[idx]


def main():
    ap = argparse.ArgumentParser(description="Post-process generated jokes")
    ap.add_argument("--inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min_sent", type=int, default=2)
    ap.add_argument("--max_sent", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    lines = args.inp.read_text(encoding="utf-8").splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        prefix, response = split_prefix(line)
        response = clean_response(response, args.min_sent, args.max_sent)
        if not passes_quality(response):
            response = fallback_response(prefix)
        cleaned.append(f"{prefix} {response}".strip())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(cleaned) + "\n", encoding="utf-8")
    print(f"[done] saved {len(cleaned)} lines to {args.out}")


if __name__ == "__main__":
    main()
