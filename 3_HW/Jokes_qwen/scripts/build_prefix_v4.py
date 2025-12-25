import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset


def normalize_spaces(text: str) -> str:
    text = text.replace("\u200b", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_prefix_line(line: str) -> str:
    line = line.strip()
    if not line:
        return ""
    # drop leading numbering like "12. " or "12) "
    line = re.sub(r"^\s*\d+[\.)-]*\s*", "", line)
    return normalize_spaces(line)


def normalize_for_match(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("ё", "е")
    text = text.replace("—", "-").replace("–", "-")
    text = normalize_spaces(text)
    return text


def build_prefix_patterns(prefixes):
    patterns = []
    for prefix in prefixes:
        parts = []
        for ch in prefix:
            if ch in ("е", "ё", "Е", "Ё"):
                parts.append("[её]")
            elif ch.isspace():
                parts.append(r"\s+")
            else:
                parts.append(re.escape(ch))
        pattern = r"^" + "".join(parts)
        patterns.append((prefix, re.compile(pattern, re.IGNORECASE)))
    # longest first to avoid partial matches
    patterns.sort(key=lambda x: len(x[0]), reverse=True)
    return patterns


def cyr_ratio(text: str) -> float:
    if not text:
        return 0.0
    total = len(text)
    cyr = len(re.findall(r"[А-Яа-яЁё]", text))
    return cyr / max(total, 1)


def contains_bad_words(text: str) -> bool:
    bad = [
        "хуй", "пизд", "еба", "ёба", "ебл", "ебан", "ебуч", "бляд", "сука",
        "говн", "муд", "пидор", "пидр", "жоп", "хер", "залуп", "дроч",
        "мля", "бля",
    ]
    low = text.lower()
    return any(b in low for b in bad)


def extract_keywords(prefix: str):
    stop = {
        "и", "в", "на", "по", "у", "к", "о", "об", "от", "до", "за", "над",
        "под", "при", "как", "что", "это", "эти", "этот", "кто", "когда",
        "уже", "еще", "ещё", "или", "а", "но", "да", "нет", "же", "ли",
        "то", "тут", "там", "где", "куда", "откуда", "про", "после", "перед",
        "-", "—",
    }
    verb_stop = {
        "идет", "идёт", "приходит", "приходят", "встречаются", "сидят",
        "сидит", "заходит", "спрашивает", "доказывает", "пишет", "решает",
        "вышел", "вышла", "вышло",
    }
    norm = normalize_for_match(prefix)
    words = re.split(r"[^а-яёa-z0-9]+", norm)
    words = [w for w in words if w]
    keywords = [
        w for w in words
        if len(w) >= 4 and w not in stop and w not in verb_stop
    ]
    if not keywords:
        keywords = [w for w in words if len(w) >= 4]
    return keywords


def truncate_sentences(text: str, max_sent: int) -> str:
    parts = re.split(r"([.!?])", text)
    res = []
    for i in range(0, len(parts) - 1, 2):
        sent = (parts[i] + parts[i + 1]).strip()
        if sent:
            res.append(sent)
        if len(res) >= max_sent:
            break
    return " ".join(res) if res else text.strip()


def clean_joke(text: str) -> str:
    text = text.replace("\r", "\n")
    # remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    text = normalize_spaces(text)
    # drop leading numbering or bullets
    text = re.sub(r"^\s*[-*•]+\s*", "", text)
    text = re.sub(r"^\s*\d+[\.)-]*\s*", "", text)
    # drop leading quotes/dashes
    text = re.sub(r"^[\"'«»„“”\-–—]+\s*", "", text)
    text = re.sub(r"\.\s*\.\s*\.", "...", text)
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    text = text.strip(" \"'«»")
    return normalize_spaces(text)


def auto_prefix(text: str, min_words: int = 5, max_words: int = 7):
    words = text.split()
    if len(words) < min_words + 2:
        return None, None
    n = min(max_words, len(words) - 2)
    if n < min_words:
        n = min_words
    prefix = " ".join(words[:n]).strip(" ,:;!-")
    if len(prefix) > 60 and n > min_words:
        n = min_words
        prefix = " ".join(words[:n]).strip(" ,:;!-")
    if not re.match(r"[A-ZА-ЯЁ]", prefix):
        return None, None
    cont = " ".join(words[n:]).lstrip(" ,:;—-")
    if not prefix or not cont:
        return None, None
    return prefix, cont


def iter_local_files(paths):
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line:
                yield line


def iter_hf_anekdots():
    ds = load_dataset("igorktech/anekdots", split="train", streaming=True)
    for row in ds:
        text = row.get("text") or ""
        if text:
            yield str(text)


def iter_hf_dialogs():
    ds = load_dataset("igorktech/anekdots_dialogs", split="train", streaming=True)
    for row in ds:
        text = row.get("original") or ""
        if text:
            yield str(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefixes", default=str(Path("..") / "prefixes.txt"))
    parser.add_argument("--out-prefixes", default=str(Path("data") / "prefixes_clean.txt"))
    parser.add_argument("--out-matched", default=str(Path("data") / "jokes_prefix_matched.txt"))
    parser.add_argument("--out-jsonl", default=str(Path("data") / "sft_prefix_v4.jsonl"))
    parser.add_argument("--min-len", type=int, default=40)
    parser.add_argument("--max-len", type=int, default=350)
    parser.add_argument("--max-sent", type=int, default=3)
    parser.add_argument("--max-auto", type=int, default=60000)
    parser.add_argument("--min-per-prefix", type=int, default=25)
    parser.add_argument("--matched-multiplier", type=int, default=5)
    parser.add_argument("--allow-profanity", action="store_true")
    args = parser.parse_args()

    random.seed(42)

    prefix_path = Path(args.prefixes)
    prefixes_raw = [clean_prefix_line(l) for l in prefix_path.read_text(encoding="utf-8").splitlines()]
    prefixes_raw = [p for p in prefixes_raw if p]
    # dedup
    seen = set()
    prefixes = []
    for p in prefixes_raw:
        key = normalize_for_match(p)
        if key in seen:
            continue
        seen.add(key)
        prefixes.append(p)

    out_prefix_path = Path(args.out_prefixes)
    out_prefix_path.parent.mkdir(parents=True, exist_ok=True)
    out_prefix_path.write_text("\n".join(prefixes) + "\n", encoding="utf-8")

    patterns = build_prefix_patterns(prefixes)

    local_paths = [
        Path("data") / "jokes_clean_strict.txt",
        Path("data") / "jokes_clean.txt",
        Path("data") / "jokes_expanded.txt",
        Path("data") / "jokes_raw.txt",
    ]

    def joke_iter():
        for line in iter_local_files(local_paths):
            yield line
        for line in iter_hf_anekdots():
            yield line
        for line in iter_hf_dialogs():
            yield line

    matched_records = []
    auto_records = []
    matched_texts = []
    seen_full = set()
    prefix_counts = Counter()

    for raw in joke_iter():
        text = clean_joke(raw)
        if not text:
            continue
        if len(text) < args.min_len or len(text) > args.max_len:
            continue
        if cyr_ratio(text) < 0.75:
            continue
        if not args.allow_profanity and contains_bad_words(text):
            continue
        if re.search(r"\b\w+\.(ru|com|net|ua|org|info)\b", text, re.IGNORECASE):
            continue
        text = truncate_sentences(text, args.max_sent)
        if not text:
            continue
        if not re.search(r"[.!?]", text):
            continue
        if text[-1] not in ".!?":
            text = text + "."
        norm_full = normalize_for_match(text)
        if norm_full in seen_full:
            continue
        seen_full.add(norm_full)

        matched = False
        for prefix, pattern in patterns:
            m = pattern.match(text)
            if not m:
                continue
            cont = text[m.end():].lstrip(" ,:;—-").strip()
            if len(cont) < 20:
                break
            if cont[-1] not in ".!?":
                cont = cont + "."
            prompt = prefix
            response = f"... {cont}"
            matched_records.append({"prompt": prompt, "response": response})
            matched_texts.append(f"{prefix} {response}".strip())
            prefix_counts[prefix] += 1
            matched = True
            break
        if matched:
            continue

        prefix, cont = auto_prefix(text)
        if not prefix or not cont:
            continue
        if re.search(r"\d", prefix):
            continue
        if len(cont) < 20:
            continue
        if cont[-1] not in ".!?":
            cont = cont + "."
        prompt = prefix
        response = f"... {cont}"
        auto_records.append({"prompt": prompt, "response": response})
        if len(auto_records) >= args.max_auto:
            break

    # ensure minimal coverage for all prefixes
    if matched_records:
        auto_responses = [r["response"] for r in auto_records]
        auto_norm = [normalize_for_match(r) for r in auto_responses]
        generic_pool = auto_responses or [r["response"] for r in matched_records]
        for prefix in prefixes:
            need = max(0, args.min_per_prefix - prefix_counts[prefix])
            if need == 0:
                continue
            keywords = extract_keywords(prefix)
            if keywords:
                candidates = [
                    auto_responses[i]
                    for i, norm in enumerate(auto_norm)
                    if any(k in norm for k in keywords)
                ]
            else:
                candidates = []
            pool = candidates or generic_pool
            for _ in range(need):
                cont = random.choice(pool)
                prompt = prefix
                response = f"... {cont}"
                matched_records.append({"prompt": prompt, "response": response})
                matched_texts.append(f"{prefix} {response}".strip())
                prefix_counts[prefix] += 1

    # final shuffle
    matched_boosted = matched_records * max(1, args.matched_multiplier)
    records = matched_boosted + auto_records
    random.shuffle(records)

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )

    out_matched = Path(args.out_matched)
    out_matched.write_text("\n".join(matched_texts) + "\n", encoding="utf-8")

    print(f"Prefixes: {len(prefixes)}")
    print(f"Matched records: {len(matched_records)}")
    print(f"Auto records: {len(auto_records)}")
    print(f"Total records: {len(records)}")
    print("Top prefixes:")
    for prefix, cnt in prefix_counts.most_common(10):
        print(f"  {cnt:5d}  {prefix}")


if __name__ == "__main__":
    main()
