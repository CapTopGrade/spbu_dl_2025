import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset


BAD_WORDS = [
    "хуй",
    "пизд",
    "еба",
    "ёба",
    "ебл",
    "ебан",
    "ебуч",
    "бляд",
    "сука",
    "говн",
    "муд",
    "пидор",
    "пидр",
    "жоп",
    "хер",
    "залуп",
    "дроч",
    "мля",
    "бля",
]


def normalize_spaces(text: str) -> str:
    text = text.replace("\u200b", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_prefix_line(line: str) -> str:
    line = line.strip()
    if not line:
        return ""
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
    patterns.sort(key=lambda x: len(x[0]), reverse=True)
    return patterns


def cyr_ratio(text: str) -> float:
    if not text:
        return 0.0
    total = len(text)
    cyr = len(re.findall(r"[А-Яа-яЁё]", text))
    return cyr / max(total, 1)


def contains_bad_words(text: str) -> bool:
    low = text.lower()
    return any(b in low for b in BAD_WORDS)


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
    text = re.sub(r"<[^>]+>", " ", text)
    text = normalize_spaces(text)
    text = re.sub(r"^\s*[-*•]+\s*", "", text)
    text = re.sub(r"^\s*\d+[\.)-]*\s*", "", text)
    text = re.sub(r"^[\"'«»„“”\-–—]+\s*", "", text)
    text = re.sub(r"\.\s*\.\s*\.", "...", text)
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    text = text.strip(" \"'«»")
    return normalize_spaces(text)


def extract_keywords(prefix: str):
    stop = {
        "и",
        "в",
        "на",
        "по",
        "у",
        "к",
        "о",
        "об",
        "от",
        "до",
        "за",
        "над",
        "под",
        "при",
        "как",
        "что",
        "это",
        "эти",
        "этот",
        "кто",
        "когда",
        "уже",
        "еще",
        "ещё",
        "или",
        "а",
        "но",
        "да",
        "нет",
        "же",
        "ли",
        "то",
        "тут",
        "там",
        "где",
        "куда",
        "откуда",
        "про",
        "после",
        "перед",
        "-",
        "—",
    }
    verb_stop = {
        "идет",
        "идёт",
        "приходит",
        "приходят",
        "встречаются",
        "сидят",
        "сидит",
        "заходит",
        "спрашивает",
        "доказывает",
        "пишет",
        "решает",
        "вышел",
        "вышла",
        "вышло",
    }
    norm = normalize_for_match(prefix)
    words = re.split(r"[^а-яёa-z0-9]+", norm)
    words = [w for w in words if w]
    keywords = [w for w in words if len(w) >= 4 and w not in stop and w not in verb_stop]
    if not keywords:
        keywords = [w for w in words if len(w) >= 4]
    return keywords


def main():
    ap = argparse.ArgumentParser(description="Build prefix-focused SFT dataset from jokes")
    ap.add_argument("--prefixes", default=str(Path("..") / "prefixes.txt"))
    ap.add_argument("--out-jsonl", default=str(Path("data") / "sft_prefix_v5.jsonl"))
    ap.add_argument("--out-prefixes", default=str(Path("data") / "prefixes_clean.txt"))
    ap.add_argument("--target-per-prefix", type=int, default=300)
    ap.add_argument("--max-sent", type=int, default=3)
    ap.add_argument("--min-len", type=int, default=40)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    prefix_path = Path(args.prefixes)
    prefixes_raw = [clean_prefix_line(l) for l in prefix_path.read_text(encoding="utf-8").splitlines()]
    prefixes_raw = [p for p in prefixes_raw if p]
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
    keywords_map = {p: extract_keywords(p) for p in prefixes}
    keyword_to_prefix = defaultdict(list)
    for pref, kws in keywords_map.items():
        for kw in kws:
            keyword_to_prefix[kw].append(pref)
    keyword_list = sorted(keyword_to_prefix.keys())
    keyword_re = re.compile(r"\b(" + "|".join(map(re.escape, keyword_list)) + r")\b", re.IGNORECASE)

    matched = defaultdict(list)
    keyword_pool = defaultdict(list)
    general_pool = []
    seen_texts = set()

    ds = load_dataset("igorktech/anekdots", split="train", streaming=True)
    for row in ds:
        text = row.get("text") or ""
        if not text:
            continue
        text = clean_joke(str(text))
        if len(text) < args.min_len or len(text) > args.max_len:
            continue
        if cyr_ratio(text) < 0.75:
            continue
        if contains_bad_words(text):
            continue
        if re.search(r"\b\w+\.(ru|com|net|ua|org|info)\b", text, re.IGNORECASE):
            continue
        text = truncate_sentences(text, args.max_sent)
        if not text or text[-1] not in ".!?":
            continue
        norm = normalize_for_match(text)
        if norm in seen_texts:
            continue
        seen_texts.add(norm)

        general_pool.append(text)

        for prefix, pattern in patterns:
            m = pattern.match(text)
            if not m:
                continue
            cont = text[m.end():].lstrip(" ,:;—-").strip()
            if len(cont) < 20:
                continue
            matched[prefix].append(cont)
            break

        for kw in keyword_re.findall(norm):
            for pref in keyword_to_prefix.get(kw.lower(), []):
                keyword_pool[pref].append(text)

        # stop early if we have enough for all prefixes
        ready = True
        for pref in prefixes:
            if len(matched[pref]) + len(keyword_pool[pref]) < args.target_per_prefix:
                ready = False
                break
        if ready and len(general_pool) > 50000:
            break

    records = []
    prefix_counts = Counter()
    for pref in prefixes:
        examples = []
        # use matched continuations first
        for cont in matched[pref]:
            examples.append(("... " + cont).strip())
        # then keyword-based jokes
        for text in keyword_pool[pref]:
            if len(examples) >= args.target_per_prefix:
                break
            examples.append("... " + text)
        # fill from general pool
        while len(examples) < args.target_per_prefix and general_pool:
            examples.append("... " + random.choice(general_pool))

        for resp in examples[: args.target_per_prefix]:
            records.append({"prompt": pref, "response": resp})
            prefix_counts[pref] += 1

    random.shuffle(records)
    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )

    print(f"Prefixes: {len(prefixes)}")
    print(f"Records: {len(records)}")
    print("Top prefixes:")
    for pref, cnt in prefix_counts.most_common(10):
        print(f"  {cnt:5d}  {pref}")


if __name__ == "__main__":
    main()
