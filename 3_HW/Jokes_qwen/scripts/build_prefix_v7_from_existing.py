import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path


BAD_RE = re.compile(
    r"\b(бля\w*|бляд\w*|[её]б\w+|пизд\w*|ху[йеёяи]\w*|сука|суч\w*|"
    r"говн\w*|мудак\w*|мудил\w*|дроч\w*|жоп\w*|залуп\w*|шлюх\w*|"
    r"проститут\w*|трах\w*|секс\w*)\b",
    re.IGNORECASE,
)


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_prefixes(path: Path) -> list[str]:
    prefixes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*\d+[\.)-]*\s*", "", line)
        line = normalize_spaces(line)
        if line:
            prefixes.append(line)
    seen = set()
    unique = []
    for p in prefixes:
        key = p.lower().replace("ё", "е")
        if key not in seen:
            unique.append(p)
            seen.add(key)
    return unique


def normalize(text: str) -> str:
    return normalize_spaces(text).lower().replace("ё", "е")


def cyr_ratio(text: str) -> float:
    if not text:
        return 0.0
    total = len(text)
    cyr = len(re.findall(r"[А-Яа-яЁё]", text))
    return cyr / max(total, 1)


def latin_ratio(text: str) -> float:
    if not text:
        return 0.0
    total = len(text)
    latin = len(re.findall(r"[A-Za-z]", text))
    return latin / max(total, 1)


def sentence_count(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", text) if s])


def truncate_sentences(text: str, max_sent: int) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(parts[:max_sent]).strip()


def clean_response(text: str, max_sent: int) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if not text.startswith("..."):
        text = "... " + text.lstrip(" .,-")
    if text and text[-1] not in ".!?":
        text += "."
    text = truncate_sentences(text, max_sent)
    return text


def iter_jsonl(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        yield json.loads(line)


def main():
    ap = argparse.ArgumentParser(description="Build prefix dataset from existing JSONL files")
    ap.add_argument("--prefixes", default=str(Path("..") / "prefixes.txt"))
    ap.add_argument("--v4", default=str(Path("data") / "sft_prefix_v4.jsonl"))
    ap.add_argument("--v5", default=str(Path("data") / "sft_prefix_v5.jsonl"))
    ap.add_argument("--out", default=str(Path("data") / "sft_prefix_v7.jsonl"))
    ap.add_argument("--max-sent", type=int, default=3)
    ap.add_argument("--min-sent", type=int, default=1)
    ap.add_argument("--min-chars", type=int, default=60)
    ap.add_argument("--max-chars", type=int, default=260)
    ap.add_argument("--min-cyr", type=float, default=0.6)
    ap.add_argument("--max-latin", type=float, default=0.05)
    ap.add_argument("--max-per-prefix", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    prefixes = clean_prefixes(Path(args.prefixes))
    norm_map = {normalize(p): p for p in prefixes}

    records = []
    for obj in iter_jsonl(Path(args.v4)):
        prompt = normalize_spaces(obj.get("prompt", ""))
        response = obj.get("response", "")
        if not prompt:
            continue
        key = normalize(prompt)
        if key not in norm_map:
            continue
        records.append((norm_map[key], response))

    for obj in iter_jsonl(Path(args.v5)):
        prompt = normalize_spaces(obj.get("prompt", ""))
        response = obj.get("response", "")
        if not prompt:
            continue
        key = normalize(prompt)
        if key not in norm_map:
            continue
        records.append((norm_map[key], response))

    filtered = []
    seen = set()
    for prefix, response in records:
        response = clean_response(response, args.max_sent)
        if len(response) < args.min_chars or len(response) > args.max_chars:
            continue
        if BAD_RE.search(response):
            continue
        if re.search(r"\d", response):
            continue
        if latin_ratio(response) > args.max_latin:
            continue
        if cyr_ratio(response) < args.min_cyr:
            continue
        if sentence_count(response) < args.min_sent:
            continue
        key = (normalize(prefix), normalize(response))
        if key in seen:
            continue
        seen.add(key)
        filtered.append((prefix, response))

    by_prefix = defaultdict(list)
    for prefix, response in filtered:
        by_prefix[prefix].append(response)

    prompt_template = (
        "Продолжи анекдот. Затравка: \"{prefix}\". "
        "Ответ начни с \"...\". 2-3 предложения, завершай мысль."
    )

    final = []
    for prefix in prefixes:
        responses = by_prefix.get(prefix, [])
        if not responses:
            continue
        if args.max_per_prefix and len(responses) > args.max_per_prefix:
            responses = random.sample(responses, args.max_per_prefix)
        for response in responses:
            final.append({"prompt": prompt_template.format(prefix=prefix), "response": response})

    random.shuffle(final)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in final:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    counts = {k: len(v) for k, v in by_prefix.items()}
    print(f"[done] prefixes={len(prefixes)} rows={len(final)}")
    print("[counts]", sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10])


if __name__ == "__main__":
    main()
