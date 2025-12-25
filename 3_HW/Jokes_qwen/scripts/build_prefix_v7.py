import argparse
import json
import random
import re
from collections import defaultdict
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
    line = re.sub(r"^\s*\d+[\.)-]*\s*", "", line)
    return normalize_spaces(line)


def clean_joke(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"<[^>]+>", " ", text)
    text = normalize_spaces(text)
    text = re.sub(r"^\s*[-*\u2022]+\s*", "", text)
    text = re.sub(r"^\s*\d+[\.)-]*\s*", "", text)
    text = text.strip(" \"'\u00ab\u00bb")
    text = re.sub(r"\.\s*\.\s*\.", "...", text)
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    return normalize_spaces(text)


def normalize_for_match(text: str) -> str:
    text = normalize_spaces(text).lower()
    return text.replace("\u0451", "\u0435")


def cyr_ratio(text: str) -> float:
    if not text:
        return 0.0
    total = len(text)
    cyr = len(re.findall(r"[\u0410-\u042f\u0430-\u044f\u0401\u0451]", text))
    return cyr / max(total, 1)


def sentence_count(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", text) if s])


def truncate_sentences(text: str, max_sent: int) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(parts[:max_sent]).strip()


def main():
    ap = argparse.ArgumentParser(description="Build prefix-focused SFT dataset (instruction style)")
    ap.add_argument("--prefixes", default=str(Path("..") / "prefixes.txt"))
    ap.add_argument("--out-jsonl", default=str(Path("data") / "sft_prefix_v7.jsonl"))
    ap.add_argument("--out-prefixes", default=str(Path("data") / "prefixes_clean.txt"))
    ap.add_argument("--max-sent", type=int, default=3)
    ap.add_argument("--min-len", type=int, default=50)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--min-cont-len", type=int, default=40)
    ap.add_argument("--min-sent", type=int, default=1)
    ap.add_argument("--min-cyr", type=float, default=0.5)
    ap.add_argument("--max-latin", type=float, default=0.02)
    ap.add_argument("--max-per-prefix", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    prefix_path = Path(args.prefixes)
    prefixes_raw = [clean_prefix_line(l) for l in prefix_path.read_text(encoding="utf-8").splitlines()]
    prefixes = [p for p in prefixes_raw if p]
    seen = set()
    unique_prefixes = []
    for p in prefixes:
        key = p.lower()
        if key not in seen:
            unique_prefixes.append(p)
            seen.add(key)

    out_prefixes = Path(args.out_prefixes)
    out_prefixes.parent.mkdir(parents=True, exist_ok=True)
    out_prefixes.write_text("\n".join(unique_prefixes) + "\n", encoding="utf-8")

    prefixes_norm = {p: normalize_for_match(p) for p in unique_prefixes}

    data = load_dataset("igorktech/anekdots", split="train")

    by_prefix = defaultdict(list)
    for row in data:
        raw = row.get("text") or row.get("joke") or row.get("content")
        if not raw:
            continue
        text = clean_joke(str(raw))
        if not text:
            continue
        if len(text) < args.min_len or len(text) > args.max_len:
            continue
        latin = len(re.findall(r"[A-Za-z]", text))
        if latin / max(len(text), 1) > args.max_latin:
            continue
        if cyr_ratio(text) < args.min_cyr:
            continue

        norm_text = normalize_for_match(text)
        for prefix, norm_prefix in prefixes_norm.items():
            if not norm_text.startswith(norm_prefix):
                continue
            rest = text[len(prefix) :].strip()
            rest = rest.lstrip(" ,:;-\u2014")
            if len(rest) < args.min_cont_len:
                continue
            rest = truncate_sentences(rest, args.max_sent)
            if sentence_count(rest) < args.min_sent:
                continue
            response = "... " + rest
            by_prefix[prefix].append(response)
            break

    prompt_template = (
        "\u041f\u0440\u043e\u0434\u043e\u043b\u0436\u0438 \u0430\u043d\u0435\u043a\u0434\u043e\u0442. "
        "\u0417\u0430\u0442\u0440\u0430\u0432\u043a\u0430: \"{prefix}\". "
        "\u041e\u0442\u0432\u0435\u0442 \u043d\u0430\u0447\u043d\u0438 \u0441 \"...\". "
        "2-3 \u043f\u0440\u0435\u0434\u043b\u043e\u0436\u0435\u043d\u0438\u044f, \u0437\u0430\u0432\u0435\u0440\u0448\u0430\u0439 \u043c\u044b\u0441\u043b\u044c."
    )

    records = []
    for prefix in unique_prefixes:
        candidates = by_prefix.get(prefix, [])
        if not candidates:
            continue
        if args.max_per_prefix and len(candidates) > args.max_per_prefix:
            candidates = random.sample(candidates, args.max_per_prefix)
        for response in candidates:
            prompt = prompt_template.format(prefix=prefix)
            records.append({"prompt": prompt, "response": response})

    random.shuffle(records)

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total = len(records)
    prefix_counts = {k: len(v) for k, v in by_prefix.items()}
    print(f"[done] prefixes={len(unique_prefixes)} rows={total}")
    print("[counts]", sorted(prefix_counts.items(), key=lambda x: x[1], reverse=True)[:10])


if __name__ == "__main__":
    main()
