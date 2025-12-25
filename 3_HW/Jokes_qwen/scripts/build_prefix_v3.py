import json
import os
import random
import re
from collections import OrderedDict
from pathlib import Path

from datasets import load_dataset


def load_prefixes(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # drop leading numbering like "12. " or "12 "
        line = re.sub(r"^[\\s\\d\\.:-]+", " ", line).strip()
        cleaned.append(line)
    return cleaned


def truncate_sentences(text: str, max_sent: int = 3) -> str:
    parts = re.split(r"([.!?])", text)
    res = []
    for i in range(0, len(parts) - 1, 2):
        sent = (parts[i] + parts[i + 1]).strip()
        if sent:
            res.append(sent)
        if len(res) >= max_sent:
            break
    return " ".join(res) if res else text.strip()


def normalize(text: str) -> str:
    text = text.replace("\\u200b", " ")
    text = re.sub(r"\\s+", " ", text)
    text = re.sub(r"\\d+", "", text)
    text = text.strip(" .,!?:;\\-\\t\\n\\r")
    return text.strip()


def cyr_ratio(text: str) -> float:
    if not text:
        return 0.0
    total = len(text)
    cyr = len(re.findall(r"[А-Яа-яЁё]", text))
    return cyr / max(total, 1)


def collect_local(path: Path):
    jokes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = normalize(line)
        if line:
            jokes.append(line)
    return jokes


def collect_hf_dataset(hf_token: str | None):
    # dataset contains field "joke"
    ds = load_dataset("igorktech/anekdots", split="train", token=hf_token)
    jokes = []
    for row in ds:
        text = row.get("joke") or ""
        text = normalize(str(text))
        if text:
            jokes.append(text)
    return jokes


def filter_jokes(jokes):
    uniq = OrderedDict()
    for joke in jokes:
        if len(joke) < 40 or len(joke) > 220:
            continue
        if cyr_ratio(joke) < 0.7:
            continue
        if re.search(r"https?://", joke):
            continue
        # drop obvious list markers
        joke = re.sub(r"^[-*\\d\\)\\.]+\\s*", "", joke).strip()
        # collapse spaces once more
        joke = re.sub(r"\\s+", " ", joke).strip()
        if not joke:
            continue
        uniq[joke] = True
    return list(uniq.keys())


def build_records(jokes, prefixes):
    random.seed(123)
    records = []
    for joke in jokes:
        joke_cut = truncate_sentences(joke, max_sent=3)
        prefix = random.choice(prefixes)
        response = joke_cut
        if not response.lower().startswith(prefix.lower()):
            response = f"{prefix} {response}"
        prompt = (
            f"Начни анекдот с фразы: \"{prefix}\". "
            "Напиши смешной, логичный анекдот на русском языке без оскорблений и цифр. "
            "Сделай 2-3 предложения, завершай мысль."
        )
        records.append({"prompt": prompt, "response": response})
    random.shuffle(records)
    return records


def main():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    prefixes = load_prefixes(data_dir / "prefixes.txt")
    base_local = collect_local(data_dir / "jokes_clean_strict.txt")
    extra = collect_hf_dataset(hf_token)

    all_jokes = filter_jokes(base_local + extra)

    expanded_path = data_dir / "jokes_expanded.txt"
    expanded_path.write_text("\n".join(all_jokes), encoding="utf-8")

    records = build_records(all_jokes, prefixes)
    out_jsonl = data_dir / "sft_prefix_v3.jsonl"
    out_jsonl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")

    print(f"Prefixes: {len(prefixes)}")
    print(f"Collected jokes: {len(all_jokes)}")
    print(f"SFT records: {len(records)}")
    print(f"Saved expanded jokes to: {expanded_path}")
    print(f"Saved SFT jsonl to: {out_jsonl}")


if __name__ == "__main__":
    main()
