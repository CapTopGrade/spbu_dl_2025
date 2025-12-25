import argparse
import json
import random
import re
from pathlib import Path

# Промпты немного варьируются, чтобы модель не залипала на одной формулировке
PROMPTS = [
    "Придумай короткий смешной анекдот на русском языке.",
    "Расскажи забавную шутку или анекдот по-русски.",
    "Сочини смешной анекдот.",
    "Напиши свежий анекдот на русском языке.",
    "Сделай остроумный анекдот, который заставит улыбнуться.",
    "Поделись добрым анекдотом без оскорблений.",
    "Придумай смешную историю в формате анекдота.",
    "Создай анекдот с неожиданной развязкой.",
    "Расскажи новый анекдот, подходящий для широкой аудитории.",
    "Напиши анекдот, избегая цитат из известных произведений.",
]


def load_jokes(path: Path, min_len: int, max_len: int) -> list[str]:
    jokes = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            joke = line.strip()
            if not joke or joke in seen:
                continue
            if len(joke) < min_len or len(joke) > max_len:
                continue
            seen.add(joke)
            jokes.append(joke)
    random.shuffle(jokes)
    return jokes


def load_prefixes(path: Path) -> list[str]:
    prefixes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cleaned = re.sub(r"^\s*\d+\s+", "", line).strip()
        if cleaned:
            prefixes.append(cleaned)
    return prefixes


def main():
    parser = argparse.ArgumentParser(description="Build SFT dataset from scraped jokes")
    parser.add_argument("--input", type=Path, default=Path("data/jokes_raw.txt"), help="Raw jokes file")
    parser.add_argument("--out", type=Path, default=Path("data/sft_dataset.jsonl"), help="Output JSONL with prompt/response")
    parser.add_argument("--min_len", type=int, default=32, help="Drop jokes shorter than this")
    parser.add_argument("--max_len", type=int, default=400, help="Drop jokes longer than this")
    parser.add_argument("--prefixes", type=Path, help="Optional path to prefixes; responses will start with them")
    args = parser.parse_args()

    jokes = load_jokes(args.input, args.min_len, args.max_len)
    prefixes = load_prefixes(args.prefixes) if args.prefixes else None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for joke in jokes:
            if prefixes:
                prefix = random.choice(prefixes)
                prompt = (
                    f"Начни анекдот с фразы: \"{prefix}\". "
                    "Напиши смешной, логичный анекдот на русском языке без оскорблений."
                )
                if joke.lower().startswith(prefix.lower()):
                    response = joke
                else:
                    response = f"{prefix} {joke}"
            else:
                prompt = random.choice(PROMPTS)
                response = joke

            record = {
                "prompt": prompt,
                "response": response,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[done] wrote {len(jokes)} examples to {args.out}")


if __name__ == "__main__":
    main()
