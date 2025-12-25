import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Rewrite prefix dataset with explicit instruction prompts")
    ap.add_argument("--inp", type=Path, default=Path("data/sft_prefix_v5.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/sft_prefix_v6.jsonl"))
    args = ap.parse_args()

    records = []
    with args.inp.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            prefix = obj["prompt"].strip()
            prompt = (
                "Продолжи анекдот. Затравка: "
                f"\"{prefix}\". "
                "Ответ начни с \"...\". 2-3 предложения, завершай мысль."
            )
            records.append({"prompt": prompt, "response": obj["response"]})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    print(f"Records: {len(records)}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
