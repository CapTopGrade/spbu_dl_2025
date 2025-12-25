import json
from pathlib import Path

from translit_utils import ru_to_lat


def main():
    src = Path("data/sft_prefix_v3.jsonl")
    dst = Path("data/sft_prefix_v3_translit.jsonl")
    prefixes_ru = Path("data/prefixes.txt").read_text(encoding="utf-8").splitlines()
    prefixes_lat = [ru_to_lat(line.split(maxsplit=1)[-1]) if line.strip() else "" for line in prefixes_ru]
    Path("data/prefixes_translit.txt").write_text("\n".join(prefixes_lat), encoding="utf-8")

    out_lines = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        rec["prompt"] = ru_to_lat(rec["prompt"])
        rec["response"] = ru_to_lat(rec["response"])
        out_lines.append(json.dumps(rec, ensure_ascii=False))

    dst.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"converted {len(out_lines)} records to {dst}")
    print("prefixes transliterated to data/prefixes_translit.txt")


if __name__ == "__main__":
    main()
