import re
from pathlib import Path


def load_prefixes(path: Path) -> list[str]:
    prefixes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*\d+\s+", "", line)
        prefixes.append(line.strip())
    return prefixes


def normalize_start(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    # remove leading quotes/dashes
    text = re.sub(r"^[\"'«»—-]+\s*", "", text)
    return text


def main():
    jokes = Path("data/jokes_expanded.txt").read_text(encoding="utf-8").splitlines()
    prefixes = load_prefixes(Path("data/prefixes.txt"))

    kept = []
    for joke in jokes:
        norm = normalize_start(joke.lower())
        for pref in prefixes:
            pref_norm = normalize_start(pref.lower())
            words = pref_norm.split()
            head = " ".join(words[:3]) if len(words) >= 3 else pref_norm
            if (
                norm.startswith(pref_norm)
                or norm.startswith(head)
                or pref_norm in norm[: max(100, len(pref_norm) + 20)]
                or head in norm[: max(100, len(head) + 20)]
            ):
                kept.append(joke)
                break

    out_path = Path("data/jokes_prefix_filtered.txt")
    out_path.write_text("\n".join(kept), encoding="utf-8")
    print(f"Kept {len(kept)} / {len(jokes)} jokes -> {out_path}")


if __name__ == "__main__":
    main()
