import re
from pathlib import Path


def clean_prefixes(path: Path):
    return [re.sub(r"^\\s*\\d+\\s+", "", line).strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def truncate_sentences(text: str, max_sent: int = 3) -> str:
    parts = re.split(r"([.!?])", text)
    res = []
    for i in range(0, len(parts) - 1, 2):
        sent = (parts[i] + parts[i + 1]).strip()
        if sent:
            res.append(sent)
        if len(res) >= max_sent:
            break
    return " ".join(res) if res else text


def clean_line(line: str) -> str:
    line = line.replace("\n", " ")
    line = re.sub(r"^\\s*\\d+\\s+", "", line).strip()
    line = line.replace("\\r", " ").replace("\\n", " ")
    line = re.sub(r"\\s+", " ", line)
    line = re.sub(r"\\d+", "", line)  # убрать числа
    line = re.sub(r"\\(с\\).*", "", line, flags=re.IGNORECASE)  # убрать ссылки вида (с) ...
    line = truncate_sentences(line, max_sent=3)
    line = re.sub(r"\\s+", " ", line).strip(" .")
    return line


def main():
    raw_path = Path("outputs/jokes_pref_raw.txt")
    prefixes = clean_prefixes(Path("data/prefixes.txt"))
    lines = raw_path.read_text(encoding="utf-8", errors="replace").splitlines()

    selected = []
    used = set()
    for pref in prefixes:
        best = None
        for ln in lines:
            cl = clean_line(ln)
            cl = cl.replace("\\n", " ").replace("\n", " ")
            if not cl.startswith(pref):
                continue
            if len(cl) < 30 or len(cl) > 200:
                continue
            if cl in used:
                continue
            best = cl
            break
        if best is None:
            best = f"{pref} ... (не удалось подобрать анекдот)"
        used.add(best)
        selected.append(best)

    out_path = Path("outputs/jokes_final.txt")
    out_path.write_text("\\n".join(selected) + "\\n", encoding="utf-8")
    print(f"[done] wrote {len(selected)} jokes to {out_path}")


if __name__ == "__main__":
    main()
