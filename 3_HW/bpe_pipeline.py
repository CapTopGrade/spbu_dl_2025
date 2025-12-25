import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.trainers import BpeTrainer

# We reuse the GPT-2 ByteLevel pretokenization pattern (OpenAI, 2019) to keep
# compatibility with byte-level BPE used in open models.
SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>"]
RANDOM_SEED = 42
DATA_DIR = Path("dataverse_files/texts")
ARTIFACTS = Path("artifacts")
TOKENIZER_DIR = ARTIFACTS / "tokenizer_final"
PLOTS_DIR = ARTIFACTS / "plots"
METRICS_PATH = ARTIFACTS / "tokenizer_metrics.json"

# Vocabulary sizes we explore for the compression curve.
VOCAB_SIZES = [1000, 2000, 4000, 8000, 12000]
# The tokenizer we will reuse in downstream tasks.
FINAL_VOCAB_SIZE = 12000


def read_texts(paths: List[Path]) -> List[str]:
    texts = []
    for path in paths:
        texts.append(path.read_text(encoding="utf-8"))
    return texts


def split_corpus() -> (List[Path], List[Path]):
    paths = sorted(DATA_DIR.glob("*.txt"))
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(paths)
    split_idx = int(0.8 * len(paths))
    return paths[:split_idx], paths[split_idx:]


def train_bpe_tokenizer(vocab_size: int, files: List[Path], save_dir: Path | None = None) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.train(files=[str(p) for p in files], trainer=trainer)
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(save_dir / "tokenizer.json"))
    return tokenizer


def words_from_texts(texts: List[str]) -> Counter:
    pattern = re.compile(r"\w+", flags=re.UNICODE)
    all_words: Counter = Counter()
    for text in texts:
        all_words.update(pattern.findall(text))
    return all_words


def token_length_for_words(tokenizer: Tokenizer, word_counter: Counter) -> float:
    total_tokens = 0
    total_occurrences = 0
    for word, count in word_counter.items():
        total_tokens += len(tokenizer.encode(word).ids) * count
        total_occurrences += count
    return total_tokens / total_occurrences if total_occurrences else 0.0


def token_length_for_top_words(tokenizer: Tokenizer, word_counter: Counter, top_fraction: float = 0.1) -> float:
    if not word_counter:
        return 0.0
    top_k = max(1, int(len(word_counter) * top_fraction))
    common = word_counter.most_common(top_k)
    total_tokens = 0
    total_occurrences = 0
    for word, count in common:
        total_tokens += len(tokenizer.encode(word).ids) * count
        total_occurrences += count
    return total_tokens / total_occurrences if total_occurrences else 0.0


def compute_metrics(tokenizer: Tokenizer, texts: List[str]) -> Dict:
    encoded = tokenizer.encode_batch(texts)
    token_counts = [len(enc.ids) for enc in encoded]
    total_tokens = sum(token_counts)
    total_bytes = sum(len(t.encode("utf-8")) for t in texts)
    total_chars = sum(len(t) for t in texts)
    word_counter = words_from_texts(texts)
    return {
        "total_tokens": total_tokens,
        "total_bytes": total_bytes,
        "total_chars": total_chars,
        "compression_tokens_per_byte": total_tokens / total_bytes if total_bytes else math.nan,
        "compression_tokens_per_char": total_tokens / total_chars if total_chars else math.nan,
        "mean_tokens_per_word": token_length_for_words(tokenizer, word_counter),
        "mean_tokens_per_word_top10pct": token_length_for_top_words(tokenizer, word_counter),
        "word_types": len(word_counter),
        "word_tokens": sum(word_counter.values()),
    }


def compute_domain_metrics(tokenizer: Tokenizer, domains: Dict[str, List[str]]) -> Dict[str, Dict]:
    return {name: compute_metrics(tokenizer, texts) for name, texts in domains.items()}


def collect_unused_tokens(tokenizer: Tokenizer, texts: List[str]) -> Dict[str, float]:
    vocab_size = tokenizer.get_vocab_size()
    used_ids = set()
    for enc in tokenizer.encode_batch(texts):
        used_ids.update(enc.ids)
    unused = vocab_size - len(used_ids)
    return {"unused_count": unused, "unused_fraction": unused / vocab_size if vocab_size else math.nan}


def build_vocab_curve(train_paths: List[Path], eval_texts: List[str]):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    ratios = []
    for size in VOCAB_SIZES:
        tok = train_bpe_tokenizer(size, train_paths)
        metrics = compute_metrics(tok, eval_texts)
        ratios.append((size, metrics["compression_tokens_per_byte"]))
    xs, ys = zip(*ratios)
    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Vocab size")
    plt.ylabel("Tokens per byte (compression ratio)")
    plt.title("Vocab size vs compression (lower is better)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "vocab_vs_compression.png", dpi=200)
    return ratios


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    train_paths, holdout_paths = split_corpus()
    train_texts = read_texts(train_paths)
    holdout_texts = read_texts(holdout_paths)

    # Train final tokenizer.
    tokenizer = train_bpe_tokenizer(FINAL_VOCAB_SIZE, train_paths, TOKENIZER_DIR)

    # Metrics on holdout poetry.
    holdout_metrics = compute_metrics(tokenizer, holdout_texts)

    # Domain-specific corpora to compare efficiency.
    metadata_text = Path("dataverse_files/metadata.tsv").read_text(encoding="utf-8")
    bibliography_text = Path("dataverse_files/bibliography.tsv").read_text(encoding="utf-8")
    domain_texts = {
        "poetry_holdout": holdout_texts,
        "metadata": [metadata_text],
        "bibliography": [bibliography_text],
    }
    domain_metrics = compute_domain_metrics(tokenizer, domain_texts)

    # Vocab curve.
    curve = build_vocab_curve(train_paths, holdout_texts)

    # Unused tokens on full Pushkin corpus.
    pushkin_texts = read_texts(sorted(DATA_DIR.glob("*.txt")))
    unused = collect_unused_tokens(tokenizer, pushkin_texts)

    result = {
        "final_vocab_size": FINAL_VOCAB_SIZE,
        "holdout_metrics": holdout_metrics,
        "domain_metrics": domain_metrics,
        "vocab_curve": [{"vocab_size": s, "tokens_per_byte": r} for s, r in curve],
        "unused_tokens": unused,
        "train_files_count": len(train_paths),
        "holdout_files_count": len(holdout_paths),
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved tokenizer to {TOKENIZER_DIR}")
    print(f"Saved plot to {PLOTS_DIR / 'vocab_vs_compression.png'}")


if __name__ == "__main__":
    main()
