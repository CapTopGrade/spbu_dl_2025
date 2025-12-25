import math
import random
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer

RANDOM_SEED = 42
DATA_DIR = Path("dataverse_files/texts")
TOKENIZER_PATH = Path("artifacts/tokenizer_final/tokenizer.json")
ARTIFACTS_DIR = Path("artifacts/simple_model")


def split_corpus() -> List[Path]:
    paths = sorted(DATA_DIR.glob("*.txt"))
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(paths)
    split_idx = int(0.8 * len(paths))
    return paths[:split_idx]


def load_token_ids(tokenizer: Tokenizer, files: List[Path], max_tokens: int = 120_000) -> torch.Tensor:
    ids = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        ids.extend(tokenizer.encode(text).ids)
        if len(ids) >= max_tokens:
            break
    return torch.tensor(ids[:max_tokens], dtype=torch.long)


class LMSequenceDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        x = self.tokens[idx: idx + self.seq_len]
        y = self.tokens[idx + 1: idx + self.seq_len + 1]
        return x, y


class TinyGRULM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embed(x)
        out, _ = self.rnn(emb)
        return self.proj(out)


def train():
    torch.manual_seed(RANDOM_SEED)
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    train_files = split_corpus()
    token_tensor = load_token_ids(tokenizer, train_files)

    seq_len = 64
    dataset = LMSequenceDataset(token_tensor, seq_len)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

    model = TinyGRULM(tokenizer.get_vocab_size())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 2
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        avg_loss = total_loss / batches
        ppl = math.exp(avg_loss)
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, ppl={ppl:.2f}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ARTIFACTS_DIR / "tiny_gru_lm.pt")
    with open(ARTIFACTS_DIR / "train_log.txt", "w", encoding="utf-8") as f:
        f.write(f"avg_loss={avg_loss:.4f}, ppl={ppl:.2f}\n")

    print(f"Saved model to {ARTIFACTS_DIR / 'tiny_gru_lm.pt'}")


if __name__ == "__main__":
    train()
