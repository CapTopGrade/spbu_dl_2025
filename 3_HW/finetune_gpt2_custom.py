from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

RANDOM_SEED = 42
DATA_DIR = Path("dataverse_files/texts")
TOKENIZER_PATH = Path("artifacts/tokenizer_final/tokenizer.json")
OUTPUT_DIR = Path("artifacts/gpt2_custom")


def split_corpus() -> List[Path]:
    paths = sorted(DATA_DIR.glob("*.txt"))
    import random

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(paths)
    split_idx = int(0.8 * len(paths))
    return paths[:split_idx]


class BlockDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: PreTrainedTokenizerFast, block_size: int = 128):
        joined = "\n\n".join(texts)
        encoded = tokenizer(
            joined,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[0]
        blocks = len(encoded) // block_size
        encoded = encoded[: blocks * block_size]
        self.examples = encoded.view(blocks, block_size)

    def __len__(self):
        return self.examples.size(0)

    def __getitem__(self, idx):
        return {"input_ids": self.examples[idx]}


def main():
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(TOKENIZER_PATH),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    tokenizer.padding_side = "right"

    train_paths = split_corpus()[:120]  # keep fine-tune quick
    train_texts = [p.read_text(encoding="utf-8") for p in train_paths]
    dataset = BlockDataset(train_texts, tokenizer)

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        warmup_steps=0,
        logging_steps=10,
        save_strategy="no",
        eval_strategy="no",
        max_steps=50,
        seed=RANDOM_SEED,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    train_result = trainer.train()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    trainer.save_state()

    metrics_path = OUTPUT_DIR / "train_metrics.txt"
    metrics_path.write_text(str(train_result.metrics), encoding="utf-8")
    print(f"Saved fine-tuned GPT-2 to {OUTPUT_DIR}")
    print(f"Training metrics: {train_result.metrics}")


if __name__ == "__main__":
    main()
