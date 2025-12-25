import argparse
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)


def load_text(path: Path) -> list[str]:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            lines.append(line)
    return lines


def build_dataset(tokenizer, texts: list[str], max_length: int) -> Dataset:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def _tok(text: str):
        tok = tokenizer(
            text + tokenizer.eos_token,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        labels = [(tid if tid != tokenizer.pad_token_id else -100) for tid in tok["input_ids"]]
        return {
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "labels": labels,
        }

    return Dataset.from_list([_tok(t) for t in texts])


def get_model_and_tokenizer(model_id: str, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map={"": 0},
        trust_remote_code=trust_remote_code,
    )
    model.config.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def attach_lora(model, r: int, alpha: int, dropout: float, target_modules: list[str]):
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def parse_args():
    ap = argparse.ArgumentParser(description="LM-style LoRA training on jokes")
    ap.add_argument("--data", type=Path, default=Path("data/jokes_prefix_filtered.txt"))
    ap.add_argument("--model_id", type=str, default="Qwen/Qwen3-0.6B")
    ap.add_argument("--output_dir", type=Path, default=Path("outputs/qwen3-lm-lora"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--trust_remote_code", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True

    texts = load_text(args.data)
    model, tokenizer = get_model_and_tokenizer(args.model_id, args.trust_remote_code)
    model = attach_lora(
        model,
        r=16,
        alpha=32,
        dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    ds = build_dataset(tokenizer, texts, args.max_length)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.0,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=False,
        fp16=True,
        optim="adamw_torch",
        report_to=["none"],
        seed=42,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=training_args,
        data_collator=default_data_collator,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    main()
