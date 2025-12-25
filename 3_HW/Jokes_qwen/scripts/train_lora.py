import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

SYSTEM_PROMPT = (
    "Ты дружелюбный рассказчик анекдотов, который пишет краткие, логичные шутки без оскорблений и цифр. "
    "Будь вежлив и заканчивай мысль."
)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def build_dataset(tokenizer, data: list[dict], max_length: int, plain_format: bool) -> Dataset:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def _format(example: dict):
        if plain_format:
            text = (
                f"Instruction: {example['prompt']}\n"
                f"Answer: {example['response']}{tokenizer.eos_token}"
            )
            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            labels = [
                (tok if tok != tokenizer.pad_token_id else -100)
                for tok in tokenized["input_ids"]
            ]
            return {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": labels,
            }

        messages_full = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["response"]},
        ]
        messages_prefix = messages_full[:2]

        prompt_text = tokenizer.apply_chat_template(
            messages_prefix, tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            messages_full, tokenize=False, add_generation_prompt=False
        )

        tokenized_prefix = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        prefix_len = len(tokenized_prefix["input_ids"])

        tokenized_full = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

        input_ids = tokenized_full["input_ids"]
        labels = [-100] * len(input_ids)
        for idx in range(min(prefix_len, len(input_ids)), len(input_ids)):
            labels[idx] = input_ids[idx]

        labels = [
            (tok if tok != tokenizer.pad_token_id else -100) for tok in labels
        ]

        return {
            "input_ids": input_ids,
            "attention_mask": tokenized_full["attention_mask"],
            "labels": labels,
        }

    rows = [_format(ex) for ex in data]
    return Dataset.from_list(rows)


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
    parser = argparse.ArgumentParser(description="Train LoRA on jokes for Qwen3/2.5")
    parser.add_argument("--data", type=Path, default=Path("data/sft_dataset.jsonl"))
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/qwen3-jokes-lora"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--plain_format", action="store_true", help="Use simple instruction/answer text instead of chat template")
    parser.add_argument("--resume_from_checkpoint", type=Path, help="Path to a checkpoint to resume from")
    parser.add_argument("--init_adapter", type=Path, help="Load an existing LoRA adapter and continue training")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True

    records = load_jsonl(args.data)
    model, tokenizer = get_model_and_tokenizer(args.model_id, args.trust_remote_code)
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        model.print_trainable_parameters()
    else:
        model = attach_lora(
            model,
            r=16,
            alpha=32,
            dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

    ds = build_dataset(tokenizer, records, args.max_length, plain_format=args.plain_format)
    data_collator = default_data_collator

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
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    main()
