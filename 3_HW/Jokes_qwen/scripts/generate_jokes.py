import argparse
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_prefixes(path: Path) -> list[str]:
    prefixes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cleaned = re.sub(r"^\\s*\\d+\\s+", "", line).strip()
        if cleaned:
            prefixes.append(cleaned)
    return prefixes


def build_prompt(tokenizer, plain_format: bool, prefix: str | None, prefill_prefix: bool) -> str:
    if plain_format:
        instr = (
            "Задание: "
            "Начни анекдот с фразы: "
            f"\"{prefix}\". "
            "Напиши смешной, логичный анекдот на русском языке без оскорблений и цифр. "
            "Сделай 2-3 предложения, заверши мысль.\n"
            "Ответ:"
        )
        return f"{instr} {prefix} " if prefill_prefix and prefix else instr

    system_prompt = "Ты — доброжелательный рассказчик анекдотов. Отвечай коротко и логично, без оскорблений."
    user_prompt = (
        "Начни анекдот с указанной фразы. Напиши смешной и связный анекдот на русском языке, без оскорблений и цифр. "
        "Сделай 2-3 предложения и заверши мысль."
    )
    if prefix:
        user_prompt += f' Начни с фразы: "{prefix}".'
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if prefill_prefix and prefix:
        prompt += prefix + " "
    return prompt


def cyr_ratio(text: str) -> float:
    if not text:
        return 0.0
    total = len(text)
    cyr = len(re.findall(r"[А-Яа-яЁё]", text))
    return cyr / max(total, 1)


def clean_text(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    parser = argparse.ArgumentParser(description="Generate jokes with a LoRA-tuned Qwen3 model")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter", type=Path, default=Path("outputs/qwen3-jokes-lora"))
    parser.add_argument("--out", type=Path, default=Path("outputs/jokes.txt"))
    parser.add_argument("--num", type=int, default=300, help="Number of samples if prefixes are not provided")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--min_len", type=int, default=24, help="Drop generations shorter than this")
    parser.add_argument("--max_len", type=int, default=240, help="Drop generations longer than this")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--plain_format", action="store_true", help="Use plain instruction/answer format (must match training)")
    parser.add_argument("--prefixes", type=Path, help="Optional prefixes file; will generate for each prefix")
    parser.add_argument("--num_per_prefix", type=int, default=1, help="How many samples to try per prefix when provided")
    parser.add_argument("--prefill_prefix", action="store_true", help="Seed the answer with the prefix to force start")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map={"": 0},
        trust_remote_code=args.trust_remote_code,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.eval()

    prefixes = load_prefixes(args.prefixes) if args.prefixes else None
    tasks: list[tuple[str | None, int]] = []
    if prefixes:
        for pref in prefixes:
            tasks.append((pref, args.num_per_prefix))
    else:
        tasks.append((None, args.num))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    kept = 0

    for prefix, total in tasks:
        prompt_text = build_prompt(tokenizer, args.plain_format, prefix, args.prefill_prefix)
        model_inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        input_len = model_inputs["input_ids"].shape[1]

        for idx in range(total):
            with torch.no_grad():
                out = model.generate(
                    **model_inputs,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated_tokens = out[0][input_len:]
            text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            joke = clean_text(text)

            if prefix and args.prefill_prefix and not joke.startswith(prefix):
                joke = f"{prefix} {joke}"
            if not joke:
                continue
            if len(joke) < args.min_len or len(joke) > args.max_len:
                continue
            if cyr_ratio(joke) < 0.7:
                continue
            if re.search(r"\d", joke):
                continue
            # limit to max 3 sentences
            parts = re.split(r"(?<=[.!?])\\s+", joke)
            joke = " ".join(parts[:3]).strip()
            if joke in seen:
                continue
            seen.add(joke)
            kept += 1
            with args.out.open("a", encoding="utf-8") as f:
                f.write(joke + "\n")
            if kept % 25 == 0:
                print(f"[gen] kept {kept} so far")

    print(f"[done] kept {kept} jokes to {args.out}")


if __name__ == "__main__":
    main()
