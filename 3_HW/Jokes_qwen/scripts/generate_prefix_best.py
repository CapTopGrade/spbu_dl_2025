import argparse
import re
from pathlib import Path
from typing import List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def clean_prefixes(path: Path) -> List[str]:
    prefixes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cleaned = re.sub(r"^\s*\d+\s+", "", line).strip()
        if cleaned:
            prefixes.append(cleaned)
    return prefixes


def build_prompt(prefix: str) -> str:
    return (
        f"Инструкция: Начни анекдот с фразы: \"{prefix}\". "
        "Напиши смешной, логичный анекдот на русском языке без оскорблений, без цифр, без списков и дат. "
        "Сделай 2-4 предложения, завершай мысль. "
        f"Ответ: {prefix} "
    )


def is_good(text: str, min_len: int, max_len: int) -> bool:
    if not (min_len <= len(text) <= max_len):
        return False
    if any(ch.isdigit() for ch in text):
        return False
    if re.search(r"(\\b\\w+\\b)(\\s+\\1){2,}", text):
        return False
    sentences = sum(text.count(c) for c in ".!?")
    if sentences < 1:
        return False
    # Ensure большая доля кириллицы среди букв
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", text)
    if letters:
        cyr = sum(1 for ch in letters if re.match(r"[А-Яа-яЁё]", ch))
        if cyr / len(letters) < 0.5:
            return False
    if "..." in text:
        return False
    return True


def generate_for_prefix(
    model,
    tokenizer,
    prefix: str,
    num_samples: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_new_tokens: int,
    min_len: int,
    max_len: int,
):
    prompt = build_prompt(prefix)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    for _ in range(num_samples):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
        gen = re.sub(r"<think>.*?</think>", "", gen, flags=re.DOTALL)
        text = gen.strip().replace("\r", " ").replace("\n", " ")
        if not text.startswith(prefix):
            text = f"{prefix} {text}"
        if is_good(text, min_len=min_len, max_len=max_len):
            return text
    return None


def main():
    parser = argparse.ArgumentParser(description="Best-of generation for prefixes with strong filtering")
    parser.add_argument("--base_model", type=str, default="hf_models/Qwen3-0.6B")
    parser.add_argument("--adapter", type=Path, default=Path("model_new_plain_full/checkpoint-2120"))
    parser.add_argument("--prefixes", type=Path, default=Path("data/prefixes.txt"))
    parser.add_argument("--out", type=Path, default=Path("outputs/jokes_pref_best.txt"))
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=90)
    parser.add_argument("--temperature", type=float, default=0.55)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    parser.add_argument("--min_len", type=int, default=60)
    parser.add_argument("--max_len", type=int, default=180)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.eval()

    prefixes = clean_prefixes(args.prefixes)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with args.out.open("w", encoding="utf-8") as f:
        for idx, pref in enumerate(prefixes, 1):
            print(f"[gen] {idx}/{len(prefixes)}: {pref}")
            joke = generate_for_prefix(
                model,
                tokenizer,
                prefix=pref,
                num_samples=args.num_samples,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                max_new_tokens=args.max_new_tokens,
                min_len=args.min_len,
                max_len=args.max_len,
            )
            if joke:
                kept += 1
                f.write(joke + "\n")
            else:
                f.write(f"{pref} ... (не удалось сгенерировать осмысленный анекдот)\n")
            f.flush()
    print(f"[done] kept {kept}/{len(prefixes)} jokes to {args.out}")


if __name__ == "__main__":
    main()
