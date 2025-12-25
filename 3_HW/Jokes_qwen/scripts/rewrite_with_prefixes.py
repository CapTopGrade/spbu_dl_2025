import argparse
import random
import re
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_list(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_prefixes(path: Path) -> List[str]:
    prefixes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cleaned = re.sub(r"^\s*\d+\s+", "", line).strip()
        if cleaned:
            prefixes.append(cleaned)
    return prefixes


def truncate_sentences(text: str, max_sentences: int = 2) -> str:
    parts = re.split(r"([.!?])", text)
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        sent = (parts[i] + parts[i + 1]).strip()
        if sent:
            sentences.append(sent)
        if len(sentences) >= max_sentences:
            break
    return " ".join(sentences) if sentences else text


def clean_text(text: str, prefix: str, min_len: int, max_len: int) -> str | None:
    text = text.strip().replace("\r", " ").replace("\n", " ")
    # cut off obvious instruction tails
    text = re.split(r"(Нужно .*|Требуется .*|Ваша задача .*|Ответ:)", text)[0].strip()
    text = truncate_sentences(text, max_sentences=2)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\d+", "", text)
    if not text.startswith(prefix):
        text = f"{prefix} {text}"
    if any(ch.isdigit() for ch in text):
        return None
    if len(text) < min_len or len(text) > max_len:
        return None
    return text


def main():
    parser = argparse.ArgumentParser(description="Rewrite existing jokes to fit prefixes")
    parser.add_argument("--base_model", type=str, default="hf_models/Qwen3-0.6B")
    parser.add_argument("--prefixes", type=Path, default=Path("data/prefixes.txt"))
    parser.add_argument("--jokes", type=Path, default=Path("data/jokes_clean_strict.txt"))
    parser.add_argument("--out", type=Path, default=Path("outputs/jokes_rewritten.txt"))
    parser.add_argument("--num_candidates", type=int, default=3)
    parser.add_argument("--min_len", type=int, default=50)
    parser.add_argument("--max_len", type=int, default=200)
    args = parser.parse_args()

    torch.manual_seed(42)
    jokes = load_list(args.jokes)
    prefixes = load_prefixes(args.prefixes)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.eval()

    prompt_tpl = (
        "Перепиши анекдот, сделай его короче (2-3 предложения) и начни строго с фразы \"{prefix}\". "
        "Сохрани юмор, не используй цифры.\n"
        "Вот анекдот: {joke}\n"
        "Ответ: "
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with args.out.open("w", encoding="utf-8") as f:
        for idx, pref in enumerate(prefixes, 1):
            base_joke = random.choice(jokes)
            prompt = prompt_tpl.format(prefix=pref, joke=base_joke)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            input_len = inputs["input_ids"].shape[1]

            accepted = None
            for _ in range(args.num_candidates):
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        do_sample=True,
                        temperature=0.55,
                        top_p=0.9,
                        repetition_penalty=1.12,
                        max_new_tokens=80,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                gen = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
                candidate = clean_text(gen, pref, args.min_len, args.max_len)
                if candidate:
                    accepted = candidate
                    break
            if accepted is None:
                accepted = f"{pref} ... (не удалось переписать анекдот)"
            f.write(accepted + "\n")
            kept += 1
            if kept % 10 == 0:
                print(f"[rewrite] {kept}/{len(prefixes)} готово")
    print(f"[done] сохранено {kept} строк в {args.out}")


if __name__ == "__main__":
    main()
