import argparse
import math
import random
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BAD_WORDS = [
    "???",
    "????",
    "??",
    "??",
    "???",
    "???",
    "???",
    "???",
    "???",
    "???",
    "???",
    "???",
    "???",
    "???",
    "???",
    "????",
    "????",
    "???",
    "????",
    "?????",
    "?????",
    "????",
    "???",
    "?????",
    "?????",
    "??????",
    "????",
    "?????",
    "?????",
    "????",
    "?????????",
    "????",
    "????",
    "????",
    "?????:",
    "????? -",
    "????? ?",
    "??????:",
    "?????? -",
    "?????? ?",
]


def load_prefixes(path: Path) -> list[str]:
    prefixes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*\d+[\.)-]*\s*", "", line)
        if line:
            prefixes.append(line)
    return prefixes


def clean_text(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"\b(?:Instruction|Answer)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sentence_count(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", text) if s])


def cyr_ratio(text: str) -> float:
    if not text:
        return 0.0
    total = len(text)
    cyr = len(re.findall(r"[А-Яа-яЁё]", text))
    return cyr / max(total, 1)


def repetition_penalty_score(text: str) -> float:
    words = re.findall(r"[A-Za-zА-Яа-яЁё]+", text.lower())
    if not words:
        return -2.0
    uniq = len(set(words))
    ratio = uniq / max(len(words), 1)
    if ratio < 0.45:
        return -2.0
    if ratio < 0.55:
        return -1.0
    return 0.0


def has_bad_words(text: str) -> bool:
    low = text.lower()
    return any(b in low for b in BAD_WORDS)


def passes_filters(
    full: str,
    min_chars: int,
    max_chars: int,
    min_sent: int,
    max_sent: int,
    min_cyr: float,
    min_unique_ratio: float,
) -> bool:
    if len(full) < min_chars or len(full) > max_chars:
        return False
    if re.search(r"\d", full):
        return False
    if has_bad_words(full):
        return False
    if cyr_ratio(full) < min_cyr:
        return False
    words = re.findall(r"[A-Za-zА-Яа-яЁё]+", full.lower())
    if not words:
        return False
    uniq_ratio = len(set(words)) / max(len(words), 1)
    if uniq_ratio < min_unique_ratio:
        return False
    n_sent = sentence_count(full)
    if n_sent < min_sent or n_sent > max_sent:
        return False
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", full) if s.strip()]
    for s in sentences:
        if len(s) < 20:
            return False
    last = sentences[-1]
    if re.search(r"\b[А-Яа-яA-Za-z]\.$", last):
        return False
    if re.search(r"\.\.\.\s*$", last):
        return False
    return True


def score_candidate(prefix: str, response: str) -> float:
    if not response.startswith("..."):
        return -3.0
    full = f"{prefix} {response}".strip()
    if re.search(r"\d", full):
        return -3.0
    if has_bad_words(full):
        return -2.5
    if len(full) < 50:
        return -1.5
    if len(full) > 320:
        return -1.0

    sc = 0.0
    sc += repetition_penalty_score(full)
    n_sent = sentence_count(full)
    if 2 <= n_sent <= 3:
        sc += 2.0
    elif n_sent == 1:
        sc += 0.2
    elif n_sent == 4:
        sc += 0.5
    else:
        sc -= 0.5
    return sc


def build_prompt(prefix: str, template: str) -> str:
    body = template.format(prefix=prefix)
    return f"Instruction: {body}\nAnswer:"


def generate_candidates(
    model,
    tokenizer,
    prefix: str,
    template: str,
    num: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    num_beams: int,
):
    prompt = build_prompt(prefix, template)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    do_sample = num_beams <= 1
    num_return = num if num_beams <= 1 else min(num, num_beams)

    candidates = []
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=num_return,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=True if num_beams > 1 else False,
        )
    if output.dim() == 1:
        output = output.unsqueeze(0)
    for seq in output:
        gen = seq[input_len:]
        text = tokenizer.decode(gen, skip_special_tokens=True)
        text = clean_text(text)
        # strip any leading prompt echoes
        if text.startswith(prefix):
            text = text[len(prefix) :].lstrip()
        # ensure starts with ellipsis
        if not text.startswith("..."):
            text = "... " + text.lstrip(" .,-")
        # ensure ending punctuation
        if text and text[-1] not in ".!?":
            text = text + "."
        # limit to at most 5 sentences
        parts = re.split(r"(?<=[.!?])\s+", text)
        text = " ".join(parts[:5]).strip()
        candidates.append(text)
    return candidates


def main():
    ap = argparse.ArgumentParser(description="Generate jokes from prefixes with a LoRA adapter")
    ap.add_argument("--base_model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--prefixes", type=Path, default=Path("..") / "prefixes.txt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num_per_prefix", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--prompt_template", type=str, default="{prefix}")
    ap.add_argument("--max_attempts", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--min_score", type=float, default=-0.3)
    ap.add_argument("--min_chars", type=int, default=80)
    ap.add_argument("--max_chars", type=int, default=260)
    ap.add_argument("--min_sentences", type=int, default=2)
    ap.add_argument("--max_sentences", type=int, default=3)
    ap.add_argument("--min_cyr", type=float, default=0.8)
    ap.add_argument("--min_unique_ratio", type=float, default=0.55)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=3)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trust_remote_code", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map={"": 0},
        trust_remote_code=args.trust_remote_code,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.eval()

    prefixes = load_prefixes(args.prefixes)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    kept = []
    scores = []
    for prefix in prefixes:
        best = None
        best_score = -1e9
        tried = 0
        while tried < args.max_attempts:
            batch = min(args.batch_size, args.max_attempts - tried)
            tried += batch
            candidates = generate_candidates(
                model,
                tokenizer,
                prefix,
                args.prompt_template,
                batch,
                args.max_new_tokens,
                args.temperature,
                args.top_p,
                args.repetition_penalty,
                args.no_repeat_ngram_size,
                args.num_beams,
            )
            for cand in candidates:
                score = score_candidate(prefix, cand)
                full = f"{prefix} {cand}".strip()
                if score > best_score:
                    best_score = score
                    best = cand
                if score >= args.min_score and passes_filters(
                    full,
                    args.min_chars,
                    args.max_chars,
                    args.min_sentences,
                    args.max_sentences,
                    args.min_cyr,
                    args.min_unique_ratio,
                ):
                    best = cand
                    best_score = score
                    tried = args.max_attempts
                    break
        if best is None:
            best = "... "
            best_score = -5.0
        scores.append(best_score)
        kept.append(f"{prefix} {best}".strip())

    args.out.write_text("\n".join(kept) + "\n", encoding="utf-8")
    avg_score = sum(scores) / max(len(scores), 1)
    print(f"[done] saved {len(kept)} jokes to {args.out}")
    print(f"[score] avg={avg_score:.3f}")


if __name__ == "__main__":
    main()
