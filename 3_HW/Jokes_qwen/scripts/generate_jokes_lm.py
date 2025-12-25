import argparse
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_prefixes(path: Path) -> list[str]:
    prefixes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = re.sub(r"^\\s*\\d+\\s+", "", line).strip()
        if line:
            prefixes.append(line)
    return prefixes


def main():
    ap = argparse.ArgumentParser(description="LM generation for prefixes (no chat formatting)")
    ap.add_argument("--base_model", type=str, default="Qwen/Qwen3-0.6B")
    ap.add_argument("--adapter", type=Path, default=Path("model_new_anec"))
    ap.add_argument("--out", type=Path, default=Path("outputs/jokes_pref_raw_lm.txt"))
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    ap.add_argument("--min_len", type=int, default=30)
    ap.add_argument("--max_len", type=int, default=240)
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--prefixes", type=Path, default=Path("data/prefixes.txt"))
    ap.add_argument("--num_per_prefix", type=int, default=5)
    args = ap.parse_args()

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

    prefixes = load_prefixes(args.prefixes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    kept = 0

    for pref in prefixes:
        prompt = pref + " "
        model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = model_inputs["input_ids"].shape[1]
        best = None
        for _ in range(args.num_per_prefix):
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
            gen_tokens = out[0][input_len:]
            text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            text = text.replace("\r", " ").replace("\n", " ")
            text = re.sub(r"\s+", " ", text).strip()
            if not text.startswith(pref):
                text = f"{pref} {text}"
            if len(text) < args.min_len or len(text) > args.max_len:
                continue
            parts = re.split(r"(?<=[.!?])\\s+", text)
            text = " ".join(parts[:3]).strip()
            if text in seen:
                continue
            seen.add(text)
            best = text
            break
        if best:
            kept += 1
            with args.out.open("a", encoding="utf-8") as f:
                f.write(best + "\n")
            if kept % 25 == 0:
                print(f"[gen] kept {kept}")
        else:
            print(f"[warn] no sample for prefix: {pref}")

    print(f"[done] kept {kept} -> {args.out}")


if __name__ == "__main__":
    main()
