import argparse
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from translit_utils import ru_to_lat, lat_to_ru


def load_prefixes(path: Path) -> list[str]:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cleaned = re.sub(r"^\\s*\\d+\\s+", "", line).strip()
        if cleaned:
            lines.append(cleaned)
    return lines


def build_prompt(prefix_lat: str, prefill_prefix: bool) -> str:
    instr = (
        "Zadanie: nachni anekdot s frazy: "
        f"\"{prefix_lat}\". "
        "Napiši smeshnoj, logičnyj anekdot po-russki (translit) bez oskorblenij i cifr. "
        "Sdelaj 2-3 predloženija, zavershaj mysl'.\n"
        "Otvet:"
    )
    return f"{instr} {prefix_lat} " if prefill_prefix else instr


def clean_lat(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    parser = argparse.ArgumentParser(description="Generate jokes in translit then convert to Cyrillic")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", type=Path, default=Path("model_new_anec"))
    parser.add_argument("--out", type=Path, default=Path("outputs/jokes_pref_raw_translit.txt"))
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.08)
    parser.add_argument("--min_len", type=int, default=30)
    parser.add_argument("--max_len", type=int, default=240)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--prefixes", type=Path, default=Path("data/prefixes.txt"))
    parser.add_argument("--num_per_prefix", type=int, default=3)
    parser.add_argument("--prefill_prefix", action="store_true")
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

    prefixes_ru = load_prefixes(args.prefixes)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen_lat: set[str] = set()
    kept = 0
    results_ru = []

    for pref_ru in prefixes_ru:
        pref_lat = ru_to_lat(pref_ru)
        prompt_text = build_prompt(pref_lat, args.prefill_prefix)
        model_inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        input_len = model_inputs["input_ids"].shape[1]

        best_line = None
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
            text_lat = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            text_lat = clean_lat(text_lat)
            if args.prefill_prefix and not text_lat.lower().startswith(pref_lat.lower()):
                text_lat = f"{pref_lat} {text_lat}"
            if text_lat in seen_lat:
                continue
            text_ru = lat_to_ru(text_lat)
            # enforce prefix start
            if not text_ru.startswith(pref_ru):
                text_ru = f"{pref_ru} {text_ru}"
            # trim to 3 sentences
            parts = re.split(r"(?<=[.!?])\\s+", text_ru)
            text_ru = " ".join(parts[:3]).strip()
            if len(text_ru) < args.min_len or len(text_ru) > args.max_len:
                continue
            best_line = text_ru
            break

        if best_line:
            kept += 1
            results_ru.append(best_line)
            with args.out.open("a", encoding="utf-8") as f:
                f.write(best_line + "\n")
            if kept % 25 == 0:
                print(f"[gen] kept {kept}")
        else:
            print(f"[warn] no sample for prefix: {pref_ru}")

    print(f"[done] kept {kept} jokes to {args.out}")


if __name__ == "__main__":
    main()
