import torch, re
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

prefixes = [re.sub(r'^\s*\d+\s+','',l).strip() for l in Path('data/prefixes.txt').read_text(encoding='utf-8').splitlines()][:5]
print('prefix count', len(prefixes))

model_id='hf_models/Qwen3-0.6B'
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map={'':0}, trust_remote_code=True)
model.eval()

def build(prefix: str) -> str:
    return (
        f'Инструкция: Начни анекдот с фразы: "{prefix}". '
        'Напиши короткий смешной анекдот на русском языке, 2-3 предложения, без цифр и списков, завершай мысль. '
        f'Ответ: {prefix} '
    )

for pref in prefixes:
    prompt = build(pref)
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    out = model.generate(
        **inputs,
        do_sample=True,
        temperature=0.4,
        top_p=0.85,
        repetition_penalty=1.15,
        max_new_tokens=70,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    print('\n', pref, '->', gen)
