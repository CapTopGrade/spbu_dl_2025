import torch
import re
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

prefixes = [re.sub(r'^\s*\d+\s+','',l).strip() for l in Path('data/prefixes.txt').read_text(encoding='utf-8').splitlines()][:3]
print('prefix count', len(prefixes))

tokenizer = AutoTokenizer.from_pretrained('model_new', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('hf_models/Qwen3-0.6B', torch_dtype=torch.float16, device_map={'':0}, trust_remote_code=True)
model = PeftModel.from_pretrained(model, 'model_new')
model = model.merge_and_unload()
model.eval()

def build(prefix: str) -> str:
    return (
        f'Инструкция: Начни анекдот с фразы: "{prefix}". '
        'Напиши смешной, логичный анекдот на русском языке без оскорблений. '
        'Сделай 2-4 предложения, без списков и номеров, заверши мысль. '
        f'Ответ: {prefix} '
    )

for pref in prefixes:
    prompt = build(pref)
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    out = model.generate(
        **inputs,
        do_sample=True,
        temperature=0.55,
        top_p=0.8,
        repetition_penalty=1.2,
        max_new_tokens=60,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    print('\n', pref, '->', gen)
