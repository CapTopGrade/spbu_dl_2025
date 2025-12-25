import re, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

adapter = "model_new_plain_full/checkpoint-2120"
model_id = 'hf_models/Qwen3-0.6B'

prefixes = [re.sub(r'^\s*\d+\s+','',l).strip() for l in Path('data/prefixes.txt').read_text(encoding='utf-8').splitlines()[:5]]

print('using adapter', adapter)

tokenizer = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map={'':0}, trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload(); model.eval()

prompt_tpl = (
    'Инструкция: Начни анекдот с фразы: "{pref}". '
    'Напиши смешной, логичный анекдот на русском языке без оскорблений. '
    'Сделай 2-4 предложения, без цифр и списков, заверши мысль. '
    'Ответ: {pref} '
)

for pref in prefixes:
    prompt = prompt_tpl.format(pref=pref)
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    out = model.generate(**inputs, do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.12, max_new_tokens=80, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
    gen = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    if not gen.startswith(pref):
        gen = f'{pref} {gen}'
    print('\n', pref, '->', gen)
