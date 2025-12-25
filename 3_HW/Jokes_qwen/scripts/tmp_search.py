import re, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

prefix = re.sub(r'^\s*\d+\s+','', Path('data/prefixes.txt').read_text(encoding='utf-8').splitlines()[0]).strip()
print('prefix', prefix)

model_id = 'hf_models/Qwen3-0.6B'
tokenizer = AutoTokenizer.from_pretrained('model_new', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map={'':0}, trust_remote_code=True)
model = PeftModel.from_pretrained(model, 'model_new')
model = model.merge_and_unload(); model.eval()

prompt = (
    f'Инструкция: Начни анекдот с фразы: "{prefix}". '
    'Напиши смешной, логичный анекдот на русском языке без оскорблений. '
    'Сделай 2-4 предложения, без цифр и списков, заверши мысль. '
    f'Ответ: {prefix} '
)
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

candidates = []
for i in range(50):
    out = model.generate(**inputs, do_sample=True, temperature=0.9, top_p=0.95, repetition_penalty=1.1, max_new_tokens=90, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
    gen = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    text = gen
    if not text.startswith(prefix):
        text = f'{prefix} {text}'
    text = text.replace('\n',' ').strip()
    if any(ch.isdigit() for ch in text):
        continue
    if len(text) < 40 or len(text) > 220:
        continue
    puncts = sum(text.count(c) for c in '.!?')
    if puncts < 1:
        continue
    # penalize repeating segments
    if re.search(r'(\b\w+\b)(\s+\1){2,}', text):
        continue
    candidates.append(text)

print('kept', len(candidates))
for t in candidates[:5]:
    print('-', t)
