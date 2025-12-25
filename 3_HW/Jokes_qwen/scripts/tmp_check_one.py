import re, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from generate_prefix_best import is_good

prefix = re.sub(r'^\s*\d+\s+', '', Path('data/prefixes.txt').read_text(encoding='utf-8').splitlines()[0]).strip()
print('prefix', prefix)

tokenizer = AutoTokenizer.from_pretrained('model_new_plain_full/checkpoint-2120', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('hf_models/Qwen3-0.6B', torch_dtype=torch.float16, device_map={'':0}, trust_remote_code=True)
model = PeftModel.from_pretrained(model, 'model_new_plain_full/checkpoint-2120')
model = model.merge_and_unload(); model.eval()

prompt = (
    f'Инструкция: Начни анекдот с фразы: "{prefix}". '
    'Напиши смешной, логичный анекдот на русском языке без оскорблений и без цифр. '
    'Сделай 2-4 предложения, без списков, заверши мысль. '
    f'Ответ: {prefix} '
)
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
out = model.generate(**inputs, do_sample=True, temperature=0.6, top_p=0.9, repetition_penalty=1.12, max_new_tokens=80, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
gen = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
text = gen.strip().replace('\n',' ')
if not text.startswith(prefix):
    text = f'{prefix} {text}'
print('raw:', text)
print('len', len(text), 'digits', any(ch.isdigit() for ch in text))
letters = re.findall(r"[A-Za-zА-Яа-яЁё]", text)
cyr = sum(1 for ch in letters if re.match(r"[А-Яа-яЁё]", ch))
print('cyr ratio', cyr/len(letters) if letters else 0)
print('sentences', sum(text.count(c) for c in '.!?'))
print('is_good', is_good(text, 40, 200))
