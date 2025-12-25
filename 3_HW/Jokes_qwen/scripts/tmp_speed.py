import re, time, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

prefix = re.sub(r'^\s*\d+\s+', '', Path('data/prefixes.txt').read_text(encoding='utf-8').splitlines()[0]).strip()
print('prefix', prefix)
start = time.time()

tokenizer = AutoTokenizer.from_pretrained('model_new_plain_full/checkpoint-2120', trust_remote_code=True)
print('tok loaded', time.time() - start)
model = AutoModelForCausalLM.from_pretrained('hf_models/Qwen3-0.6B', torch_dtype=torch.float16, device_map={'': 0}, trust_remote_code=True)
model = PeftModel.from_pretrained(model, 'model_new_plain_full/checkpoint-2120')
model = model.merge_and_unload()
model.eval()
print('model ready', time.time() - start)

prompt = (
    f'Инструкция: Начни анекдот с фразы: "{prefix}". '
    'Напиши смешной, логичный анекдот на русском языке без оскорблений и без цифр. '
    'Сделай 2-4 предложения, без списков, заверши мысль. '
    f'Ответ: {prefix} '
)
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
t0 = time.time()
out = model.generate(
    **inputs,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
    repetition_penalty=1.15,
    max_new_tokens=80,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
print('gen time', time.time() - t0)
text = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(text[:400])
print('total', time.time() - start)
