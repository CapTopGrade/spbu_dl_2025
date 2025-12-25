import re, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

prefix = re.sub(r'^\s*\d+\s+', '', Path('data/prefixes.txt').read_text(encoding='utf-8').splitlines()[0]).strip()
print('prefix', prefix)

model_id='Qwen/Qwen3-0.6B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=None)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map={'':0}, trust_remote_code=True, token=None)
model.eval()

prompt = f'Ты — комик. Придумай короткий смешной анекдот на русском, начинающийся с "{prefix}". 2-3 предложения, без чисел.'
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
out = model.generate(**inputs, do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.1, max_new_tokens=80, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
text = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(text)
