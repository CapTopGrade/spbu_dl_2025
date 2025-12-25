import re, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

prefix = re.sub(r'^\s*\d+\s+', '', Path('data/prefixes.txt').read_text(encoding='utf-8').splitlines()[0]).strip()

print('prefix', prefix)

tokenizer = AutoTokenizer.from_pretrained('hf_models/Qwen3-0.6B', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('hf_models/Qwen3-0.6B', torch_dtype=torch.float16, device_map={'':0}, trust_remote_code=True)
model.eval()

prompt = f'Сочини короткий смешной анекдот, начинающийся с фразы: "{prefix}". Пиши 2-3 предложения, без чисел. {prefix} '
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
out = model.generate(**inputs, do_sample=True, temperature=0.6, top_p=0.9, repetition_penalty=1.1, max_new_tokens=80, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
text = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip().replace('\n',' ')
print(text[:300])
