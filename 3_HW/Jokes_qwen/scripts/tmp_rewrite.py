import random, re, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

prefix = re.sub(r'^\s*\d+\s+', '', Path('data/prefixes.txt').read_text(encoding='utf-8').splitlines()[0]).strip()
joke = random.choice(Path('data/jokes_clean_strict.txt').read_text(encoding='utf-8').splitlines())
print('prefix', prefix)
print('src joke:', joke)

model_id='hf_models/Qwen3-0.6B'
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map={'':0}, trust_remote_code=True)
model.eval()

prompt = (
    f'Перепиши анекдот, сделай его короче (2-3 предложения) и начни строго с фразы "{prefix}". '
    'Сохрани юмор, не используй цифры. Вот анекдот: ' + joke + '\nОтвет: '
)
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
out = model.generate(**inputs, do_sample=True, temperature=0.5, top_p=0.9, repetition_penalty=1.1, max_new_tokens=80, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
text = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
print('rewritten:', text)
