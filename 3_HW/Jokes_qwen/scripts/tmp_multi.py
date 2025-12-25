import torch, re, math, random
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

prefix = re.sub(r'^\s*\d+\s+','', Path('data/prefixes.txt').read_text(encoding='utf-8').splitlines()[0]).strip()
print('prefix', prefix)

tokenizer = AutoTokenizer.from_pretrained('model_new', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('hf_models/Qwen3-0.6B', torch_dtype=torch.float16, device_map={'':0}, trust_remote_code=True)
model = PeftModel.from_pretrained(model, 'model_new')
model = model.merge_and_unload()
model.eval()

prompt = (
    f'Инструкция: Начни анекдот с фразы: "{prefix}". '
    'Напиши смешной, логичный анекдот на русском языке без оскорблений. '
    'Сделай 2-4 предложения, избегай списков и дат, заверши мысль. '
    f'Ответ: {prefix} '
)
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

for i in range(6):
    out = model.generate(**inputs, do_sample=True, temperature=0.8, top_p=0.95, repetition_penalty=1.12, max_new_tokens=80, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
    gen = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    print(f'\nSample {i+1}: {gen}')
