import re, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

prefix = re.sub(r'^\s*\d+\s+', '', Path('data/prefixes.txt').read_text(encoding='utf-8').splitlines()[0]).strip()

model_id='hf_models/Qwen3-0.6B'
tokenizer=AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model=AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map={'':0}, trust_remote_code=True)
model.eval()

fewshot = (
"Начало: 'Приходит программа в бар'\nАнекдот: Приходит программа в бар и просит налить памяти. Бармен говорит: памяти хватит всем, но зависать нельзя.\n\n"
"Начало: 'Жена говорит мужу'\nАнекдот: Жена говорит мужу, что у него три руки. Он посмотрел на штаны и понял, что вилка лишняя.\n\n"
)

prompt = (
    fewshot +
    f"Начало: '{prefix}'\n"
    "Анекдот: "
)

inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
out = model.generate(**inputs, do_sample=True, temperature=0.3, top_p=0.7, repetition_penalty=1.05, max_new_tokens=60, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
text = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
print(text)
