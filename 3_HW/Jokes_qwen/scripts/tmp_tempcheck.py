import re, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

prefix = re.sub(r'^\s*\d+\s+', '', Path('data/prefixes.txt').read_text(encoding='utf-8').splitlines()[0]).strip()

print('prefix', prefix)

tokenizer = AutoTokenizer.from_pretrained('model_new_plain_full/checkpoint-2120', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('hf_models/Qwen3-0.6B', torch_dtype=torch.float16, device_map={'':0}, trust_remote_code=True)
model = PeftModel.from_pretrained(model, 'model_new_plain_full/checkpoint-2120')
model = model.merge_and_unload(); model.eval()

prompt = (
    f'Инструкция: Начни анекдот с фразы: "{prefix}". '
    'Пиши смешной, понятный анекдот без чисел и списков. 2-3 предложения, завершай мысль. '
    f'Ответ: {prefix} '
)
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
for t in [0.4, 0.5]:
    out = model.generate(
        **inputs,
        do_sample=True,
        temperature=t,
        top_p=0.85,
        repetition_penalty=1.15,
        max_new_tokens=60,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip().replace('\n', ' ')
    if not text.startswith(prefix):
        text = f'{prefix} {text}'
    cleaned = re.sub(r'\s+', ' ', text)
    print(f'\nT={t}:', cleaned)
