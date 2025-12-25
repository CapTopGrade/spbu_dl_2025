import os, re, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id='Qwen/Qwen2-0.5B-Instruct'
print('loading', model_id)
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
tokenizer=AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=token)
model=AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map={'':0}, trust_remote_code=True, token=token)
model.eval()

prefixes=[re.sub(r'^\s*\d+\s+','',l).strip() for l in Path('data/prefixes.txt').read_text(encoding='utf-8').splitlines() if l.strip()]

def build(prefix):
    return f"Ты — комик. Придумай короткий смешной анекдот на русском языке, начни строго с фразы: \"{prefix}\". 2–3 предложения, без цифр и списков, закончи мысль."

out_lines=[]
for pref in prefixes:
    best=None
    for _ in range(6):
        prompt=build(pref)
        inputs=tokenizer(prompt, return_tensors='pt').to(model.device)
        gen_tokens=model.generate(**inputs, do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.05, max_new_tokens=96, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
        text=tokenizer.decode(gen_tokens[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        text=text.replace('\n',' ').replace('\r',' ').strip()
        text=re.sub(r'\d+','', text)
        parts=re.split(r'([.!?])', text)
        res=[]
        for i in range(0,len(parts)-1,2):
            s=(parts[i]+parts[i+1]).strip()
            if s:
                res.append(s)
            if len(res)>=2:
                break
        if res:
            text=' '.join(res)
        if not text.startswith(pref):
            text=f"{pref} {text}"
        if len(text)<30 or len(text)>200:
            continue
        if '...' in text:
            continue
        best=text
        break
    if best is None:
        best=f"{pref} ... (не удалось сгенерировать анекдот)"
    out_lines.append(best)

Path('outputs/jokes_instruct.txt').write_text('\n'.join(out_lines)+'\n', encoding='utf-8')
print('written', len(out_lines))

