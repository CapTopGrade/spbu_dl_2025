import re, random, json
from pathlib import Path

raw = Path('data/jokes_clean_strict.txt').read_text(encoding='utf-8').splitlines()
prefixes = [re.sub(r'^\s*\d+\s+','',l).strip() for l in Path('data/prefixes.txt').read_text(encoding='utf-8').splitlines() if l.strip()]
random.seed(42)

def truncate_sentences(text, max_sent=3):
    parts = re.split(r'([.!?])', text)
    res = []
    for i in range(0, len(parts)-1, 2):
        s = (parts[i] + parts[i+1]).strip()
        if s:
            res.append(s)
        if len(res) >= max_sent:
            break
    return ' '.join(res) if res else text

records = []
for joke in raw:
    joke = re.sub(r'\d+', '', joke)
    joke = re.sub(r'\s+', ' ', joke).strip()
    if len(joke) < 40 or len(joke) > 200:
        continue
    joke = truncate_sentences(joke, max_sent=3)
    prefix = random.choice(prefixes)
    response = joke
    if not response.lower().startswith(prefix.lower()):
        response = f"{prefix} {response}"
    prompt = (
        f"Начни анекдот с фразы: \"{prefix}\". "
        "Напиши смешной, логичный анекдот на русском языке без оскорблений и цифр. "
        "Сделай 2-3 предложения, завершай мысль."
    )
    records.append({"prompt": prompt, "response": response})

random.shuffle(records)
out_path = Path('data/sft_prefix_v2.jsonl')
out_path.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in records) + '\n', encoding='utf-8')
print('records', len(records), 'saved to', out_path)
