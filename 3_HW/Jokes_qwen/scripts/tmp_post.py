import re
from pathlib import Path

prefixes=[re.sub(r'^\s*\d+\s+','',l).strip() for l in Path('data/prefixes.txt').read_text(encoding='utf-8').splitlines() if l.strip()]
raw=Path('outputs/jokes_pref_raw.txt').read_text(encoding='utf-8', errors='replace')
chunks=[]
for block in raw.splitlines():
    for part in re.split(r'\\n', block):
        part=part.replace('\n',' ')
        if part.strip():
            chunks.append(part)

def clean(txt):
    txt=txt.replace('\n',' ').replace('\\n',' ')
    txt=re.sub(r'^\s*\d+\s+','',txt).strip()
    txt=re.sub(r'\d+','',txt)
    txt=re.sub(r'\s+',' ',txt)
    txt=re.split(r'(Нужно .*|Ответ:.*|Требуется .*|\(с\).*)', txt)[0].strip()
    parts=re.split(r'([.!?])', txt)
    res=[]
    for i in range(0, len(parts)-1, 2):
        s=(parts[i]+parts[i+1]).strip()
        if s:
            res.append(s)
        if len(res)>=2:
            break
    if res:
        txt=' '.join(res)
    return txt

selected=[]
used=set()
for pref in prefixes:
    pick=None
    for c in chunks:
        c2=clean(c)
        if not c2.startswith(pref):
            continue
        if len(c2)<30 or len(c2)>200:
            continue
        if c2 in used:
            continue
        pick=c2
        break
    if pick is None:
        pick=f"{pref} ... (не удалось подобрать анекдот)"
    used.add(pick)
    selected.append(pick)

out=Path('outputs/jokes_final.txt')
out.write_text('\n'.join(selected)+'\n', encoding='utf-8')
print('wrote', len(selected))
