"""Is the shared position code causal? (11-04, three). Push "fourth minus first" (from ordinal sentences) into the day/month token
at layer 11 and see whether the predicted next day/month moves forward. Laptop."""
import torch, sys
from transformers import AutoTokenizer, AutoModelForImageTextToText
torch.set_num_threads(8); torch.manual_seed(0)
P = '/home/three/models/qwen3.5-0.8b'; L = 11
tok = AutoTokenizer.from_pretrained(P); m = AutoModelForImageTextToText.from_pretrained(P, dtype=torch.float32).eval()
layer = m.model.language_model.layers[L]
DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
ORD = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh']
OT = ["It was the {} time I saw her.", "She finished in {} place.", "This is my {} attempt.", "He lives on the {} floor.",
      "Turn left at the {} light.", "The {} chapter is short.", "It was their {} child.", "We sat in the {} row.",
      "That was the {} question.", "The {} day was the hardest.", "Read the {} line again.", "Our {} meeting is today."]
DT = ["Today is {}.", "See you on {}!", "The shop is closed on {}.", "I'll call you on {}.", "The deadline is next {}.",
      "Our flight leaves {} morning.", "The package arrived on {}.", "We have practice on {} evenings.", "It happened on a quiet {}.",
      "The report is due by {}.", "My favourite day is {}.", "The museum is free on {}."]
st = {'pos': None, 'add': None}
def hook(mod, a, o):
    if st['add'] is None: return None
    h = o[0] if isinstance(o, tuple) else o
    h = h.clone(); h[0, st['pos']] += st['add']
    return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
grab = {}
def g_hook(mod, a, o): grab['h'] = (o[0] if isinstance(o, tuple) else o).detach()
hg = layer.register_forward_hook(g_hook)
def raw_means(words, temps):
    X = torch.zeros(len(temps), len(words), 1024)
    for i, t in enumerate(temps):
        for j, w in enumerate(words):
            pre = t.split('{}')[0]; n = len(tok(pre + w, add_special_tokens=False).input_ids) - 1
            with torch.no_grad(): m(input_ids=torch.tensor([tok(t.format(w), add_special_tokens=False).input_ids]))
            X[i, j] = grab['h'][0, n]
    return (X - X.mean(1, keepdim=True)).mean(0)
O = raw_means(ORD, OT); D = raw_means(DAYS, DT)
hg.remove(); layer.register_forward_hook(hook)
vec = {'ordinal 4th-1st': O[3] - O[0], 'days Thu-Mon (positive control)': D[3] - D[0]}
r = torch.randn(1024); vec['random (same length)'] = r / r.norm() * (O[3] - O[0]).norm()
print('push lengths:', {k: round(float(v.norm()), 2) for k, v in vec.items()}, ' typical notepad length ~', round(float(grab['h'][0, 1:].norm(dim=-1).median()), 1))
def ids_of(words): return [tok(' ' + w, add_special_tokens=False).input_ids[0] for w in words]
CASES = [(t, d, DAYS) for t in ("Today is {}. Tomorrow is", "The day after {} is", "If today is {}, tomorrow is") for d in DAYS[:3]] + \
        [("The month after {} is", mo, MONTHS) for mo in MONTHS[:3]] + [("{} comes right before", mo, MONTHS) for mo in MONTHS[:3]]
def dist(prompt, word, words, add):
    pre = prompt.split('{}')[0]; pos = len(tok(pre + word, add_special_tokens=False).input_ids) - 1
    ids = tok(prompt.format(word), add_special_tokens=False).input_ids
    st['pos'], st['add'] = pos, add
    with torch.no_grad(): lo = m(input_ids=torch.tensor([ids])).logits[0, -1]
    st['add'] = None
    p = torch.softmax(lo[ids_of(words)], 0); return p
for name, v in vec.items():
    for c in (1, 2, 4):
        shifts, moved, mass = [], 0, []
        for prompt, word, words in CASES:
            K = len(words); p0 = dist(prompt, word, words, None); p1 = dist(prompt, word, words, c * v)
            pos = torch.arange(K).float()
            # circular expected shift relative to unpushed argmax
            a0 = int(p0.argmax()); d = ((pos - a0 + K // 2) % K) - K // 2
            shifts.append(float((p1 * d).sum() - (p0 * d).sum())); moved += int(p1.argmax() != p0.argmax())
        print(f'{name:32s} x{c}: mean shift {sum(shifts)/len(shifts):+.2f} positions, top answer changed in {moved}/{len(CASES)}')
# show one case in full
for name in ('ordinal 4th-1st',):
    for c in (0, 2, 4):
        p = dist("Today is {}. Tomorrow is", 'Monday', DAYS, None if c == 0 else c * vec[name])
        print(f'\nToday is Monday. Tomorrow is ... [{name} x{c}]: ' + ' '.join(f'{d[:3]} {float(x):.2f}' for d, x in zip(DAYS, p)))

print('\nwhere the top answer lands, ordinal push x2 (target: unpushed answer + 3):')
from collections import Counter
for name in ('ordinal 4th-1st', 'random (same length)'):
    land = Counter()
    for prompt, word, words in CASES:
        K = len(words); a0 = int(dist(prompt, word, words, None).argmax()); a1 = int(dist(prompt, word, words, 2 * vec[name]).argmax())
        land[((a1 - a0 + K // 2) % K) - K // 2] += 1
        if name.startswith('ordinal'): print(f'   {prompt.format(word)!r}: {words[a0]} -> {words[a1]}')
    print(f'  {name}: moved by ' + ', '.join(f'{k:+d}: {v}' for k, v in sorted(land.items())))
