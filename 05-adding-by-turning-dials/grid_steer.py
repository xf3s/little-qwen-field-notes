"""Arithmetic on the grid (11-04, three). Does pushing "4x" toward "5x" along little Qwen's tens direction change its sums? Laptop."""
import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText
torch.set_num_threads(8); torch.manual_seed(0)
P = '/home/three/models/qwen3.5-0.8b'; L = 11
tok = AutoTokenizer.from_pretrained(P); m = AutoModelForImageTextToText.from_pretrained(P, dtype=torch.float32).eval()
layer = m.model.language_model.layers[L]
M = torch.load('/home/three/interp/atlas/weber_L11.pt'); n = torch.arange(1, 101)
two = M[9:99]; nn = n[9:99]; tens, units = nn // 10, nn % 10
tm = {t: two[tens == t].mean(0) for t in range(1, 10)}; um = {u: two[units == u].mean(0) for u in range(10)}
tstep = torch.stack([tm[t + 1] - tm[t] for t in range(1, 9)]).mean(0)       # one tens step up
ustep = torch.stack([um[u + 1] - um[u] for u in range(1, 8)]).mean(0)       # one units step up
r = torch.randn(1024); rstep = r / r.norm() * tstep.norm()
print('step lengths: tens', round(float(tstep.norm()), 2), 'units', round(float(ustep.norm()), 2))
st = {'pos': None, 'add': None}
def hook(mod, a, o):
    if st['add'] is None: return None
    h = (o[0] if isinstance(o, tuple) else o).clone(); h[0, st['pos']] += st['add']
    return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
layer.register_forward_hook(hook)
DIG = [tok(str(d), add_special_tokens=False).input_ids[0] for d in range(10)]
def answer(a, add):
    prompt = f"Calculate: {a} + 1 = "
    ids = tok(prompt, add_special_tokens=False).input_ids
    pos = len(tok(f"Calculate: {a}", add_special_tokens=False).input_ids) - 1
    st['pos'], st['add'] = pos, add
    with torch.no_grad(): lo = m(input_ids=torch.tensor([ids])).logits[0, -1]
    t = int(lo[DIG].argmax())
    with torch.no_grad(): lo2 = m(input_ids=torch.tensor([ids + [DIG[(a + 1) // 10]]])).logits[0, -1]
    u = int(lo2[DIG].argmax()); st['add'] = None
    return t, u
cases = [10 * t + u for t in range(2, 8) for u in range(1, 8, 2)]
base = {a: answer(a, None) for a in cases}
ok = sum(base[a] == ((a + 1) // 10, (a + 1) % 10) for a in cases)
print(f'unpushed: {ok}/{len(cases)} sums right (tens and units)')
for name, v in (('tens step', tstep), ('units step', ustep), ('random', rstep)):
    for c in (1, 2, 3, 5):
        tup = sum(answer(a, c * v)[0] == base[a][0] + 1 for a in cases); tch = sum(answer(a, c * v)[0] != base[a][0] for a in cases)
        uch = sum(answer(a, c * v)[1] != base[a][1] for a in cases); uup = sum(answer(a, c * v)[1] == (base[a][1] + 1) % 10 for a in cases)
        print(f'{name:10s} x{c}: tens digit +1 in {tup}/{len(cases)} (changed {tch}); units digit +1 in {uup} (changed {uch})')
for a in (42, 63):
    for c in (0, 3, 5):
        t, u = answer(a, None if c == 0 else c * tstep)
        print(f'   {a} + 1 = ? [tens push x{c}] -> tens digit {t}, units digit (after the true tens) {u}')
