"""Grid arithmetic, part 2 (11-04, three): push at the tens-digit token instead of the last token. Laptop."""
import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText
torch.set_num_threads(8); torch.manual_seed(0)
P = '/home/three/models/qwen3.5-0.8b'; L = 11
tok = AutoTokenizer.from_pretrained(P); m = AutoModelForImageTextToText.from_pretrained(P, dtype=torch.float32).eval()
layer = m.model.language_model.layers[L]
grab = {}; gh = layer.register_forward_hook(lambda mod, a, o: grab.__setitem__('h', (o[0] if isinstance(o, tuple) else o).detach()))
NT = ["I have {} cats.", "We waited {} hours.", "She bought {} apples.", "It took {} days.", "He ran {} miles.", "I read {} books."]
def first_digit_pos(prefix, a): return len(tok(prefix, add_special_tokens=False).input_ids) + len(tok(' ', add_special_tokens=False).input_ids) - 1 + 1
reps = {}
for t in range(2, 9):
    vs = []
    for tpl in NT:
        pre = tpl.split('{}')[0]
        for u in (1, 3, 5, 7):
            a = 10 * t + u; ids = tok(tpl.format(a), add_special_tokens=False).input_ids
            pos = len(tok(pre + str(a)[0], add_special_tokens=False).input_ids) - 1
            assert tok.decode([ids[pos]]) == str(t), (tok.decode([ids[pos]]), t)
            with torch.no_grad(): m(input_ids=torch.tensor([ids]))
            vs.append(grab['h'][0, pos])
    reps[t] = torch.stack(vs).mean(0)
gh.remove()
tstep = torch.stack([reps[t + 1] - reps[t] for t in range(2, 8)]).mean(0)
M = torch.load('/home/three/interp/atlas/weber_L11.pt'); dstep = torch.stack([M[d] - M[d - 1] for d in range(2, 8)]).mean(0)
r = torch.randn(1024); rstep = r / r.norm() * tstep.norm()
print('step lengths: tens-token', round(float(tstep.norm()), 2), ' single-digit', round(float(dstep.norm()), 2))
st = {'pos': None, 'add': None}
def hook(mod, a, o):
    if st['add'] is None: return None
    h = (o[0] if isinstance(o, tuple) else o).clone(); h[0, st['pos']] += st['add']
    return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
layer.register_forward_hook(hook)
DIG = [tok(str(d), add_special_tokens=False).input_ids[0] for d in range(10)]
def answer(a, add):
    prompt = f"Calculate: {a} + 1 = "; ids = tok(prompt, add_special_tokens=False).input_ids
    pos = len(tok(f"Calculate: {str(a)[0]}", add_special_tokens=False).input_ids) - 1
    assert tok.decode([ids[pos]]) == str(a)[0]
    st['pos'], st['add'] = pos, add
    with torch.no_grad(): lo = m(input_ids=torch.tensor([ids])).logits[0, -1]
    t = int(lo[DIG].argmax())
    with torch.no_grad(): lo2 = m(input_ids=torch.tensor([ids + [DIG[t]]])).logits[0, -1]
    st['add'] = None; return t, int(lo2[DIG].argmax())
cases = [10 * t + u for t in range(2, 8) for u in range(1, 8, 2)]
base = {a: answer(a, None) for a in cases}
print(f'unpushed: {sum(base[a] == ((a + 1) // 10, (a + 1) % 10) for a in cases)}/{len(cases)} right')
for name, v in (('tens-token step', tstep), ('single-digit step', dstep), ('random', rstep)):
    for c in (1, 2, 4, 8):
        res = {a: answer(a, c * v) for a in cases}
        up = sum(res[a][0] == base[a][0] + 1 for a in cases); ch = sum(res[a][0] != base[a][0] for a in cases)
        full = sum(res[a] == (base[a][0] + 1, base[a][1]) for a in cases)
        print(f'{name:18s} x{c}: tens digit +1 in {up}/{len(cases)} (changed {ch}); whole answer exactly +10 in {full}')
for a in (42, 63):
    print('  ', a, '+ 1 =', ' | '.join(f'x{c}: {"".join(map(str, answer(a, None if c == 0 else c * tstep)))}' for c in (0, 2, 4, 8)))
