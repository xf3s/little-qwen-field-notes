"""Sums by rotation (11-04, three: "yes, let's try rotation!!"). Turn little Qwen's number circles by one notch at the units-digit
token and see whether "42 + 1" answers as if it were 43 + 1. Laptop."""
import torch, math
from transformers import AutoTokenizer, AutoModelForImageTextToText
torch.set_num_threads(8); torch.manual_seed(0)
P = '/home/three/models/qwen3.5-0.8b'; L = 11
tok = AutoTokenizer.from_pretrained(P); m = AutoModelForImageTextToText.from_pretrained(P, dtype=torch.float32).eval()
layer = m.model.language_model.layers[L]
M = torch.load('/home/three/interp/atlas/weber_L11.pt'); n = torch.arange(1, 101).float()
PER = (2, 5, 10, 100)
def feats(x):
    cols = [x / 100, (x < 10).float(), (x >= 100).float(), torch.ones_like(x)]
    for T in PER: cols += [torch.cos(2 * math.pi * x / T), torch.sin(2 * math.pi * x / T)]
    return torch.stack(cols, 1)
W = torch.linalg.lstsq(feats(n), M).solution                    # [features, 1024]
circ = {T: (W[4 + 2 * i], W[5 + 2 * i]) for i, T in enumerate(PER)}
print('circle amplitudes (direction lengths):', {T: round(float((c.norm() + s.norm()) / 2), 3) for T, (c, s) in circ.items()})
def turn(a, notches, periods):
    d = torch.zeros(1024)
    for T in periods:
        wc, ws = circ[T]; th = 2 * math.pi * a / T; dl = 2 * math.pi * notches / T
        d += wc * (math.cos(th + dl) - math.cos(th)) + ws * (math.sin(th + dl) - math.sin(th))
    return d
st = {'pos': None, 'add': None}
def hook(mod, a, o):
    if st['add'] is None: return None
    h = (o[0] if isinstance(o, tuple) else o).clone(); h[0, st['pos']] += st['add']
    return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
layer.register_forward_hook(hook)
DIG = [tok(str(d), add_special_tokens=False).input_ids[0] for d in range(10)]
def answer(a, add):
    ids = tok(f"Calculate: {a} + 1 = ", add_special_tokens=False).input_ids
    st['pos'], st['add'] = len(tok(f"Calculate: {a}", add_special_tokens=False).input_ids) - 1, add
    with torch.no_grad(): t = int(m(input_ids=torch.tensor([ids])).logits[0, -1][DIG].argmax())
    with torch.no_grad(): lo = m(input_ids=torch.tensor([ids + [DIG[(a + 1) // 10]]])).logits[0, -1][DIG]
    st['add'] = None
    return t, int(lo.argmax()), torch.softmax(lo, 0)
cases = [10 * t + u for t in range(2, 8) for u in range(1, 8, 2)]
base = {a: answer(a, None) for a in cases}
print(f'unpushed: {sum(base[a][:2] == ((a + 1) // 10, (a + 1) % 10) for a in cases)}/24 sums right')
conds = {'A: period-10 only, +1': lambda a: turn(a, 1, (10,)), 'B: all circles, +1': lambda a: turn(a, 1, PER),
         'C: all circles, -1': lambda a: turn(a, -1, PER)}
def rnd(a):
    v = torch.randn(1024, generator=torch.Generator().manual_seed(a)); return v / v.norm() * turn(a, 1, PER).norm()
conds['D: random, same length as B'] = rnd
print('push length B (x1), mean:', round(float(torch.stack([turn(a, 1, PER).norm() for a in cases]).mean()), 3))
want = {'A': 1, 'B': 1, 'C': -1, 'D': 1}
for name, f in conds.items():
    for c in (1, 2, 4, 8):
        res = [(a, answer(a, c * f(a))) for a in cases]
        hit = sum(r[1] == (base[a][1] + want[name[0]]) % 10 for a, r in res)
        ch = sum(r[1] != base[a][1] for a, r in res); tch = sum(r[0] != base[a][0] for a, r in res)
        pmove = sum(float(r[2][(base[a][1] + want[name[0]]) % 10] - base[a][2][(base[a][1] + want[name[0]]) % 10]) for a, r in res) / 24
        print(f'{name:30s} x{c}: units {"+1" if want[name[0]] > 0 else "-1"} in {hit}/24 (changed {ch}); tens changed {tch}; '
              f'mean p(target digit) {pmove:+.3f}', flush=True)
for a in (42, 63):
    for c in (0, 2, 4, 8):
        t, u, _ = answer(a, None if c == 0 else c * turn(a, 1, PER))
        print(f'   {a} + 1 = ? [all circles +1 notch x{c}] -> {t}{u}')
