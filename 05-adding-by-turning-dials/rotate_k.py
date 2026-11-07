"""Other sums by turning (11-05 01:10). Sums by rotation (11-04, three: "yes, let's try rotation!!"). Turn little Qwen's number circles by one notch at the units-digit
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
cases = [10 * t + u for t in range(2, 8) for u in (1, 2, 3)]
base = {a: answer(a, None) for a in cases}
print(f'unpushed: {sum(base[a][:2] == ((a + 1) // 10, (a + 1) % 10) for a in cases)}/18 sums right', flush=True)
def rnd(a, k, ps):
    v = torch.randn(1024, generator=torch.Generator().manual_seed(a)); return v / v.norm() * turn(a, k, ps).norm()
for name, ps in (('{2,5}', (2, 5)), ('all four', PER)):
    for k in (1, 2, 3, -2):
        row = []
        for c in (1, 2):
            res = [(a, answer(a, c * turn(a, k, ps))) for a in cases]
            hit = sum(r[1] == (a + 1 + k) % 10 for a, r in res); ch = sum(r[1] != base[a][1] for a, r in res)
            got = ''.join(str(r[1]) for a, r in res[:6])
            row.append(f'x{c}: {hit:2d}/18 (chg {ch:2d}; units for 21..23,31..33: {got})')
        print(f'{name:8s} k={k:+d} | ' + ' | '.join(row), flush=True)
for c in (1, 2):
    res = [(a, answer(a, c * rnd(a, 2, (2, 5)))) for a in cases]
    print(f'random (k=+2 length) x{c}: {sum(r[1] == (a + 3) % 10 for a, r in res)}/18 (chg {sum(r[1] != base[a][1] for a, r in res)})', flush=True)
for k in (1, 2, 3):
    print(f'   42 + 1 = ? [{{2,5}} turned {k} notches x2] -> ' + ''.join(map(str, answer(42, 2 * turn(42, k, (2, 5)))[:2])))
