"""A ring-shaped feature (11-04, three: "go build the ring shape >:3"). Toy dictionaries on L11 calendar-word activations:
A = 16 rays; B = 1 ring (2-D plane, angle = which day) + 14 rays. Laptop."""
import torch, math, random, itertools
from transformers import AutoTokenizer, AutoModelForImageTextToText
torch.set_num_threads(8); torch.manual_seed(0); random.seed(0)
P = '/home/three/models/qwen3.5-0.8b'; L = 11
tok = AutoTokenizer.from_pretrained(P); m = AutoModelForImageTextToText.from_pretrained(P, dtype=torch.float32).eval()
DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
A1 = ["Today is", "See you on", "The meeting moved to", "Every", "She was born on a", "The shop is closed on", "I'll call you on",
      "Last", "The deadline is next", "He feels tired on", "Our flight leaves", "The package came on", "Practice is on",
      "It happened on a quiet", "The report is due by", "My favourite is", "The festival starts this", "On", "Can we meet",
      "The class was cancelled on", "By", "I do laundry on", "The museum is free on", "They married on a"]
A2 = ["", " morning", " night"]
T = [a + " {}" + b + "." for a in A1 for b in A2]           # 72 templates
def grab(words):
    X = torch.zeros(len(T), len(words), 1024)
    for i, t in enumerate(T):
        for j, w in enumerate(words):
            pre = t.split('{}')[0]
            n = len(tok(pre + w, add_special_tokens=False).input_ids) - 1
            ids = tok(t.format(w), add_special_tokens=False).input_ids
            with torch.no_grad(): X[i, j] = m(input_ids=torch.tensor([ids]), output_hidden_states=True).hidden_states[L + 1][0, n]
    return X - X.mean(1, keepdim=True)                       # word-specific part within each template
D, M = grab(DAYS), grab(MONTHS)
torch.save({'days': D, 'months': M}, '/home/three/interp/atlas/ring_sae_data.pt')
te = list(range(0, 72, 4)); tr = [i for i in range(72) if i not in te]
def flat(X, idx): return X[idx].reshape(-1, 1024)
Xtr = torch.cat([flat(D, tr), flat(M, tr)]); scale = Xtr.norm(dim=1).mean(); Xtr = Xtr / scale
class Dict(torch.nn.Module):
    def __init__(s, rays, ring):
        super().__init__(); s.ring = ring
        s.We = torch.nn.Parameter(torch.randn(1024, rays) * .03); s.be = torch.nn.Parameter(torch.zeros(rays))
        s.Wd = torch.nn.Parameter(torch.randn(rays, 1024) * .03)
        if ring:
            s.Re = torch.nn.Parameter(torch.randn(1024, 2) * .03); s.Rd = torch.nn.Parameter(torch.randn(2, 1024) * .03)
            s.th = torch.nn.Parameter(torch.tensor(0.1))
    def forward(s, x, k=2):
        a = torch.relu(x @ s.We + s.be); top = a.topk(k, dim=1)
        a = torch.zeros_like(a).scatter(1, top.indices, top.values)
        Wd = s.Wd / s.Wd.norm(dim=1, keepdim=True); out = a @ Wd; z = None
        if s.ring:
            z = x @ s.Re; r = z.norm(dim=1, keepdim=True)
            z = z * torch.sigmoid((r - s.th) * 20)           # gate: the ring is on when its length clears a threshold
            Q, _ = torch.linalg.qr(s.Rd.T); out = out + z @ (Q.T * 1.0) if False else out + z @ s.Rd
        return out, a, z
def train(model, steps=3000):
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        out, a, z = model(Xtr); loss = ((out - Xtr) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model
def fve(model, X):
    Y = X.reshape(-1, 1024) / scale; out, _, _ = model(Y)
    return float(1 - ((out - Y) ** 2).sum() / (Y ** 2).sum())
A = train(Dict(16, False)); B = train(Dict(14, True))
print("held-out variance explained (days / months):")
for name, mod in (('A: 16 rays', A), ('B: ring + 14 rays', B)):
    print(f"  {name:20s} {fve(mod, D[te]):.3f} / {fve(mod, M[te]):.3f}")
with torch.no_grad():
    for X, names, lab in ((D, DAYS, 'days'), (M, MONTHS, 'months')):
        Y = X[te] / scale; ang = []; rad = []
        for j in range(len(names)):
            _, a, z = B(Y[:, j]); zm = z.mean(0); ang.append(math.degrees(math.atan2(zm[1], zm[0]))); rad.append(float(z.norm(dim=1).mean()))
        order = sorted(range(len(names)), key=lambda j: ang[j]); i0 = order.index(0); cyc = order[i0:] + order[:i0]
        exact = cyc == list(range(len(names))) or cyc == [0] + list(range(len(names) - 1, 0, -1))
        print(f"\nB's ring on {lab}: exact order {exact}; around the ring: {' '.join(names[c][:3] for c in cyc)}")
        print("  angle per word: " + "  ".join(f"{names[j][:3]} {ang[j]:+.0f}" for j in range(len(names))))
        print("  ring length per word: " + "  ".join(f"{names[j][:3]} {rad[j]:.2f}" for j in range(len(names))))
    # what B's rays do: mean activation per day/month
    Y = torch.cat([D[te], M[te]], 1) / scale; names = DAYS + MONTHS
    _, a, _ = B(Y.reshape(-1, 1024)); a = a.reshape(len(te), len(names), -1).mean(0)
    _, a2, _ = A(Y.reshape(-1, 1024)); a2 = a2.reshape(len(te), len(names), -1).mean(0)
    for lab, act in (("A's rays", a2), ("B's rays", a)):
        print(f"\n{lab}: which words each ray fires on (mean >= 30% of its max)")
        for f in range(act.shape[1]):
            col = act[:, f]
            if col.max() < 1e-3: print(f"  ray {f:2d}: (dead)"); continue
            on = [names[j][:3] for j in range(len(names)) if col[j] >= .3 * col.max()]
            print(f"  ray {f:2d}: {' '.join(on)}")
torch.save({'A': A.state_dict(), 'B': B.state_dict(), 'scale': scale}, '/home/three/interp/atlas/ring_sae.pt')
