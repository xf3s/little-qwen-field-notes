"""The self ray (11-05, three): where little Qwen's next thinking line talks about itself (I / My) vs about the thing
(The / This / It), with doubt lines excluded from both. Same position and method as doubt.py, so cosines compare.
    python selfray.py   -> self_dirs.pt, separability per layer, cos with the doubt direction"""
import json, re, torch
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")

SELF = re.compile(r"^\s*[-*]*\s*\**\s*(I\b|I'm|I'll|I've|I'd|My\b|Me\b)")
OTHER = re.compile(r"^\s*[-*]*\s*\**\s*(The\b|This\b|It\b|That\b|These\b|There\b)")
RECON = re.compile(r"(wait|hold on|however|but\b|maybe|is there|possible|what if|hmm|actually|perhaps|double[- ]check|"
                   r"re-?read|alternatively|mistake|wrong|confus|unsure|not sure|check)", re.I)

def _unused_collect(problem, text, maxlen=2500):
    h, t = probe.build(problem, text, True)
    ids = torch.tensor([(h + t)[:maxlen]])
    with torch.no_grad(): out = model(input_ids=ids, output_hidden_states=True)
    hs = torch.stack(out.hidden_states[1:])[:, 0]
    ids0 = ids[0].tolist(); X, y, T = [], [], []
    for p in range(len(h), len(ids0) - 1):
        if "\n" not in tok.decode([ids0[p]]): continue
        nxt = tok.decode(ids0[p + 1:p + 14]).split("\n")[0]
        if not nxt.strip() or RECON.search(nxt): continue
        lab = 1 if SELF.match(nxt) else 0 if OTHER.match(nxt) else None
        if lab is not None: X.append(hs[:, p]); y.append(lab); T.append(ids0[p])
    return X, y, T

import sys
if "--from" in sys.argv:
    D = torch.load(sys.argv[sys.argv.index("--from") + 1]); X = D["X"].float(); y, src, tid = D["y"], D["src"], D["tid"]
    X = list(X)
else:
    runs = [r for r in json.load(open("runs_cloud_base_16k.json")) if r["split"] == "dev"]
    probs = [json.loads(l) for l in open("math500.jsonl")]
    X, y, src, tid = [], [], [], []
    for n, r in enumerate(runs):
        a, b, t = collect(probs[r["idx"]]["problem"], r["think"])
        X += a; y += b; src += [r["idx"]] * len(a); tid += t
        if n % 10 == 0: print(f"run {n}/{len(runs)}: {sum(y)} self, {len(y) - sum(y)} other so far", flush=True)
X = torch.stack(X); y = torch.tensor(y); src = torch.tensor(src); tid = torch.tensor(tid)
top = tid.mode().values.item(); keep = tid == top; X, y, src = X[keep], y[keep], src[keep]
probs_with_self = sorted({int(s) for s in src[y == 1]})
held = probs_with_self[-3:]                       # hold out the last 3 problems that have self lines
te = torch.isin(src, torch.tensor(held)); tr = ~te
print(f"kept {tok.decode([top])!r}: {int(y.sum())} self / {int((y == 0).sum())} other; held problems {held}: "
      f"{int(y[te].sum())} self / {int((y[te] == 0).sum())} other")
doubt = torch.load("doubt_dirs_both.pt")["dirs"].float()
dirs = []
print("layer  sep   held-bal-acc  cos(self, doubt)")
for L in range(X.shape[1]):
    Z = X[:, L].float(); mu1, mu0 = Z[tr & (y == 1)].mean(0), Z[tr & (y == 0)].mean(0)
    d = mu1 - mu0; dirs.append(d); s = Z @ d; thr = ((mu1 + mu0) / 2) @ d
    pred = s > thr
    bal = 0.5 * ((pred[te & (y == 1)]).float().mean() + (~pred[te & (y == 0)]).float().mean()).item()
    sep = ((s[tr & (y == 1)].mean() - s[tr & (y == 0)].mean()) / s[tr].std()).item()
    c = (d @ doubt[L] / (d.norm() * doubt[L].norm())).item()
    print(f"L{L:<4d} {sep:5.2f}  {bal:8.2f}      {c:+.3f}")
torch.save({"dirs": torch.stack(dirs), "n": len(y), "self": int(y.sum()), "held": held}, "self_dirs.pt")
