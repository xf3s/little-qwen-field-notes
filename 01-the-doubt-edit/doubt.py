"""Where is "the next line will reconsider" decided? (11-02 night, after n998's flip plateaued: Wait gone, doubt not.)
At line starts in thinking transcripts, label each by the line that follows: RECONSIDER (Wait, Hold on, However, But, Maybe,
Is there, What if, Hmm, Actually, Perhaps, Double-check, re-read...) vs COMPUTE (starts with $, a digit, \\, =, or a bullet
followed by math). Take the notepad after every layer at those positions; per layer, the difference of class means is a
"doubt direction"; how well a held-out half separates along it says where the decision is readable.
    python doubt.py RUNS_FILE N [N ...]     (thinking runs)  -> doubt_dirs.pt, prints separability per layer
"""
import json, re, sys, torch
import probe
from probe import tok, model, layers

RECON = re.compile(r"^\s*[-*]*\s*\**\s*(wait|hold on|however|but\b|maybe|is there|is it possible|what if|hmm|actually|perhaps|"
                   r"double[- ]check|let me re|let's re|re-?read|alternatively|or maybe|no[,.]|hang on|could it)", re.I)
COMP = re.compile(r"^\s*[-*]*\s*(\$|\\\(|\\\[|\d|=)")

def collect(problem, text, maxlen=2500):
    h, t = probe.build(problem, text, True)
    ids = torch.tensor([(h + t)[:maxlen]])
    with torch.no_grad(): out = model(input_ids=ids, output_hidden_states=True)
    hs = torch.stack(out.hidden_states[1:])[:, 0]                      # [24, T, 1024]: after each layer
    ids0 = ids[0].tolist()
    X, y, T = [], [], []
    for p in range(len(h), len(ids0) - 1):
        if "\n" not in tok.decode([ids0[p]]): continue
        nxt = tok.decode(ids0[p + 1:p + 12]).split("\n")[0]
        if not nxt.strip(): continue
        lab = 1 if RECON.match(nxt) else 0 if COMP.match(nxt) else None
        if lab is not None: X.append(hs[:, p]); y.append(lab); T.append(ids0[p])
    return X, y, T

if __name__ == "__main__":
    f = sys.argv[1]; ns = [int(n) for n in sys.argv[2:]]
    runs = json.load(open(f)); probs = [json.loads(l) for l in open("/home/three/interp/wsp/math500.jsonl")]
    X, y, src, tid = [], [], [], []
    for n in ns:
        r = runs[n]; a, b, t = collect(probs[r["idx"]]["problem"], r["think"])
        X += a; y += b; src += [n] * len(a); tid += t
        print(f"run {n} (problem {r['idx']}): {sum(b)} reconsider, {len(b) - sum(b)} compute line starts", flush=True)
    X = torch.stack(X); y = torch.tensor(y); src = torch.tensor(src); tid = torch.tensor(tid)
    # the line-start token itself (indentation) can give the class away; keep only the commonest newline token
    top = tid.mode().values.item(); keep = tid == top
    X, y, src = X[keep], y[keep], src[keep]
    print(f"kept token {tok.decode([top])!r}: {int(keep.sum())} of {len(keep)}; test run: {int((y[src == ns[-1]] == 1).sum())} reconsider, "
          f"{int((y[src == ns[-1]] == 0).sum())} compute; majority baseline "
          f"{max((y[src == ns[-1]] == 1).float().mean().item(), (y[src == ns[-1]] == 0).float().mean().item()):.2f}")
    # train on all runs but the last, test on the last (a different problem)
    tr, te = src != ns[-1], src == ns[-1]
    dirs = []
    print("layer  train-sep  held-run-acc")
    for L in range(X.shape[1]):
        Z = X[:, L]
        mu1, mu0 = Z[tr & (y == 1)].mean(0), Z[tr & (y == 0)].mean(0)
        d = mu1 - mu0; dirs.append(d)
        s = Z @ d; thr = ((mu1 + mu0) / 2) @ d
        acc = ((s[te] > thr) == (y[te] == 1)).float().mean().item()
        sep = (s[tr & (y == 1)].mean() - s[tr & (y == 0)].mean()) / s[tr].std()
        print(f"L{L:<4d}  {sep.item():8.2f}  {acc:8.2f}")
    import os; torch.save({"dirs": torch.stack(dirs), "n": len(y), "pos": int(y.sum())}, os.environ.get("OUT", "/home/three/interp/wsp/doubt_dirs.pt"))
