"""Score dial12_*.pt: Mantel correlation of the 12x12 (or 7x7) distance matrix with ring / line / fifths models, per layer."""
import sys, torch
D = torch.load(sys.argv[1]); g = torch.Generator().manual_seed(1106)
def ring(n, order=None):
    pos = torch.arange(n) if order is None else torch.tensor(order)
    d = (pos[:, None] - pos[None, :]).abs(); return torch.minimum(d, n - d).float()
def line(n): p = torch.arange(n); return (p[:, None] - p[None, :]).abs().float()
def mantel(A, B, perms=2000):
    n = A.shape[0]; iu = torch.triu_indices(n, n, 1)
    a = A[iu[0], iu[1]]; b = B[iu[0], iu[1]]
    r = float(torch.corrcoef(torch.stack([a, b]))[0, 1]); hit = 0
    for _ in range(perms):
        p = torch.randperm(n, generator=g); bp = B[p][:, p][iu[0], iu[1]]
        hit += float(torch.corrcoef(torch.stack([a, bp]))[0, 1]) >= r
    return r, (hit + 1) / (perms + 1)
def dist(X):
    X = X - X.mean(0); return torch.cdist(X, X)
models = {"months": {"ring": ring(12), "line": line(12)}, "hours": {"ring": ring(12), "line": line(12)}}
# naturals only: C D E F G A B -> semitone positions 0 2 4 5 7 9 11; fifths positions (x7 mod 12, as ring of 12)
nat = [0, 2, 4, 5, 7, 9, 11]
semi = ring(12)[nat][:, nat]; fifths = ring(12, [(p * 7) % 12 for p in range(12)])[nat][:, nat]
for dom in ("months", "hours", "naturals"):
    X = D["notes"][nat] if dom == "naturals" else D[dom]
    mods = {"semitone ring": semi, "fifths ring": fifths} if dom == "naturals" else models[dom]
    best = {}
    for L in range(X.shape[1]):
        dm = dist(X[:, L].float())
        for name, M in mods.items():
            r, p = mantel(dm, M, perms=500)
            best.setdefault(name, []).append((r, p, L))
    print(f"== {dom}")
    for name, rows in best.items():
        r, p, L = max(rows); npass = sum(pp < 0.01 for _, pp, _ in rows)
        print(f"   {name:14s} best r {r:+.2f} at L{L + 1} (p {p:.3f}); layers with p < 0.01: {npass}/{len(rows)}")
