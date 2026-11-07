"""Shape-aware Matryoshka BatchTopK SAE (11-04, three's design: "allow for circles and stuff in the prev matryoshka layers too,
like, the week would go layer 0 as 2 directions"). Each latent group owns d >= 1 decoder directions. A 1-D group is an ordinary
ray (ReLU). A d-D group is a small plane or space: its encoder gives a d-vector z, its strength is |z|, and when it is chosen
it writes z through its d decoder rows, so where it points (its angle) carries meaning: the day of the week, say.
BatchTopK spends a budget of k directions per token: a d-D group costs d. Matryoshka: losses on nested prefixes of the columns.
Pure torch, no Modal here; the same module trains on the laptop and in the cloud."""
import torch

def layout(n_dirs=16384, tier0=1024, groups0=((2, 128), (4, 32))):
    """Group dims in column order: tier 0 = the multi-D groups then rays, then rays to n_dirs."""
    dims = []
    for d, n in groups0: dims += [d] * n
    dims += [1] * (tier0 - sum(dims)) + [1] * (n_dirs - tier0)
    return dims

class ShapeSAE(torch.nn.Module):
    def __init__(self, d_model, dims, prefixes, k):
        super().__init__()
        self.dims, self.prefixes, self.k = list(dims), list(prefixes), k
        D = sum(dims); self.D = D
        gid = torch.repeat_interleave(torch.arange(len(dims)), torch.tensor(dims))
        self.register_buffer('gid', gid)                                     # [D] group of each column
        self.register_buffer('gdim', torch.tensor(dims, dtype=torch.float32)[gid])   # [D] its group's dim
        self.register_buffer('multi', (self.gdim > 1))
        W = torch.randn(D, d_model); W = W / W.norm(dim=1, keepdim=True)
        self.W_dec = torch.nn.Parameter(W.clone())
        self.W_enc = torch.nn.Parameter(W.T.clone() * 0.1)
        self.b_enc = torch.nn.Parameter(torch.zeros(D)); self.b_dec = torch.nn.Parameter(torch.zeros(d_model))

    def dec(self): return self.W_dec / self.W_dec.norm(dim=1, keepdim=True)

    def encode(self, x, k=None):
        k = k or self.k
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc                    # [B, D]
        G = len(self.dims)
        sq = torch.zeros(x.shape[0], G, device=x.device).index_add_(1, self.gid, pre * pre)
        strength_multi = sq.clamp_min(1e-12).sqrt()[:, self.gid]            # |z| of the column's group
        s = torch.where(self.multi, strength_multi / self.gdim.sqrt(), torch.relu(pre))   # per-direction score
        # BatchTopK on per-direction scores: a d-D group shows up as d columns with the same score, so it costs d
        n = min(k * x.shape[0], s.numel())
        thr = s.flatten().topk(n).values[-1]
        on_col = s >= thr
        on_grp = torch.zeros(x.shape[0], G, device=x.device).index_add_(1, self.gid, on_col.float()) > 0
        on = on_grp[:, self.gid] & (s > 0)
        a = torch.where(self.multi, pre, torch.relu(pre)) * on
        return a, on

    def forward(self, x):
        a, on = self.encode(x)
        W = self.dec()
        outs = [a[:, :p] @ W[:p] + self.b_dec for p in self.prefixes]        # Matryoshka: nested reconstructions
        return outs, a, on

    def loss(self, x):
        outs, a, on = self(x)
        return sum(((o - x) ** 2).sum(1).mean() for o in outs) / len(outs), outs[-1], a, on

def fve(x, xhat): return float(1 - ((xhat - x) ** 2).sum() / ((x - x.mean(0)) ** 2).sum())
