"""The needle-lifter neuron's vectors (11-03, three: "do you need to clobber what's already on that neuron? can you add at
right angles again?" -> a brand-new neuron in layer 14, nothing clobbered). It reads "same again" (the overlap of the
public and thinking SAEs' feature #807 at L11) and writes the model's own "want out" direction (stuck - compute at L13,
which the tuned lens reads as end / boxed / <|im_end|>). The MLP sees RMSNorm(h) * w, so rows are divided by w.
  gate = beta * (r - c * mu) / w   fires only well above ordinary lines (c*mu acts as the bias Qwen's MLPs don't have)
  up   = mu / (m_med * w)          ~1 on every token
  down = |stuck - comp| * e / act_stuck   (scaled by the run's implant factor)
"""
import torch
d = torch.load("/home/three/interp/wsp/implant_dirs.pt"); r, mu, w = d["r"], d["mu"], d["w14"]
g = torch.load("/home/three/interp/wsp/groove_v2.pt"); M = g["means"]; K = g["keep"]
e = M["stuck"][13] - M["comp"][13]; size = e.norm(); e = e / size
nrm = lambda X: X * torch.rsqrt(X.pow(2).mean(-1, keepdim=True) + 1e-6)
beta, c = 2.0, 0.18
m_med = float(torch.cat([nrm(K[k][:, 13]) for k in ("comp", "noodle")]) @ mu).__class__(0) if False else float((torch.cat([nrm(K[k][:, 13]) for k in ("comp", "noodle")]) @ mu).median())
gate = beta * (r - c * mu) / w; up = mu / (m_med * w)
silu = torch.nn.functional.silu
def act(X): x = nrm(X) * w; return silu(x @ gate) * (x @ up)
a = {k: act(K[k][:, 13]) for k in ("stuck", "comp", "noodle")}
for k, v in a.items(): print(f"{k:7s} activation 5/50/95%: {[round(float(v.quantile(p)), 2) for p in (0.05, 0.5, 0.95)]}")
act_stuck = float(a["stuck"].median())
down = size * e / act_stuck
torch.save({"gate": gate, "up": up, "down": down}, "/home/three/interp/wsp/implant_vectors.pt")
print(f"saved; at factor 1 a typical stuck moment gets +{float(size):.2f} along 'want out' (doubling the model's own)")
