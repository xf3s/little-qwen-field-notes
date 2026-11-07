"""Compare the 8-run and 30-run doubt directions (11-03, three: "can we compare the two directions?").
Per-layer cosine; then at L13 split d30 into its part along d8 and the rest, and read each through the unembedding
(logit lens: final norm weight folded in) to see which words each one leans toward."""
import torch
from transformers import AutoTokenizer
from safetensors import safe_open
import glob
M = "/home/three/models/qwen3.5-0.8b"
tok = AutoTokenizer.from_pretrained(M)
a = torch.load("doubt_dirs_8runs.pt")["dirs"]; b = torch.load("doubt_dirs_30.pt")["dirs"]
print("per-layer cosine (8-run vs 30-run):", " ".join(f"L{i}:{torch.nn.functional.cosine_similarity(a[i], b[i], 0):.2f}" for i in range(24)))
f = glob.glob(M + "/*.safetensors")[0]
with safe_open(f, "pt") as s:
    E = s.get_tensor("model.language_model.embed_tokens.weight").float()
    g = s.get_tensor("model.language_model.norm.weight").float()
g = 1 + g if g.mean() < 0.5 else g   # Qwen3.5's norm is (1 + w)
def lens(v, n=12):
    v = v / v.norm(); lo = E @ (v * g)
    top = [tok.decode([i]) for i in lo.topk(n).indices]; bot = [tok.decode([i]) for i in (-lo).topk(n).indices]
    return top, bot
d8, d30 = a[13] / a[13].norm(), b[13] / b[13].norm()
along = (d30 @ d8) * d8; rest = d30 - along
print(f"\nL13: d30 = {d30 @ d8:.2f} x d8 + a remainder of length {rest.norm():.2f}")
for name, v in (("d8 (original)", d8), ("d30 (sturdier)", d30), ("what d30 adds (remainder)", rest), ("what d8 has that d30 lacks", d8 - (d8 @ d30) * d30)):
    t, bt = lens(v)
    print(f"\n{name}\n   toward: {t}\n   away:   {bt}")

# 11-03 08:50 (three: "can we get where they overlap, for a slightly clearer 'but'?"): the bisector of the two unit
# directions keeps what they share (cos 0.97 with each) and halves what's unique to either.
both = (d8 + d30) / (d8 + d30).norm()
t, bt = lens(both, 16)
print(f"\noverlap (bisector): cos with d8 {both @ d8:.3f}, with d30 {both @ d30:.3f}\n   toward: {t}\n   away:   {bt}")
out = torch.load("doubt_dirs_8runs.pt"); out["dirs"] = a.clone(); out["dirs"][13] = both
torch.save(out, "doubt_dirs_both.pt")
