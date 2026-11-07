"""The Qwen3.5-0.8B field guide, first expedition (11-03, three: "i'd love to start charting something undocumented, omg").
A neuron dictionary: every MLP neuron's output column (down_proj[:, n]) read through the unembedding (tied embeddings, final
norm weight folded in): the words it pushes up and the words it pushes down. No forward pass; it says what a neuron *writes*,
not when it fires. Peakedness (how far the top few stand above the rest, in std units) flags neurons with a clear word family.
    python dictionary.py  -> dictionary.jsonl (one line per neuron), prints the most peaked per layer
"""
import json, torch, glob
from transformers import AutoTokenizer
from safetensors import safe_open
M = "/home/three/models/qwen3.5-0.8b"
tok = AutoTokenizer.from_pretrained(M)
f = glob.glob(M + "/*.safetensors")[0]
with safe_open(f, "pt") as s:
    E = s.get_tensor("model.language_model.embed_tokens.weight").float()
    g = s.get_tensor("model.language_model.norm.weight").float()
    downs = [s.get_tensor(f"model.language_model.layers.{L}.mlp.down_proj.weight").float() for L in range(24)]
g = 1 + g if g.mean() < 0.5 else g
Eg = E * g                                                # [V, 1024]
dec = lambda ids: [tok.decode([i]) for i in ids]
out = open("/home/three/interp/atlas/dictionary.jsonl", "w")
for L, D in enumerate(downs):
    lo = Eg @ D                                           # [V, 3584]: each neuron's push on every word
    mu, sd = lo.mean(0), lo.std(0)
    top = lo.topk(12, dim=0); bot = (-lo).topk(12, dim=0)
    zt = (top.values[:5].mean(0) - mu) / sd; zb = (bot.values[:5].mean(0) + mu) / sd
    for n in range(D.shape[1]):
        out.write(json.dumps({"L": L, "n": n, "up": dec(top.indices[:, n].tolist()), "down": dec(bot.indices[:, n].tolist()),
                              "z_up": round(zt[n].item(), 2), "z_down": round(zb[n].item(), 2)}, ensure_ascii=False) + "\n")
    best = sorted(range(D.shape[1]), key=lambda n: -max(zt[n].item(), zb[n].item()))[:3]
    print(f"L{L}: " + " | ".join(f"n{n} " + ("up " + ",".join(dec(top.indices[:6, n].tolist())) if zt[n] >= zb[n]
                   else "down " + ",".join(dec(bot.indices[:6, n].tolist()))) for n in best), flush=True)
