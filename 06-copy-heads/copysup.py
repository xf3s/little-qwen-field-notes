"""Is there a general copy-suppression head in little Qwen? (11-04, three). Direct logit effect of every head on the model's
own predicted next tokens, split into ones already in the context (repeats) vs not. Laptop."""
import torch, glob, collections
from datasets import load_dataset
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForImageTextToText
torch.set_num_threads(8)
P = '/home/three/models/qwen3.5-0.8b'
tok = AutoTokenizer.from_pretrained(P); m = AutoModelForImageTextToText.from_pretrained(P, dtype=torch.float32).eval()
lm = m.model.language_model
E = lm.embed_tokens.weight.detach()
g = lm.norm.weight.detach(); g = 1 + g if float(g.mean()) < 0.5 else g
texts = [t for t in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")["text"] if len(t) > 800][:40]
grab = {}
hooks = []
for L, layer in enumerate(lm.layers):
    proj = layer.self_attn.o_proj if hasattr(layer, 'self_attn') else layer.linear_attn.out_proj
    hooks.append(proj.register_forward_hook(lambda mod, a, o, L=L: grab.__setitem__(L, a[0].detach())))
nh = {L: (8 if hasattr(lay, 'self_attn') else 16) for L, lay in enumerate(lm.layers)}
score = collections.defaultdict(lambda: [0.0, 0, 0.0, 0])        # sum rep, n rep, sum non, n non
with torch.no_grad():
    for t in texts:
        ids = tok(t, add_special_tokens=False).input_ids[:256]
        out = m(input_ids=torch.tensor([ids]), output_hidden_states=True)
        logits = out.logits[0]; hfin = out.hidden_states[-1][0]                # pre-final-norm residual
        rms = hfin.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
        top = logits.topk(10, dim=-1).indices                                   # [T, 10]
        seen = set(); rep_mask = torch.zeros_like(top, dtype=torch.bool)
        for p in range(len(ids)):
            seen.add(ids[p])
            if p < 20: continue
            rep_mask[p] = torch.tensor([int(x) in seen for x in top[p]])
        valid = torch.arange(len(ids)) >= 20
        for L in range(24):
            x = grab[L][0]; H = nh[L]; hd = x.shape[-1] // H
            proj = lm.layers[L].self_attn.o_proj if hasattr(lm.layers[L], 'self_attn') else lm.layers[L].linear_attn.out_proj
            W = proj.weight                                                     # [1024, H*hd]
            for h in range(H):
                c = x[:, h*hd:(h+1)*hd] @ W[:, h*hd:(h+1)*hd].T                # [T, 1024] this head's write
                c = c / rms * g
                eff = torch.einsum('td,tkd->tk', c, E[top])                    # effect on each top-10 token
                r = rep_mask & valid[:, None]; n = (~rep_mask) & valid[:, None]
                s = score[(L, h)]
                s[0] += float(eff[r].sum()); s[1] += int(r.sum()); s[2] += float(eff[n].sum()); s[3] += int(n.sum())
res = sorted(((s[0]/max(s[1],1) - s[2]/max(s[3],1), s[0]/max(s[1],1), s[2]/max(s[3],1), L, h) for (L, h), s in score.items()))
print("most copy-suppressing heads (effect on predicted repeats minus predicted non-repeats, logits):")
for d, r, n, L, h in res[:10]: print(f"  L{L:<2d} head {h:<2d} {'full' if nh[L]==8 else 'lin '}  score {d:+.3f}   repeats {r:+.3f}  non-repeats {n:+.3f}")
print("most copy-BOOSTING heads (the opposite: pushing repeats up):")
for d, r, n, L, h in res[-5:][::-1]: print(f"  L{L:<2d} head {h:<2d} {'full' if nh[L]==8 else 'lin '}  score {d:+.3f}   repeats {r:+.3f}  non-repeats {n:+.3f}")
b = [x for x in res if x[3] == 19 and x[4] == 4][0]; print(f"\nBrake Neck Bear (L19 head 4): rank {res.index(b)+1} of {len(res)}, score {b[0]:+.3f}")
