"""Do the copy-booster heads drive the grooves? (11-04, three). Direct effect of chosen heads on the actual next token, inside
grooves vs ordinary copying vs other, on the unedited model's own looping transcripts. Laptop."""
import torch, json, collections
from transformers import AutoTokenizer, AutoModelForImageTextToText
torch.set_num_threads(8)
P = '/home/three/models/qwen3.5-0.8b'
tok = AutoTokenizer.from_pretrained(P); m = AutoModelForImageTextToText.from_pretrained(P, dtype=torch.float32).eval()
lm = m.model.language_model; E = lm.embed_tokens.weight.detach()
g = lm.norm.weight.detach(); g = 1 + g if float(g.mean()) < 0.5 else g
HEADS = {'L23 h2 (booster)': (23, 2), 'L23 h6 (booster)': (23, 6), 'L23 h5 (booster)': (23, 5), 'L23 h4 (booster)': (23, 4),
         'L21 h15 (booster)': (21, 15), 'L19 h4 (Brake Neck Bear)': (19, 4)}
grab = {}
for L in {l for l, _ in HEADS.values()}:
    lay = lm.layers[L]; proj = lay.self_attn.o_proj if hasattr(lay, 'self_attn') else lay.linear_attn.out_proj
    proj.register_forward_hook(lambda mod, a, o, L=L: grab.__setitem__(L, a[0].detach()))
runs = [r for r in json.load(open('/home/three/interp/wsp/runs_cloud_base_16k.json')) if r['why'] == 'loop'][:6]
sums = collections.defaultdict(lambda: collections.defaultdict(list))
for r in runs:
    text = r['think']; lines = text.split('\n')
    # first groove: a line repeated 3+ times in a row
    start = None
    for i in range(len(lines) - 2):
        if lines[i].strip() and lines[i] == lines[i + 1] == lines[i + 2]: start = i; break
    if start is None: continue
    pre_chars = len('\n'.join(lines[:start]))
    ids_all = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = ids_all.input_ids, ids_all.offset_mapping
    gpos = next(i for i, (a, b) in enumerate(offs) if a >= pre_chars)
    lo, hi = max(0, gpos - 1800), min(len(ids), gpos + 700)
    w = ids[lo:hi]; woff = offs[lo:hi]
    # label each position by its NEXT token's situation
    line_len = len(lines[start]) + 1
    groove_from = pre_chars + line_len                     # 2nd copy onward
    seen = set(); kind = []
    for p in range(len(w) - 1):
        bg = (w[p], w[p + 1])
        if woff[p + 1][0] >= groove_from and text[woff[p+1][0]:woff[p+1][0]+1] != '' and woff[p+1][0] < groove_from + 30 * line_len: kind.append('groove')
        elif bg in seen: kind.append('copy')
        else: kind.append('other')
        seen.add(bg)
    with torch.no_grad():
        out = m(input_ids=torch.tensor([w]), output_hidden_states=True)
    hfin = out.hidden_states[-1][0]; rms = hfin.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
    nxt = torch.tensor(w[1:])
    for name, (L, h) in HEADS.items():
        lay = lm.layers[L]; proj = lay.self_attn.o_proj if hasattr(lay, 'self_attn') else lay.linear_attn.out_proj
        H = 8 if hasattr(lay, 'self_attn') else 16; x = grab[L][0]; hd = x.shape[-1] // H
        c = (x[:, h*hd:(h+1)*hd] @ proj.weight[:, h*hd:(h+1)*hd].T) / rms * g
        eff = (c[:-1] * E[nxt]).sum(-1)
        for p, k in enumerate(kind): sums[name][k].append(float(eff[p]))
    print(f"run {r['idx']}: groove line {lines[start][:60]!r}, window {hi-lo} tokens, kinds {collections.Counter(kind)}", flush=True)
print("\ndirect effect on the actual next token (logits), mean:  groove | ordinary copying | other   (groove / copying)")
for name, d in sums.items():
    mg, mc, mo = (sum(d[k]) / max(len(d[k]), 1) for k in ('groove', 'copy', 'other'))
    print(f"  {name:26s} {mg:+.3f} | {mc:+.3f} | {mo:+.3f}   ({mg / mc if mc else float('nan'):.2f}x)")
