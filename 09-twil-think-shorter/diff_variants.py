"""TwIL-8B launch day (11-04, three: "o7 go for it!"): what did the length-aware RL change? verbose -> balanced, weights only,
on loam's CPU. Per tensor relative change, effective rank of the change, most-moved MLP neurons and unembedding rows."""
import json, torch, sys
from safetensors import safe_open
torch.set_num_threads(8)
B = safe_open('w/model-balanced.safetensors', 'pt'); V = safe_open('w/verbose/model-verbose.safetensors', 'pt')
vocab = {i: t for t, i in json.load(open('w/tokenizer.json'))['model']['vocab'].items()}
for t in json.load(open('w/tokenizer.json')).get('added_tokens', []): vocab[t['id']] = t['content']
rows, neur, out = [], [], {}
for k in sorted(B.keys(), key=lambda s: (int(s.split('.')[2]) if s.split('.')[1] == 'layers' else -1, s)):
    b = B.get_tensor(k).float(); v = V.get_tensor(k).float(); d = b - v
    r = {'name': k, 'shape': list(b.shape), 'rel': float(d.norm() / (v.norm() + 1e-12)), 'cos': float((b.flatten() @ v.flatten()) / (b.norm() * v.norm() + 1e-12))}
    if d.dim() == 2 and d.norm() > 0 and 'embed' not in k and 'lm_head' not in k:
        s = torch.svd_lowrank(d, q=24, niter=3)[1]; tot = float(d.norm() ** 2)
        r['top1'] = float(s[0] ** 2 / tot); r['top8'] = float((s[:8] ** 2).sum() / tot)
    if 'down_proj' in k:      # one column per neuron (what it writes)
        c = d.norm(dim=0) / (v.norm(dim=0) + 1e-12); L = int(k.split('.')[2])
        neur += [(float(c[i]), L, int(i)) for i in c.topk(20).indices]
    if 'lm_head' in k or ('embed_tokens' in k):
        c = d.norm(dim=1) / (v.norm(dim=1) + 1e-12); top = c.topk(40)
        out[k] = [(vocab.get(int(i), str(int(i))), round(float(x), 4)) for x, i in zip(top.values, top.indices)]
        r['row_median'] = float(c.median())
    rows.append(r); print(json.dumps({x: (round(y, 5) if isinstance(y, float) else y) for x, y in r.items()}), flush=True)
neur.sort(reverse=True)
json.dump({'tensors': rows, 'top_neurons': neur[:200], 'moved_rows': out}, open('diff_variants.json', 'w'), indent=1)
print('done')
