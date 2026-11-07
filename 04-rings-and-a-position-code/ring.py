"""The weekday ring (11-04, three: "yes yes!!!"). Are days and months on a circle in little Qwen's notepad? Laptop, fp32."""
import torch, itertools, math, json, sys
from transformers import AutoTokenizer, AutoModelForImageTextToText
torch.set_num_threads(8)
P = '/home/three/models/qwen3.5-0.8b'
tok = AutoTokenizer.from_pretrained(P)
m = AutoModelForImageTextToText.from_pretrained(P, dtype=torch.float32).eval()
DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
T = ["Today is {}.", "See you on {}!", "The meeting was moved to {} afternoon.", "Every {} we go to the market.",
     "She was born on a {}.", "The shop is closed on {}.", "I'll call you back on {}.", "Last {} it rained all day.",
     "The deadline is next {}.", "He always feels tired on {}.", "Our flight leaves {} morning.", "The package arrived on {}.",
     "We have practice on {} evenings.", "It happened on a quiet {}.", "The report is due by {}.", "My favourite day is {}.",
     "The festival starts this {}.", "On {} the trains run late.", "Can we meet {} instead?", "The class was cancelled on {}.",
     "By {} everything was ready.", "I usually do laundry on {}.", "The museum is free on {}.", "They got married on a {}."]
TM = [t.replace('day', 'time') for t in T]   # month templates: same sentences, "favourite time is May" etc.
def reps(words, temps):
    out = torch.zeros(len(temps), len(words), 24, 1024)
    for i, t in enumerate(temps):
        for j, w in enumerate(words):
            pre, post = t.split('{}')
            ids_pre = tok(pre + (' ' if pre and not pre.endswith(' ') else ''), add_special_tokens=False).input_ids if pre else []
            text = t.format(w)
            ids = tok(text, add_special_tokens=False).input_ids
            # the word's last token
            n = len(tok((pre + w), add_special_tokens=False).input_ids) - 1
            with torch.no_grad():
                hs = m(input_ids=torch.tensor([ids]), output_hidden_states=True).hidden_states
            out[i, j] = torch.stack(hs[1:])[:, 0, n]
    return out
def ring(R, names):
    k = len(names); res = []
    for L in range(24):
        X = R[:, :, L] - R[:, :, L].mean(1, keepdim=True)      # centre each template
        M = X.mean(0)                                          # [k, 1024] word means
        U, S, V = torch.linalg.svd(M - M.mean(0), full_matrices=False)
        var2 = float((S[:2] ** 2).sum() / (S ** 2).sum())
        xy = (M - M.mean(0)) @ V[:2].T
        ang = torch.atan2(xy[:, 1], xy[:, 0])
        order = ang.argsort().tolist()
        i0 = order.index(0); cyc = order[i0:] + order[:i0]
        exact = cyc == list(range(k)) or cyc == [0] + list(range(k - 1, 0, -1))
        # adjacency score: fraction of true neighbours (incl. wraparound) that are angular neighbours
        pos = {w: p for p, w in enumerate(order)}
        adj = sum(1 for a in range(k) if abs(pos[a] - pos[(a + 1) % k]) in (1, k - 1)) / k
        gaps = sorted(((ang[(a + 1) % k] - ang[a]) % (2 * math.pi)).item() for a in range(k))
        res.append((L, var2, exact, adj, cyc, xy))
    return res
for words, temps, label in ((DAYS, T, 'days'), (MONTHS, TM, 'months')):
    R = reps(words, temps)
    torch.save(R, f'/home/three/interp/atlas/ring_{label}.pt')
    print(f'\n== {label}: layer | variance in top-2 plane | exact cycle? | neighbours kept | order around the ring')
    for L, var2, exact, adj, cyc, xy in ring(R, words):
        print(f'L{L:<3d} {var2:.2f}  {"YES" if exact else " no"}  {adj:.2f}  {" ".join(words[c][:3] for c in cyc)}')
