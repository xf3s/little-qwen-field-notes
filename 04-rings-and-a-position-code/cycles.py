"""Positions in a cycle (11-04, three). Do ordinals / numbers / digits / solfege share little Qwen's day-month position code? Laptop."""
import torch, random
from transformers import AutoTokenizer, AutoModelForImageTextToText
torch.set_num_threads(8)
P = '/home/three/models/qwen3.5-0.8b'; L = 11
tok = AutoTokenizer.from_pretrained(P); m = AutoModelForImageTextToText.from_pretrained(P, dtype=torch.float32).eval()
SETS = {
 'ordinals': (['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh'],
              ["It was the {} time I saw her.", "She finished in {} place.", "This is my {} attempt.", "He lives on the {} floor.",
               "Turn left at the {} light.", "The {} chapter is short.", "It was their {} child.", "We sat in the {} row.",
               "That was the {} question.", "The {} day was the hardest.", "Read the {} line again.", "Our {} meeting is today."]),
 'numbers': (['one', 'two', 'three', 'four', 'five', 'six', 'seven'],
             ["I have {} cats.", "We waited {} hours.", "She bought {} apples.", "There are {} chairs here.", "It took {} days.",
              "He ran {} miles.", "They have {} children.", "We need {} more.", "I read {} books.", "The box holds {} cups.",
              "Only {} people came.", "Add {} eggs."]),
 'digits': (['1', '2', '3', '4', '5', '6', '7'],
            ["I have {} cats.", "We waited {} hours.", "She bought {} apples.", "There are {} chairs here.", "It took {} days.",
             "He ran {} miles.", "They have {} children.", "We need {} more.", "I read {} books.", "The box holds {} cups.",
             "Only {} people came.", "Add {} eggs."]),
 'solfege': (['do', 're', 'mi', 'fa', 'sol', 'la', 'ti'],
             ["Sing the note {} now.", "The scale goes up to {}.", "Hold the {} a little longer.", "She hit a clear {}.",
              "Start the song on {}.", "The melody lands on {}.", "In solfege, {} comes next.", "Tune the string to {}.",
              "The choir sang {} together.", "Play {} on the piano.", "The bell rang a high {}.", "He hummed a soft {}."]),
}
def word_means(words, temps):
    X = torch.zeros(len(temps), len(words), 1024)
    for i, t in enumerate(temps):
        for j, w in enumerate(words):
            pre = t.split('{}')[0]; n = len(tok(pre + w, add_special_tokens=False).input_ids) - 1
            ids = tok(t.format(w), add_special_tokens=False).input_ids
            with torch.no_grad(): X[i, j] = m(input_ids=torch.tensor([ids]), output_hidden_states=True).hidden_states[L + 1][0, n]
    X = X - X.mean(1, keepdim=True); M = X.mean(0); return M / M.norm(dim=1, keepdim=True)
ref = torch.load('/home/three/interp/atlas/ring_sae_data.pt')
D = ref['days'].mean(0); D = D / D.norm(dim=1, keepdim=True)
Mo = ref['months'].mean(0)[:7]; Mo = Mo / Mo.norm(dim=1, keepdim=True)
def score(A, B):
    C = A @ B.T; same = C.diag().mean(); off = (C.sum() - C.diag().sum()) / (C.numel() - 7)
    g = torch.Generator().manual_seed(0); null = torch.tensor([float(C[torch.arange(7), torch.randperm(7, generator=g)].mean()) for _ in range(20000)])
    best = sum(int(C[i].argmax() == i) for i in range(7))
    return float(same), float(off), float((null >= same).float().mean()), best, C
out = {}
for name, (words, temps) in SETS.items():
    W = word_means(words, temps); out[name] = W
    for refname, R in (('days', D), ('months Jan-Jul', Mo)):
        s, o, p, b, C = score(W, R)
        print(f'{name:9s} vs {refname:14s}: same-position {s:+.2f} vs other {o:+.2f}, p = {p:.4f}, own position is best match {b}/7')
    print('   diag: ' + ' '.join(f'{words[i]}~{["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][i]} {float((W @ D.T)[i, i]):+.2f}' for i in range(7)))
names = list(SETS)
print('\nbetween the new sets (same-position mean, p):')
for a in range(len(names)):
    for b in range(a + 1, len(names)):
        s, o, p, bb, _ = score(out[names[a]], out[names[b]]); print(f'  {names[a]} vs {names[b]}: {s:+.2f} vs {o:+.2f}, p = {p:.4f}')
torch.save(out, '/home/three/interp/atlas/cycles.pt')
