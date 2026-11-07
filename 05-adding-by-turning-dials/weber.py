"""Is little Qwen's number line compressed? (11-04, three: "try it!!"). Digits 1..100, L11, last token of the number.
Target-free: distances between numbers vs |n-m| and |log n - log m|, controlling for the shared last digit. Laptop."""
import torch, math, itertools
from transformers import AutoTokenizer, AutoModelForImageTextToText
torch.set_num_threads(8)
P = '/home/three/models/qwen3.5-0.8b'; L = 11
tok = AutoTokenizer.from_pretrained(P); m = AutoModelForImageTextToText.from_pretrained(P, dtype=torch.float32).eval()
NT = ["I have {} cats.", "We waited {} hours.", "She bought {} apples.", "There are {} chairs here.", "It took {} days.",
      "He ran {} miles.", "They have {} children.", "We need {} more.", "I read {} books.", "The box holds {} cups.",
      "Only {} people came.", "Add {} eggs."]
N = list(range(1, 101))
X = torch.zeros(len(NT), len(N), 1024)
for i, t in enumerate(NT):
    for j, n in enumerate(N):
        pre = t.split('{}')[0]; pos = len(tok(pre + str(n), add_special_tokens=False).input_ids) - 1
        with torch.no_grad(): X[i, j] = m(input_ids=torch.tensor([tok(t.format(n), add_special_tokens=False).input_ids]), output_hidden_states=True).hidden_states[L + 1][0, pos]
X = X - X.mean(1, keepdim=True); M = X.mean(0); torch.save(M, '/home/three/interp/atlas/weber_L11.pt')
D = torch.cdist(M, M)
def spearman(a, b):
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float(); return float(torch.corrcoef(torch.stack([ra, rb]))[0, 1])
pairs = [(i, j) for i in range(100) for j in range(i + 1, 100)]
for name, sel in (('all pairs', lambda a, b: True), ('same last digit', lambda a, b: a % 10 == b % 10),
                  ('different last digit', lambda a, b: a % 10 != b % 10), ('both >= 10, same last digit', lambda a, b: a >= 10 and b >= 10 and a % 10 == b % 10)):
    ps = [(i, j) for i, j in pairs if sel(N[i], N[j])]
    d = torch.tensor([float(D[i, j]) for i, j in ps]); lin = torch.tensor([abs(N[i] - N[j]) for i, j in ps], dtype=torch.float)
    lg = torch.tensor([abs(math.log(N[i]) - math.log(N[j])) for i, j in ps])
    print(f'{name:28s} ({len(ps):4d} pairs): Spearman with |n-m| {spearman(d, lin):+.3f}, with |log n - log m| {spearman(d, lg):+.3f}')
steps = [(10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]
print('\ndistance for each step of ten: ' + '  '.join(f'{a}->{b} {float(D[a-1, b-1]):.2f}' for a, b in steps))
ones = [(1, 2), (2, 3), (3, 4), (5, 6), (8, 9), (11, 12), (21, 22), (51, 52), (81, 82), (91, 92)]
print('distance for steps of one:  ' + '  '.join(f'{a}->{b} {float(D[a-1, b-1]):.2f}' for a, b in ones))
