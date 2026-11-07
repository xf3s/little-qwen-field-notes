"""Map little Qwen's shared number line (11-04, three: "did we find that number line? can we map it?"). Laptop."""
import torch, sys
from transformers import AutoTokenizer, AutoModelForImageTextToText
torch.set_num_threads(8)
P = '/home/three/models/qwen3.5-0.8b'; L = 11
tok = AutoTokenizer.from_pretrained(P); m = AutoModelForImageTextToText.from_pretrained(P, dtype=torch.float32).eval()
OT = ["It was the {} time I saw her.", "She finished in {} place.", "This is my {} attempt.", "He lives on the {} floor.",
      "Turn left at the {} light.", "The {} chapter is short.", "It was their {} child.", "We sat in the {} row.",
      "That was the {} question.", "The {} day was the hardest.", "Read the {} line again.", "Our {} meeting is today."]
NT = ["I have {} cats.", "We waited {} hours.", "She bought {} apples.", "There are {} chairs here.", "It took {} days.",
      "He ran {} miles.", "They have {} children.", "We need {} more.", "I read {} books.", "The box holds {} cups.",
      "Only {} people came.", "Add {} eggs."]
SETS = {'ordinals': ('first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth'.split(), OT),
        'numbers': ('one two three four five six seven eight nine ten eleven twelve'.split(), NT),
        'digits': ([str(i) for i in range(1, 13)], NT)}
def means(words, temps):
    X = torch.zeros(len(temps), len(words), 1024)
    for i, t in enumerate(temps):
        for j, w in enumerate(words):
            pre = t.split('{}')[0]; n = len(tok(pre + w, add_special_tokens=False).input_ids) - 1
            with torch.no_grad(): X[i, j] = m(input_ids=torch.tensor([tok(t.format(w), add_special_tokens=False).input_ids]), output_hidden_states=True).hidden_states[L + 1][0, n]
    X = X - X.mean(1, keepdim=True); return X.mean(0)
out = {k: means(w, t) for k, (w, t) in SETS.items()}
ref = torch.load('/home/three/interp/atlas/ring_sae_data.pt')
out['days'] = ref['days'].mean(0); out['months'] = ref['months'].mean(0)
torch.save(out, '/home/three/interp/atlas/numberline.pt'); print({k: tuple(v.shape) for k, v in out.items()})
