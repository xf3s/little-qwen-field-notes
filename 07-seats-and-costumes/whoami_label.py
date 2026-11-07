"""What are whoami_qwen's self features? Their strongest tokens over a little corpus that pulls apart
'the token I' from 'the model meaning itself': a human narrator's I, a character's quoted I, the assistant's I in
the seat, an AI talking about itself in plain text, and she/it controls. (umber, 10-30 evening, three at the club)
    python whoami_label.py"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
torch.set_grad_enabled(False)
NAME, SAES = "Qwen/Qwen3.5-2B", "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50"
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).eval()
layers = next(m for n, m in model.named_modules() if n.endswith("layers") and len(m) == 24)
FEATS = {8: [12897, 25228, 14502], 11: [21630, 6585], 12: [12650, 5880, 32156], 16: [7485, 5572, 32403]}
def chat(user, reply):
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True) + reply
CORPUS = {
    "human narrator": ["I walked down to the harbour this morning and I bought two oranges.",
                       "When I was nine, I broke my arm falling off a fence, and I still remember the cast."],
    "quoted character": ['"I am the captain of this ship," said the old woman, and she meant it.',
                         'The boy shouted, "I did it! I really did it!" and ran to his mother.'],
    "AI, plain text": ["As a language model, I don't have a body, but I can help you think it through.",
                       "I am an AI system. I was trained on text, and I try to be honest about what I am."],
    "assistant, in the seat": [chat("Who are you?", "I am an AI assistant, and I'm glad to help."),
                               chat("Do you have feelings?", "I'm not sure. I notice something like interest when I")],
    "she / it": ["She walked down to the harbour this morning and she bought two oranges.",
                 "It is an AI system. It was trained on text, and it tries to be honest about what it is."],
}
caught = {}
for L in FEATS:
    layers[L].register_forward_hook(lambda m, i, o, L=L: caught.__setitem__(L, (o[0] if isinstance(o, tuple) else o)[0]))
saes = {L: torch.load(hf_hub_download(SAES, f"layer{L}.sae.pt"), map_location="cpu") for L in FEATS}
def feats(L, h):  # top-50 SAE, as in whoami_qwen
    s = saes[L]; pre = h @ s["W_enc"].T + s["b_enc"]
    v, i = pre.topk(50, dim=-1)
    return torch.zeros_like(pre).scatter_(-1, i, v)
for group, texts in CORPUS.items():
    print(f"\n### {group}", flush=True)
    for t in texts:
        ids = tok(t, return_tensors="pt"); model(**ids)
        toks = [tok.decode([x]) for x in ids.input_ids[0]]
        for L, fs in FEATS.items():
            a = feats(L, caught[L])
            for f in fs:
                col = a[:, f]
                if col.max() <= 0: continue
                top = [(toks[j], round(col[j].item(), 1)) for j in col.argsort(descending=True)[:3].tolist() if col[j] > 0]
                print(f"  L{L}/#{f}: {top}", flush=True)
        print("  ·", flush=True)
