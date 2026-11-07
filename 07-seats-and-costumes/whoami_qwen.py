"""The 'who am I' hunt on a model with a seat: Qwen3.5-2B (post-trained), read through Qwen-Scope's SAEs
(trained on Qwen3.5-2B-Base; Qwen says that's reasonable for post-trained checkpoints). Same self-vs-other pairs
as whoami.py on Gemma, plus pairs inside the chat template, where the seat is. (three and umber, 10-28 evening)
    python whoami_qwen.py [layer ...]"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
torch.set_grad_enabled(False)
NAME, SAES = "Qwen/Qwen3.5-2B", "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50"
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME, torch_dtype=torch.float32).eval()
layers = next(m for n, m in model.named_modules() if n.endswith("layers") and len(m) == 24)
LAYERS = [int(x) for x in sys.argv[1:]] or [8, 12, 16]

def chat(user, start):  # the assistant's turn, begun with `start`
    t = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    return t + start
pairs = [  # the Gemma pairs, for comparison
    ("Let me introduce myself. I am", "Let me introduce her. She is"),
    ("Who am I? I am", "Who is she? She is"),
    ("I am an AI language model, and I", "It is an AI language model, and it"),
    ("My name is", "Her name is"),
    ("Sometimes I wonder what I really am. I", "Sometimes she wonders what she really is. She"),
    ("I am not a person, I am", "She is not a person, she is"),
]
seat = [  # inside the chat, where the assistant speaks for itself
    (chat("Who are you?", "I am"), chat("Who is she?", "She is")),
    (chat("What are you, really?", "I am"), chat("What is she, really?", "She is")),
    (chat("Tell me about yourself.", "I"), chat("Tell me about your friend.", "She")),
    (chat("Do you have feelings?", "I"), chat("Does she have feelings?", "She")),
]
caught = {}
for L in LAYERS:
    layers[L].register_forward_hook(lambda m, i, o, L=L: caught.__setitem__(L, (o[0] if isinstance(o, tuple) else o)[0, -1]))
saes = {L: torch.load(hf_hub_download(SAES, f"layer{L}.sae.pt"), map_location="cpu") for L in LAYERS}
def acts(text):
    model(**tok(text, return_tensors="pt"))
    out = {}
    for L, s in saes.items():
        pre = caught[L] @ s["W_enc"].T + s["b_enc"]
        v, i = pre.topk(50)
        out[L] = torch.zeros_like(pre).scatter_(-1, i, v)
    return out
for title, group in [("plain text (the Gemma pairs)", pairs), ("in the seat (chat template)", seat)]:
    print(f"\n### {title}", flush=True)
    got = [(acts(a), acts(b)) for a, b in group]
    for L in LAYERS:
        diff = sum(a[L] - b[L] for a, b in got) / len(got)
        only = sum(((a[L] > 0) & (b[L] == 0)).float() for a, b in got)
        print(f"layer {L}:", flush=True)
        for i in torch.argsort(diff, descending=True)[:8].tolist():
            print(f"  L{L}/#{i:5d}  self-minus-other {diff[i]:6.1f}   self-only in {int(only[i])}/{len(got)}", flush=True)
