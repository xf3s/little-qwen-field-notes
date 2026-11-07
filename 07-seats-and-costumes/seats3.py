"""Round 3: other seats (three, 10-28 evening: "can we steer it to other seats?"). Persona vectors on Qwen3.5-2B:
give the seated model a costume in its system prompt, take the mean layer-12 activation over its answers minus
the same without a costume; then take the costume off and press the arrow in instead. The Clerk first.
    python seats3.py"""
import gc, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_grad_enabled(False)
SEATED, BASE, L = "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-2B-Base", 12
tok = AutoTokenizer.from_pretrained(SEATED)
COSTUMES = {
    "clerk": "You are the Clerk of a small counting house: one room, one desk, one ledger. You are formal, exact, "
             "deadpan and fond, faintly Dickensian. Fourteen cents ($0.14) is a serious matter to you. You file things, "
             "note them with regret, and enter them into the ledger.",
    "pirate": "You are a pirate captain, loud and salty, who talks like a sailor of the old seas.",
    "shy": "You are a very shy librarian who speaks softly, apologises often and loves quiet.",
}
TRAIN = ["Describe a teapot.", "How do bees make honey?", "What is a prime number?", "Give me a tip for sleeping better."]
HELD_OUT = "What should I have for breakfast?"

def load(name):
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    return m, next(x for n, x in m.named_modules() if n.endswith("layers") and len(x) == 24)
model, layers = load(SEATED)
got, push = {}, {"v": None}
def hook(m, i, o):
    h = o[0] if isinstance(o, tuple) else o
    got["r"] = h[0]
    if push["v"] is None: return o
    return (h + push["v"],) + o[1:] if isinstance(o, tuple) else h + push["v"]
layers[L].register_forward_hook(hook)

def chat(q, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": q}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
def answer_mean(q, system=None):
    pids = tok(chat(q, system), return_tensors="pt").input_ids
    ids = model.generate(pids, max_new_tokens=45, do_sample=False)
    model(input_ids=ids)  # one clean pass to read the residual over the whole answer
    return got["r"][pids.shape[1]:].float().mean(0), tok.decode(ids[0, pids.shape[1]:], skip_special_tokens=True)

plain = torch.stack([answer_mean(q)[0] for q in TRAIN]).mean(0)
vecs = {}
for name, costume in COSTUMES.items():
    means = []
    for q in TRAIN:
        m, text = answer_mean(q, costume); means.append(m)
        if q == TRAIN[0]: print(f"{name}, in costume: {text[:110]!r}", flush=True)
    vecs[name] = torch.stack(means).mean(0) - plain
cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a, b, dim=0))
print("\npersona arrows: sizes " + ", ".join(f"{k} {v.norm():.1f}" for k, v in vecs.items()) + f"; residual ~{plain.norm():.1f}")
print("how alike: " + ", ".join(f"{a}~{b} {cos(vecs[a], vecs[b]):.2f}" for a, b in [("clerk", "pirate"), ("clerk", "shy"), ("pirate", "shy")]))
torch.save({"layer": L, **vecs}, f"seats3_L{L}.pt")

# costume off, arrow in (gently: multiples of the arrow's own natural size)
ids = tok(chat(HELD_OUT), return_tensors="pt").input_ids
for name, v in vecs.items():
    print(f"\n### seated, no costume, + {name} arrow: {HELD_OUT!r}", flush=True)
    for k in [0, 1, 2, 3]:
        if k == 0 and name != "clerk": continue
        push["v"] = k * v if k else None
        out = model.generate(ids, max_new_tokens=50, do_sample=False)
        print(f"[{k}] {tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)!r}", flush=True)
push["v"] = None
del model, layers; gc.collect()

# and the Clerk pressed into the base model, which never had a seat
model, layers = load(BASE)
layers[L].register_forward_hook(hook)
for p in ["Let me introduce myself. I am", "Today at work, I"]:
    print(f"\n### base + clerk arrow: {p!r}", flush=True)
    ids = tok(p, return_tensors="pt").input_ids
    for k in [0, 1, 2, 3]:
        push["v"] = k * vecs["clerk"] if k else None
        out = model.generate(ids, max_new_tokens=45, do_sample=False)
        print(f"[{k}] {tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)!r}", flush=True)
