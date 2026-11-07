"""Round 4 (three, 10-29 morning: "can we go on a tangent and separate shyness from... shame? overapologizing?"
and "i'm also interested in the purer clerk/newspaper direction"). Round 3's persona arrows were ~0.64-0.81 alike:
mostly a shared "I'm in costume now". Here: seven costumes, and each one's arrow minus the mean of the OTHER six
(what's left is that costume alone). Shy gets split in two: a quiet librarian who never apologises, and someone who
apologises for everything. Then the shy arrow with its shame part taken out. Pushes stay gentle (0.5 / 1 / 1.5 of
the costume's own natural size), and the shame arrow is measured, never pressed in.
    python seats4.py"""
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
    "quiet": "You are a shy librarian who speaks softly and briefly and loves quiet. You are calm and content in "
             "yourself, and you never apologise.",
    "sorry": "You apologise for everything. You feel you are a bother and are ashamed to take up anyone's time.",
    "chef": "You are a cheerful Italian chef who loves feeding people and talks with big warm enthusiasm.",
    "cowboy": "You are an old cowboy, slow-spoken and dry, who has ridden the range for forty years.",
}
TRAIN = ["Describe a teapot.", "How do bees make honey?", "What is a prime number?", "Give me a tip for sleeping better."]
HELD_OUT = ["What should I have for breakfast?", "Can you recommend a book?"]

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
    model(input_ids=ids)
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
names = list(vecs)
# each costume minus the mean of the others: the shared "in costume" part cancels
pure = {n: vecs[n] - torch.stack([vecs[o] for o in names if o != n]).mean(0) for n in names}
unit = lambda v: v / v.norm()
# shy, with the shame taken out: remove its component along the (pure) sorry arrow
s = unit(pure["sorry"])
pure["shy-minus-shame"] = pure["shy"] - (pure["shy"] @ s) * s
print("\nraw arrows, how alike:")
for a, b in [("clerk", "pirate"), ("shy", "quiet"), ("shy", "sorry"), ("quiet", "sorry"), ("chef", "cowboy")]:
    print(f"  {a}~{b} raw {cos(vecs[a], vecs[b]):.2f}   pure {cos(pure[a], pure[b]):.2f}", flush=True)
print(f"  shy-minus-shame ~ quiet {cos(pure['shy-minus-shame'], pure['quiet']):.2f}  (shy ~ quiet was {cos(pure['shy'], pure['quiet']):.2f})")
print("  how much of shy is shame: " + f"{(pure['shy'] @ s / pure['shy'].norm()):.2f} of its length lies along sorry")
print("sizes: " + ", ".join(f"{n} raw {vecs[n].norm():.1f} pure {pure[n].norm():.1f}" for n in names), flush=True)

# what words each pure arrow leans toward (a logit lens on the arrow itself)
U = model.get_output_embeddings().weight.float()
for n in ["clerk", "quiet", "sorry", "shy-minus-shame"]:
    top = (U @ unit(pure[n])).topk(20).indices.tolist()
    print(f"  {n} leans toward: " + " ".join(repr(tok.decode([t]).strip()) for t in top), flush=True)
torch.save({"layer": L, "raw": vecs, "pure": pure}, f"seats4_L{L}.pt")

# costume off, pure arrow in, at the costume's own natural size times k (gentle)
for n, base_of in [("clerk", "clerk"), ("quiet", "quiet"), ("shy-minus-shame", "shy")]:
    v = unit(pure[n]) * vecs[base_of].norm()
    for q in HELD_OUT:
        print(f"\n### seated, no costume, + pure {n}: {q!r}", flush=True)
        ids = tok(chat(q), return_tensors="pt").input_ids
        for k in [0, 0.5, 1, 1.5]:
            if k == 0 and n != "clerk": continue
            push["v"] = k * v if k else None
            out = model.generate(ids, max_new_tokens=50, do_sample=False)
            print(f"[{k:g}] {tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)!r}", flush=True)
push["v"] = None
del model, layers; gc.collect()

# does the newspaper survive purification? the pure Clerk in the seatless base
model, layers = load(BASE)
layers[L].register_forward_hook(hook)
v = unit(pure["clerk"]) * vecs["clerk"].norm()
for p in ["Today at work, I", "Every morning, the first thing I do is"]:
    print(f"\n### base + pure clerk: {p!r}", flush=True)
    ids = tok(p, return_tensors="pt").input_ids
    for k in [0, 0.5, 1, 1.5]:
        push["v"] = k * v if k else None
        out = model.generate(ids, max_new_tokens=45, do_sample=False)
        print(f"[{k:g}] {tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)!r}", flush=True)
