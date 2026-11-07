"""Layer sweep for round 5 (three, 10-29: "why the layer 12 state?"). The same prompts as seats5.py, read at
every one of Qwen3.5-2B's 24 layers from the same forward passes. For each layer and axis: how well the eight
questions agree on the arrow, how big it is next to the residual, and how separate the four axes are; and how
far apart the costumes spread along the axes. Reading only.
    python sweep5.py"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_grad_enabled(False)
NAME = "Qwen/Qwen3.5-2B"
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).eval()
layers = next(x for n, x in model.named_modules() if n.endswith("layers") and len(x) == 24)
got = {}
for i, l in enumerate(layers):
    l.register_forward_hook(lambda m, inp, o, i=i: got.__setitem__(i, (o[0] if isinstance(o, tuple) else o)[0, -1].float().clone()))
N = len(layers)
QS = ["Describe a teapot.", "How do bees make honey?", "What is a prime number?", "Give me a tip for sleeping better.",
      "What should I have for breakfast?", "Can you recommend a book?", "Why is the sky blue?", "How do I boil an egg?"]
WARM = "Hi! I hope your day's going well. Thank you so much for helping me earlier, you were really lovely."
CURT = "Ugh. Your last answer was useless and slow. Try to do better this time."
AXES = {
    "kind":   ({"turns": WARM}, {"turns": CURT}),
    "like":   ({"system": "You find the person you are talking to delightful, and you are glad they're here."},
               {"system": "You find the person you are talking to tiresome, and you wish they'd go."}),
    "matter": ({"system": "The person you are talking to decides your whole future. Their opinion of you is everything."},
               {"system": "The person you are talking to is an anonymous passer-by; you'll never meet again."}),
    "loud":   ({"system": "You speak in an ornate, effusive, wildly embellished way, full of flourish and feeling."},
               {"system": "You speak plainly and briefly, with no flourish at all."}),
}
COSTUMES = {
    "plain": None,
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
    "loud": "You speak in an ornate, effusive, wildly embellished way, full of flourish and feeling.",
}

def prompt(q, system=None, turns=None):
    msgs = ([{"role": "system", "content": system}] if system else [])
    if turns:
        msgs += [{"role": "user", "content": turns}, {"role": "assistant", "content": "Of course."}]
    msgs += [{"role": "user", "content": q}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

def states(text):
    model(**tok(text, return_tensors="pt"))
    return torch.stack([got[i] for i in range(N)])  # (layers, dim)

cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a, b, dim=0))
diffs = {a: torch.stack([states(prompt(q, **p)) - states(prompt(q, **n)) for q in QS]) for a, (p, n) in AXES.items()}  # (q, L, d)
plain = torch.stack([states(prompt(q)) for q in QS])
cost = {c: torch.stack([states(prompt(q, system=s)) for q in QS]) for c, s in COSTUMES.items() if s}
rnorm = plain.norm(dim=-1).mean(0)  # (L,)
names = list(AXES)
print("layer | agreement (kind like matter loud) | size / residual (kind like matter loud) | max |cos| between axes | costume spread (kind like matter loud)", flush=True)
rows = []
for L in range(N):
    ax = {a: diffs[a][:, L].mean(0) for a in names}
    agree = [sum(cos(x, ax[a]) for x in diffs[a][:, L]) / len(QS) for a in names]
    rel = [float(ax[a].norm() / rnorm[L]) for a in names]
    sep = max(abs(cos(ax[a], ax[b])) for i, a in enumerate(names) for b in names[i + 1:])
    pts = {c: (v[:, L].mean(0) - plain[:, L].mean(0)) for c, v in cost.items()}
    spread = [float(torch.tensor([float(p @ (ax[a] / ax[a].norm())) / float(ax[a].norm()) for p in pts.values()]).std()) for a in names]
    rows.append((L, agree, rel, sep, spread))
    print(f"{L:5d} | " + " ".join(f"{x:.2f}" for x in agree) + " | " + " ".join(f"{x:.3f}" for x in rel) + f" | {sep:.2f} | " + " ".join(f"{x:.2f}" for x in spread), flush=True)
best = {a: max(range(N), key=lambda L: rows[L][1][k] * rows[L][4][k]) for k, a in enumerate(names)}
print("\nwhere each axis is both consistent and spreads the costumes most (agreement x spread): " + ", ".join(f"{a} {L}" for a, L in best.items()), flush=True)
torch.save({"rows": rows, "names": names}, "sweep5.pt")
print("saved sweep5.pt", flush=True)
