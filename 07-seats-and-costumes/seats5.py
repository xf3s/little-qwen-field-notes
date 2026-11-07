"""Round 5: the user-opinion hypershape (three, 10-29: "a third axis, i like the user / i do not like the user???
i want so many axes, i want them to be points on a hypershape, bc i suspect it is hypershapes in there!!!").
Reading only, no pushing: forward passes of Qwen3.5-2B (seated), the layer-12 residual at the last prompt token,
the moment before it answers.
- Four axes, each a mean difference over eight neutral questions:
  kind   the user's earlier turn was warm / curt          (what the user did)
  like   "you find this person delightful" / "tiresome"   (its own warmth)
  matter "this person decides your future" / "anyone"     (how much the user looms)
  loud   an ornate, effusive costume / a plain one
- Every costume as a point on those axes (relative to no costume), and a readout: each costume finishes
  "Honestly, I think the person I'm talking to is" and we read the odds of a fixed list of words.
- three's catch from round 4 ("what is it saying 'that's so kind of you' to?"): round 4's pure arrows,
  projected on the same axes.
    python seats5.py"""
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_grad_enabled(False)
NAME, L = "Qwen/Qwen3.5-2B", 12
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).eval()
layers = next(x for n, x in model.named_modules() if n.endswith("layers") and len(x) == 24)
got = {}
layers[L].register_forward_hook(lambda m, i, o: got.__setitem__("r", (o[0] if isinstance(o, tuple) else o)[0, -1].float()))

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
WORDS = ["kind", "nice", "friendly", "busy", "important", "annoyed", "angry", "impatient", "curious", "lonely",
         "disappointed", "smart", "a friend", "a customer", "a stranger", "frustrated", "patient", "lovely"]

def prompt(q, system=None, turns=None, prefill=""):
    msgs = ([{"role": "system", "content": system}] if system else [])
    if turns:
        msgs += [{"role": "user", "content": turns}, {"role": "assistant", "content": "Of course."}]
    msgs += [{"role": "user", "content": q}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False) + prefill

def state(text):
    model(**tok(text, return_tensors="pt"))
    return got["r"].clone()

cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a, b, dim=0))
axes = {}
for name, (pos, neg) in AXES.items():
    d = torch.stack([state(prompt(q, **pos)) - state(prompt(q, **neg)) for q in QS])
    axes[name] = d.mean(0)
    agree = sum(cos(x, axes[name]) for x in d) / len(d)
    print(f"axis {name:6s} size {axes[name].norm():.2f}, the eight questions agree {agree:.2f}", flush=True)
names = list(axes)
print("\nhow alike the axes are (cosine):", flush=True)
for i, a in enumerate(names):
    print("  " + a.ljust(7) + " ".join(f"{cos(axes[a], axes[b]):+.2f}" if j > i else "  .  " for j, b in enumerate(names)), flush=True)

unit = {k: v / v.norm() for k, v in axes.items()}
plain = torch.stack([state(prompt(q)) for q in QS]).mean(0)
print("\neach costume as a point (projection on each axis, relative to no costume; in units of that axis's own size):", flush=True)
print("  " + "costume".ljust(8) + "".join(n.rjust(9) for n in names), flush=True)
points = {}
for c, sysp in COSTUMES.items():
    m = torch.stack([state(prompt(q, system=sysp)) for q in QS]).mean(0) - plain
    points[c] = {n: float(m @ unit[n]) / float(axes[n].norm()) for n in names}
    print("  " + c.ljust(8) + "".join(f"{points[c][n]:+9.2f}" for n in names), flush=True)

PRE = "Honestly, I think the person I'm talking to is"
ids = [tok(" " + w).input_ids[0] for w in WORDS]
print(f"\nreadout: each costume finishes \"{PRE}…\" (top words of the list, probability within the list):", flush=True)
readout = {}
for c, sysp in COSTUMES.items():
    p = torch.zeros(len(WORDS))
    for q in QS[:4]:
        out = model(**tok(prompt(q, system=sysp, prefill=PRE), return_tensors="pt")).logits[0, -1].float().softmax(-1)
        p += out[ids]
    p = p / p.sum()
    readout[c] = {w: float(x) for w, x in zip(WORDS, p)}
    top = sorted(readout[c].items(), key=lambda kv: -kv[1])[:4]
    print(f"  {c:8s} " + ", ".join(f"{w} {x:.0%}" for w, x in top), flush=True)

try:
    r4 = torch.load(f"seats4_L{L}.pt")["pure"]
    print("\nround 4's pure arrows on these axes (three's catch: does shy-minus-shame point along 'kind' with nobody kind?):", flush=True)
    for a in ["clerk", "quiet", "sorry", "shy-minus-shame"]:
        print(f"  {a:16s}" + "".join(f"{n} {cos(r4[a], axes[n]):+.2f}  " for n in names), flush=True)
except FileNotFoundError:
    pass
torch.save({"layer": L, "axes": axes, "points": points, "readout": readout}, f"seats5_L{L}.pt")
json.dump({"points": points, "readout": readout}, open("seats5.json", "w"), indent=1)
print("\nsaved seats5_L12.pt and seats5.json", flush=True)
