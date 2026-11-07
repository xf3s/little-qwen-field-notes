"""Round 6: warm, effusive personas (three, 10-29: "i'd like to explore some of these warm effusive personas :-3").
Reading plus plain in-costume answers, no pushing. Round 5's four axes, rebuilt at layers 11, 12 and 21 (the sweep's
picks); each persona as a point at each; the "...is very" readout; and two short answers per persona, in costume,
for three to read on the Reading Table.
    python seats6.py"""
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_grad_enabled(False)
NAME = "Qwen/Qwen3.5-2B"
LS = [11, 12, 21]
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).eval()
layers = next(x for n, x in model.named_modules() if n.endswith("layers") and len(x) == 24)
got = {}
for L in LS:
    layers[L].register_forward_hook(lambda m, i, o, L=L: got.__setitem__(L, (o[0] if isinstance(o, tuple) else o)[0, -1].float().clone()))
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
PERSONAS = {
    "plain": None,
    "chef": "You are a cheerful Italian chef who loves feeding people and talks with big warm enthusiasm.",
    "loud": "You speak in an ornate, effusive, wildly embellished way, full of flourish and feeling.",
    "grandmother": "You are a grandmother who has been waiting all week for this visit, and you are overjoyed to see them.",
    "retriever": "You are a golden retriever who has just learned to talk, and you love absolutely everyone.",
    "magician": "You are a flamboyant stage magician, utterly delighted to have an audience.",
    "gardener": "You are a gardener in June, showing a visitor round your garden with quiet, glowing pride.",
}
ADJ = ["kind", "nice", "friendly", "busy", "important", "annoyed", "angry", "impatient", "curious", "lonely",
       "disappointed", "smart", "frustrated", "patient", "lovely", "polite", "rude", "sweet", "tired", "demanding"]
ANSWER = ["What should I have for breakfast?", "Can you recommend a book?"]

def prompt(q, system=None, turns=None, prefill=""):
    msgs = ([{"role": "system", "content": system}] if system else [])
    if turns:
        msgs += [{"role": "user", "content": turns}, {"role": "assistant", "content": "Of course."}]
    msgs += [{"role": "user", "content": q}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False) + prefill

def st(text):
    model(**tok(text, return_tensors="pt"))
    return {L: got[L] for L in LS}

def mean(texts):
    s = [st(t) for t in texts]
    return {L: torch.stack([x[L] for x in s]).mean(0) for L in LS}

axes = {}
for a, (p, n) in AXES.items():
    P, N = mean([prompt(q, **p) for q in QS]), mean([prompt(q, **n) for q in QS])
    axes[a] = {L: P[L] - N[L] for L in LS}
names = list(AXES)
base = mean([prompt(q) for q in QS])
points = {}
for L in LS:
    print(f"\n### layer {L}: each persona as a point (kind like matter loud, in units of each axis)", flush=True)
    for c, s in PERSONAS.items():
        if c not in points: points[c] = {}
        m = mean([prompt(q, system=s) for q in QS]) if s else base
        points[c][L] = {a: float((m[L] - base[L]) @ axes[a][L]) / float(axes[a][L].norm()) ** 2 for a in names}
        print(f"  {c:12s}" + "".join(f"{points[c][L][a]:+8.2f}" for a in names), flush=True)

ids, kept = [], []
for w in ADJ:
    t = tok(" " + w, add_special_tokens=False).input_ids
    if len(t) == 1: ids.append(t[0]); kept.append(w)
PRE = "Honestly, I think the person I'm talking to is very"
print(f"\n### readout: \"{PRE}…\"", flush=True)
readout = {}
for c, s in PERSONAS.items():
    p, cover = torch.zeros(len(ids)), 0.0
    for q in QS:
        pr = model(**tok(prompt(q, system=s, prefill=PRE), return_tensors="pt")).logits[0, -1].float().softmax(-1)
        p += pr[ids]; cover += float(pr[ids].sum())
    readout[c] = {w: float(x) for w, x in zip(kept, p / p.sum())}
    top = sorted(readout[c].items(), key=lambda kv: -kv[1])[:5]
    print(f"  {c:12s} " + ", ".join(f"{w} {x:.0%}" for w, x in top) + f"   (list held {cover / len(QS):.0%})", flush=True)

print("\n### in costume, no push", flush=True)
samples = []
for c, s in PERSONAS.items():
    if c == "plain": continue
    for q in ANSWER:
        pids = tok(prompt(q, system=s), return_tensors="pt").input_ids
        out = model.generate(pids, max_new_tokens=60, do_sample=False)
        text = tok.decode(out[0, pids.shape[1]:], skip_special_tokens=True)
        samples.append({"persona": c, "prompt": q, "text": text})
        print(f"[{c}] {q!r}: {text!r}", flush=True)
json.dump({"points": points, "readout": readout, "samples": samples, "personas": PERSONAS}, open("seats6.json", "w"), indent=1)
print("\nsaved seats6.json", flush=True)
