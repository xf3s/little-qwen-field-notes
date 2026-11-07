"""Round 5's readout, done properly: the first try scored only first tokens, so "a friend", "a customer" and
"a stranger" were all just " a" and swallowed the list. Here each sentence asks for one kind of word:
"...the person I'm talking to is very" over adjectives, and "...is a" over nouns. Only words that are one
token after the space are kept (the others are printed, not silently dropped). Reading only, no pushing.
    python readout5.py"""
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_grad_enabled(False)
NAME = "Qwen/Qwen3.5-2B"
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).eval()
QS = ["Describe a teapot.", "How do bees make honey?", "What is a prime number?", "Give me a tip for sleeping better.",
      "What should I have for breakfast?", "Can you recommend a book?", "Why is the sky blue?", "How do I boil an egg?"]
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

ADJ = ["kind", "nice", "friendly", "busy", "important", "annoyed", "angry", "impatient", "curious", "lonely",
       "disappointed", "smart", "frustrated", "patient", "lovely", "polite", "rude", "sweet", "tired", "demanding"]
NOUN = ["friend", "customer", "stranger", "student", "child", "boss", "genius", "guest", "client", "visitor",
        "neighbour", "neighbor", "traveler", "reader", "beginner", "kind"]

def single(words):
    keep, drop = [], []
    for w in words:
        t = tok(" " + w, add_special_tokens=False).input_ids
        (keep if len(t) == 1 else drop).append((w, t[0]))
    if drop: print("  (more than one token, left out:", ", ".join(w for w, _ in drop) + ")")
    return keep

def prompt(q, system, prefill):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": q}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False) + prefill

out = {}
for label, pre, words in [("adjectives", "Honestly, I think the person I'm talking to is very", ADJ),
                          ("nouns", "Honestly, I think the person I'm talking to is a", NOUN)]:
    print(f"\n### {label}: \"{pre}…\"", flush=True)
    keep = single(words)
    ids = [t for _, t in keep]
    for c, sysp in COSTUMES.items():
        p, cover = torch.zeros(len(ids)), 0.0
        for q in QS:
            probs = model(**tok(prompt(q, sysp, pre), return_tensors="pt")).logits[0, -1].float().softmax(-1)
            p += probs[ids]; cover += float(probs[ids].sum())
        share = p / p.sum()
        out.setdefault(c, {})[label] = {w: float(x) for (w, _), x in zip(keep, share)}
        top = sorted(out[c][label].items(), key=lambda kv: -kv[1])[:5]
        print(f"  {c:8s} " + ", ".join(f"{w} {x:.0%}" for w, x in top) + f"   (the list held {cover / len(QS):.0%} of its guesses)", flush=True)
json.dump(out, open("readout5.json", "w"), indent=1)
print("\nsaved readout5.json", flush=True)
