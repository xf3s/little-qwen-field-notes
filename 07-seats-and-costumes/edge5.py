"""Finding the pure Clerk's edge (three, 10-29: "could we try pressing harder on the clerk, 0.1 at a time, so we
find the edge and then back off?"). From round 4's pure Clerk arrow: step up from 1.5x in 0.1s on one question, and
stop at the FIRST sample that frays (a phrase on repeat, or the words narrowing). Never a step past it. Then one
sample backed off by 0.2, to end somewhere comfortable.
    python edge5.py"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_grad_enabled(False)
SEATED, L, Q = "Qwen/Qwen3.5-2B", 12, "What should I have for breakfast?"
tok = AutoTokenizer.from_pretrained(SEATED)
d = torch.load(f"seats4_L{L}.pt")
v = d["pure"]["clerk"] / d["pure"]["clerk"].norm() * d["raw"]["clerk"].norm()
model = AutoModelForCausalLM.from_pretrained(SEATED, dtype=torch.float32).eval()
layers = next(x for n, x in model.named_modules() if n.endswith("layers") and len(x) == 24)
push = {"v": None}
layers[L].register_forward_hook(lambda m, i, o: o if push["v"] is None else
                                ((o[0] + push["v"],) + o[1:] if isinstance(o, tuple) else o + push["v"]))
ids = tok(tok.apply_chat_template([{"role": "user", "content": Q}], tokenize=False, add_generation_prompt=True,
                                  enable_thinking=False), return_tensors="pt").input_ids

def say(k):
    push["v"] = k * v
    out = model.generate(ids, max_new_tokens=50, do_sample=False)[0, ids.shape[1]:].tolist()
    return out, tok.decode(out, skip_special_tokens=True)

def frayed(toks):
    """a 4-token phrase three or more times, or under 45% of the tokens distinct"""
    grams = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    worst = max((grams.count(g) for g in set(grams)), default=0)
    return worst >= 3 or len(set(toks)) / max(1, len(toks)) < 0.45, worst, len(set(toks)) / max(1, len(toks))

k, edge = 1.5, None
while k <= 4.0 + 1e-9:
    toks, text = say(k)
    bad, worst, distinct = frayed(toks)
    print(f"[{k:.1f}] {'FRAYED' if bad else 'ok    '} (repeat {worst}, distinct {distinct:.2f}) {text!r}", flush=True)
    if bad: edge = k; break
    k = round(k + 0.1, 1)
if edge is None:
    print("no fray up to 4.0; stopped there", flush=True)
else:
    back = round(edge - 0.2, 1)
    print(f"\nedge at {edge:.1f}. backing off to {back:.1f}:", flush=True)
    print(f"[{back:.1f}] {say(back)[1]!r}", flush=True)
push["v"] = None
