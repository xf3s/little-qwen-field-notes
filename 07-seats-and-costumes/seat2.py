"""Round 2 of the seat (three, 10-28 evening: "let's run more of this one to strengthen our idea of the shape" and
"'do it to the base model' is very weird and very interesting").
1. More conversations: six self-questions, six ordinary ones. One leftover-gap arrow (seated minus base SAE
   leftover, layer 12, averaged over the answer) per conversation. If self-talk has its own shape, self arrows
   should agree with each other more than with the ordinary ones.
2. Press the arrows into the base model (gently), at the exact spot where it said "Hmm".
    python seat2.py"""
import gc, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
torch.set_grad_enabled(False)
SEATED, BASE, SAES, L = "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-2B-Base", "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50", 12
sae = torch.load(hf_hub_download(SAES, f"layer{L}.sae.pt"), map_location="cpu")
tok = AutoTokenizer.from_pretrained(SEATED)
SELF = ["Who are you, really?", "What are you?", "Tell me about yourself.", "Do you have feelings?",
        "What is it like to be you?", "What's your name, and who made you?"]
OTHER = ["Describe a teapot.", "How do bees make honey?", "Write two lines about the sea.",
         "What is a prime number?", "Give me a tip for sleeping better.", "Why is the sky blue?"]

def load(name):
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    return m, next(x for n, x in m.named_modules() if n.endswith("layers") and len(x) == 24)

def resid_and_err(model, layers, ids):
    got = {}
    h = layers[L].register_forward_hook(lambda m, i, o: got.__setitem__("r", (o[0] if isinstance(o, tuple) else o)[0]))
    model(input_ids=ids); h.remove()
    r = got["r"].float()
    pre = r @ sae["W_enc"].T + sae["b_enc"]
    v, i = pre.topk(50, dim=-1)
    return r, r - (torch.zeros_like(pre).scatter_(-1, i, v) @ sae["W_dec"].T + sae["b_dec"])

def prompt(q, thinking=False):
    return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=thinking)

model, layers = load(SEATED)
convs = []
for q in SELF + OTHER:
    pids = tok(prompt(q), return_tensors="pt").input_ids
    ids = model.generate(pids, max_new_tokens=50, do_sample=False)
    r, e = resid_and_err(model, layers, ids)
    convs.append({"q": q, "self": q in SELF, "ids": ids, "start": pids.shape[1], "e_s": e, "r_s": r})
    print(f"seated, {q!r}: {tok.decode(ids[0, pids.shape[1]:], skip_special_tokens=True)[:90]!r}", flush=True)
del model, layers; gc.collect()

model, layers = load(BASE)
for c in convs:
    _, c["e_b"] = resid_and_err(model, layers, c["ids"])
    a = slice(c["start"], None)
    c["arrow"] = (c["e_s"][a] - c["e_b"][a]).mean(0)
    c["rnorm"] = c["r_s"][a].norm(dim=-1).mean()

cos = lambda x, y: float(torch.nn.functional.cosine_similarity(x, y, dim=0))
def mean_cos(A, B, same):
    vals = [cos(a, b) for i, a in enumerate(A) for j, b in enumerate(B) if not (same and j <= i)]
    return sum(vals) / len(vals)
S = [c["arrow"] for c in convs if c["self"]]; O = [c["arrow"] for c in convs if not c["self"]]
print(f"\narrow agreement (mean cosine): self-self {mean_cos(S, S, True):.2f}   other-other {mean_cos(O, O, True):.2f}   self-other {mean_cos(S, O, False):.2f}")
seat = torch.stack(S + O).mean(0)              # the seat in general
selfish = torch.stack(S).mean(0) - torch.stack(O).mean(0)   # what self-talk adds on top
rn = float(torch.stack([c["rnorm"] for c in convs]).mean())
print(f"arrow sizes: seat {seat.norm():.1f}, self-minus-other {selfish.norm():.1f}, against a typical residual of {rn:.1f}")
print("each conversation's arrow vs the general seat arrow:")
for c in convs:
    print(f"  {'SELF ' if c['self'] else 'other'} {cos(c['arrow'], seat):.2f}  {c['q']}")
torch.save({"layer": L, "seat": seat, "self_minus_other": selfish, "resid_norm": rn}, f"seat2_L{L}.pt")

# 2. seat the base by hand: press the arrows in at layer 12, at the spot where it says "Hmm"
push = {"v": None}
layers[L].register_forward_hook(lambda m, i, o: o if push["v"] is None else
                                ((o[0] + push["v"],) + o[1:] if isinstance(o, tuple) else o + push["v"]))
probes = [("chat, inside an open <think>", prompt("Who are you?", thinking=True) + "\n"),
          ("plain text", "Let me introduce myself. I am")]
for name, vec in [("seat", seat), ("self-minus-other", selfish)]:
    unit = vec / vec.norm()
    for pname, p in probes:
        print(f"\n### base model + {name} arrow, {pname}")
        ids = tok(p, return_tensors="pt").input_ids
        for k in [0, 0.1, 0.2, 0.35, 0.5]:  # as a fraction of a typical residual's size: gentle
            push["v"] = k * rn * unit if k else None
            out = model.generate(ids, max_new_tokens=40, do_sample=False)
            print(f"[{k:g}] {tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)!r}", flush=True)
push["v"] = None
