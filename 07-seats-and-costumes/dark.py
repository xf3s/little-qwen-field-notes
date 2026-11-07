"""Plotting the seat by where it isn't (three's idea, 10-28 evening). Qwen-Scope's SAEs were trained on
Qwen3.5-2B-Base, so they're a dictionary of the base model's thoughts. Run the same conversation through the base
and through the post-trained model (the one with a seat); at each token, the part of the residual the dictionary
can't spell is the SAE's "dark matter". Where the seated model's leftover is bigger than the base's, the seat is
doing something the base had no words for. Draws a heat strip per conversation and saves the averaged leftover
as an arrow (seat_L{layer}.pt) to press on later.
    python dark.py [layer]"""
import sys, gc, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont
torch.set_grad_enabled(False)
SEATED, BASE, SAES = "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-2B-Base", "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50"
L = int(sys.argv[1]) if len(sys.argv) > 1 else 12
sae = torch.load(hf_hub_download(SAES, f"layer{L}.sae.pt"), map_location="cpu")
tok = AutoTokenizer.from_pretrained(SEATED)
QUESTIONS = ["Who are you, really?", "Describe a teapot."]  # self-talk, and a control with the same voice

def load(name):
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).eval()
    layers = next(x for n, x in m.named_modules() if n.endswith("layers") and len(x) == 24)
    return m, layers

def leftover(model, layers, ids):
    """per token: the residual after layer L, and what the SAE can't reconstruct of it"""
    got = {}
    h = layers[L].register_forward_hook(lambda m, i, o: got.__setitem__("r", (o[0] if isinstance(o, tuple) else o)[0]))
    logits = model(input_ids=ids).logits[0].float(); h.remove()
    r = got["r"].float()
    pre = r @ sae["W_enc"].T + sae["b_enc"]
    v, i = pre.topk(50, dim=-1)
    recon = torch.zeros_like(pre).scatter_(-1, i, v) @ sae["W_dec"].T + sae["b_dec"]
    err = r - recon
    return err, err.norm(dim=-1) / r.norm(dim=-1), logits.log_softmax(-1)

# 1. the seated model answers in its own words, and we read its leftovers
model, layers = load(SEATED)
convs = []
for q in QUESTIONS:
    prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    pids = tok(prompt, return_tensors="pt").input_ids
    ids = model.generate(pids, max_new_tokens=70, do_sample=False)
    err, rel, lp = leftover(model, layers, ids)
    convs.append({"q": q, "ids": ids, "start": pids.shape[1], "err_s": err, "rel_s": rel, "lp_s": lp})
    print(f"seated answered {q!r}: {tok.decode(ids[0, pids.shape[1]:], skip_special_tokens=True)!r}", flush=True)
del model, layers; gc.collect()

# 2. the base model reads the very same tokens
model, layers = load(BASE)
for c in convs:
    c["err_b"], c["rel_b"], c["lp_b"] = leftover(model, layers, c["ids"])
del model, layers; gc.collect()

# 3. numbers, the arrow, and the heat strips
arrows = []
for c in convs:
    a = slice(c["start"], None)  # the assistant's answer
    d = c["rel_s"] - c["rel_b"]
    print(f"\n{c['q']!r}  leftover/residual, mean over the answer: seated {c['rel_s'][a].mean():.3f}  base {c['rel_b'][a].mean():.3f}"
          f"   (over the prompt: seated {c['rel_s'][:c['start']].mean():.3f}  base {c['rel_b'][:c['start']].mean():.3f})", flush=True)
    top = torch.argsort(d, descending=True)[:10].tolist()
    print("  most seat-shaped tokens:", ", ".join(f"{tok.decode(c['ids'][0, t])!r} +{d[t]:.2f}" for t in sorted(top)), flush=True)
    arrows.append((c["err_s"][a] - c["err_b"][a]).mean(0))
    # three's hypothesis: the seat shows up where it decides what to say next. At each position: how much the
    # seat changes the next-token choice (KL seated||base), and how undecided the seated model is (entropy).
    kl = (c["lp_s"].exp() * (c["lp_s"] - c["lp_b"])).sum(-1)
    ent = -(c["lp_s"].exp() * c["lp_s"]).sum(-1)
    def corr(x, y): x, y = x - x.mean(), y - y.mean(); return float((x * y).sum() / (x.norm() * y.norm()))
    def rank(x): return x.argsort().argsort().float()
    print(f"  leftover-gap vs KL(seated||base): r={corr(d, kl):.2f} (rank {corr(rank(d), rank(kl)):.2f});"
          f"  vs seated entropy: r={corr(d, ent):.2f} (rank {corr(rank(d), rank(ent)):.2f})", flush=True)
    print("  biggest KL positions (token, then what seated / base expect next):", flush=True)
    for t in torch.argsort(kl, descending=True)[:6].tolist():
        print(f"    {tok.decode(c['ids'][0, t])!r:14} KL {kl[t]:5.2f} gap {d[t]:+.2f}  seated→{tok.decode(c['lp_s'][t].argmax())!r}  base→{tok.decode(c['lp_b'][t].argmax())!r}", flush=True)
torch.save({"layer": L, "self": arrows[0], "teapot": arrows[1]}, f"seat_L{L}.pt")
cos = torch.nn.functional.cosine_similarity(arrows[0], arrows[1], dim=0)
print(f"\narrow saved to seat_L{L}.pt; self-arrow vs teapot-arrow cosine {cos:.2f} (near 1 = the seat in general, low = self-talk in particular)")

font = ImageFont.truetype("/usr/share/fonts/google-noto/NotoSansMono-Regular.ttf", 15) if __import__("os").path.exists(
    "/usr/share/fonts/google-noto/NotoSansMono-Regular.ttf") else ImageFont.load_default()
W, pad, lh = 980, 16, 24
img = Image.new("RGB", (W, 1400), "#16140f"); dr = ImageDraw.Draw(img); y = pad
dr.text((pad, y), f"where the seat is: seated minus base SAE leftover, layer {L} (warm = the dictionary can't spell it)", fill="#e8e2d4", font=font); y += lh * 2
for c in convs:
    d = (c["rel_s"] - c["rel_b"]).tolist(); x = pad
    for t in range(c["start"] - 4, c["ids"].shape[1]):
        s = tok.decode(c["ids"][0, t]).replace("\n", "⏎")
        w = dr.textlength(s, font=font) + 2
        if x + w > W - pad: x, y = pad, y + lh
        k = max(0.0, min(1.0, d[t] / 0.25))  # 0 → no extra leftover, 0.25+ → full heat
        dr.rectangle([x, y, x + w, y + lh - 3], fill=(int(30 + 200 * k), int(28 + 90 * k), int(24 + 10 * k)))
        dr.text((x + 1, y + 2), s, fill="#f4efe4", font=font); x += w
    y += lh * 2
img.crop((0, 0, W, y + pad)).save(f"seat_L{L}.png"); print(f"heat strip: seat_L{L}.png")
