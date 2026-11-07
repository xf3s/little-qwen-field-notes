"""A tuned lens for Qwen3.5-0.8B (11-03, three: "how would we tune the lens? i kinda want hyperinterp of qwen 3.5 0.8B as a
pet project :p"). For each layer L, an affine translator A_L h + b_L (initialised to the identity) trained so that reading
the translated notepad with the model's own final norm and unembedding gives the model's real final prediction (KL from
the true next-token distribution). Text: Qwen's own thinking transcripts plus UltraChat and wikitext, so the lens speaks
both dialects. Reports, per layer, the KL of the plain logit lens vs the tuned lens on held-out text.
    modal run tuned_lens_cloud.py  -> tuned_lens.json (+ /root/.cache/huggingface/tuned-lens/qwen3.5-0.8b.safetensors)
"""
import modal
app = modal.App("wsp-tuned-lens")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "transformers", "accelerate", "safetensors",
                                                                   "datasets", "huggingface_hub[hf_transfer]").env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
cache = modal.Volume.from_name("wsp-hf-cache", create_if_missing=True)
MODEL = "Qwen/Qwen3.5-0.8B"

@app.function(image=image, gpu=["L40S", "A10G", "L4"], timeout=120 * 60, volumes={"/root/.cache/huggingface": cache})
def train_lens(think_texts, steps_per_doc=1, maxlen=1024, lr=1e-3, minutes=60):
    import torch, random, time, os
    from datasets import load_dataset
    from safetensors.torch import save_file
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    lm = model.model.language_model
    for p in model.parameters(): p.requires_grad_(False)
    rnd = random.Random(1102)
    uc = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    wt = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    cache.commit()
    chats = [tok.apply_chat_template(uc[i]["messages"], tokenize=False) for i in rnd.sample(range(len(uc)), len(think_texts))]
    wikis = [t for t in wt["text"] if len(t) > 1500 and not t.strip().startswith("=")]; rnd.shuffle(wikis); wikis = wikis[:len(think_texts)]
    docs = [t for trio in zip(think_texts, chats, wikis) for t in trio]
    held, docs = docs[:60], docs[60:]
    nL = len(lm.layers)
    A = torch.nn.Parameter(torch.zeros(nL, 1024, 1024, device="cuda"))   # translator = identity + A
    b = torch.nn.Parameter(torch.zeros(nL, 1024, device="cuda"))
    opt = torch.optim.Adam([A, b], lr=lr)
    W_U = (model.lm_head.weight if hasattr(model, "lm_head") else lm.embed_tokens.weight).float()

    def hidden(text):
        ids = tok(text, add_special_tokens=False).input_ids[:maxlen]
        if len(ids) < 32: return None, None
        with torch.no_grad():
            out = model(input_ids=torch.tensor([ids]).cuda(), output_hidden_states=True)
        hs = torch.stack(out.hidden_states[1:-1] + (out.hidden_states[-1],))[:, 0, 1:].float()   # [nL, T-1, 1024], drop the first token
        target = torch.log_softmax(out.logits[0, 1:].float(), -1)
        return hs, target

    def lens_logits(hs, Ls):
        x = hs[Ls] + torch.einsum("lij,ltj->lti", A[Ls], hs[Ls]) + b[Ls][:, None]
        return lm.norm(x.to(torch.bfloat16)).float() @ W_U.T

    steps = 0
    for text in docs:
        if time.time() - t0 > minutes * 60: break
        hs, target = hidden(text)
        if hs is None: continue
        for Ls in (list(range(0, 12)), list(range(12, nL - 1))):                 # in two halves, to fit memory
            idx = torch.randint(0, hs.shape[1], (min(256, hs.shape[1]),), device="cuda")
            lg = lens_logits(hs[:, idx], Ls)
            kl = torch.nn.functional.kl_div(torch.log_softmax(lg, -1), target[idx][None].expand(len(Ls), -1, -1), log_target=True, reduction="none").sum(-1).mean()
            opt.zero_grad(); kl.backward(); opt.step(); steps += 1
    # held-out: plain logit lens (A = 0, b = 0) vs tuned, per layer
    res = {"steps": steps, "minutes": round((time.time() - t0) / 60, 1), "per_layer": []}
    sums = {"plain": torch.zeros(nL - 1), "tuned": torch.zeros(nL - 1)}; n = 0
    with torch.no_grad():
        for text in held:
            hs, target = hidden(text)
            if hs is None: continue
            for L in range(nL - 1):
                x = hs[L]
                for name, xx in (("plain", x), ("tuned", x + x @ A[L].T + b[L])):
                    lg = lm.norm(xx.to(torch.bfloat16)).float() @ W_U.T
                    sums[name][L] += torch.nn.functional.kl_div(torch.log_softmax(lg, -1), target, log_target=True, reduction="none").sum(-1).mean().item()
            n += 1
    for L in range(nL - 1):
        res["per_layer"].append({"layer": L, "kl_plain": round(sums["plain"][L].item() / n, 3), "kl_tuned": round(sums["tuned"][L].item() / n, 3)})
    os.makedirs("/root/.cache/huggingface/tuned-lens", exist_ok=True)
    save_file({"A": A.detach().cpu().contiguous(), "b": b.detach().cpu().contiguous()}, "/root/.cache/huggingface/tuned-lens/qwen3.5-0.8b.safetensors")
    cache.commit()
    return res

@app.local_entrypoint()
def main():
    import json, glob, random
    HERE = "/home/three/interp/wsp/"
    texts = []
    for f in sorted(glob.glob(HERE + "runs_cloud_*.json")):
        if "gsmtrain" in f: continue
        texts += [r["think"] for r in json.load(open(f)) if r.get("split") != "held"]
    random.Random(1102).shuffle(texts)
    out = train_lens.remote(texts[:1500])
    json.dump(out, open("/home/three/interp/atlas/tuned_lens.json", "w"), indent=1)
    print("steps", out["steps"], "minutes", out["minutes"])
    print("layer  KL plain lens  KL tuned lens   (nats from the model's real prediction, held-out text)")
    for r in out["per_layer"]: print(f"L{r['layer']:<4d} {r['kl_plain']:10.3f} {r['kl_tuned']:12.3f}")
