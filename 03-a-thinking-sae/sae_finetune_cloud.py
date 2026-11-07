"""Fine-tune the public layer-11 Qwen3.5-0.8B SAE on Qwen's own thinking traces (plan: thinking-sae-plan.md; three, 11-03:
"a thinking SAE for qwen 0.8B totally sounds like a portfolio piece :D"). Starts from decoderesearch/qwen-3.5-saes
(Chanin & Lin): Matryoshka BatchTopK, k=100, 65,536 latents with prefixes 2,048 and 16,384, one global threshold, unit
decoder rows. Trains on activations at model.layers.11 streamed from transcripts (90% of runs; 10% held out by run),
with ~20% UltraChat mixed in so it keeps its home dialect. Then evaluates the public and the tuned SAE side by side
(variance explained, splice loss) on held-out thinking runs, chat and wiki, and saves the tuned weights to the volume.
    modal run sae_finetune_cloud.py  -> sae_finetune.json (+ /root/.cache/huggingface/thinking-sae/layer11.safetensors)
"""
import modal
app = modal.App("wsp-sae-tune")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "transformers", "accelerate", "safetensors",
                                                                   "datasets", "huggingface_hub[hf_transfer]").env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
cache = modal.Volume.from_name("wsp-hf-cache", create_if_missing=True)
MODEL = "Qwen/Qwen3.5-0.8B"
L = 11
PREFIXES = (2048, 16384, 65536)

@app.function(image=image, gpu=["L40S", "A10G", "L4"], timeout=150 * 60, volumes={"/root/.cache/huggingface": cache})
def tune(train_texts, eval_texts, lr=2e-5, batch=4096, buffer_tokens=400_000, maxlen=8192, chat_frac=0.2):
    import torch, random, time, os
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    layer = model.model.language_model.layers[L]
    pub = {k: v.cuda().float() for k, v in load_file(hf_hub_download("decoderesearch/qwen-3.5-saes",
           f"qwen-3.5-0.8b/btk-mat-layer-{L}-k-100/sae_weights.safetensors")).items()}
    rnd = random.Random(1102)
    uc = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    chat_train = [tok.apply_chat_template(uc[i]["messages"], tokenize=False) for i in rnd.sample(range(len(uc)), int(len(train_texts) * chat_frac / (1 - chat_frac)) + 1)]
    ucv = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
    chat_eval = [tok.apply_chat_template(ucv[i]["messages"], tokenize=False) for i in rnd.sample(range(len(ucv)), 40)]
    wt = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    wiki_eval = [t for t in wt["text"] if len(t) > 1500 and not t.strip().startswith("=")][:40]
    cache.commit()
    special = set(tok.all_special_ids)
    grab = {}
    hd = layer.register_forward_hook(lambda m, a, o: grab.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))

    def acts_of(text):
        ids = tok(text, add_special_tokens=False).input_ids[:maxlen]
        if len(ids) < 32: return None
        with torch.no_grad(): model(input_ids=torch.tensor([ids]).cuda())
        keep = torch.tensor([i > 0 and t not in special for i, t in enumerate(ids)], device="cuda")
        return grab["h"][0][keep].float()

    # the SAE being tuned: a copy of the public one
    W_enc = pub["W_enc"].clone().requires_grad_(); b_enc = pub["b_enc"].clone().requires_grad_()
    W_dec = pub["W_dec"].clone().requires_grad_(); b_dec = pub["b_dec"].clone().requires_grad_()
    thr = float(pub["threshold"][0])
    opt = torch.optim.Adam([W_enc, b_enc, W_dec, b_dec], lr=lr, betas=(0.9, 0.999))

    def step(x):
        nonlocal thr
        pre = torch.relu((x - b_dec) @ W_enc + b_enc)
        k_all = 100 * x.shape[0]
        v, idx = pre.flatten().topk(k_all)                       # BatchTopK: the top k*B activations over the batch
        acts = torch.zeros_like(pre).flatten().scatter(0, idx, v).view_as(pre)
        loss = 0.0
        for m in PREFIXES:                                       # Matryoshka: each prefix reconstructs on its own
            rec = acts[:, :m] @ W_dec[:m] + b_dec
            loss = loss + ((rec - x) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            W_dec.data /= W_dec.data.norm(dim=1, keepdim=True)   # unit decoder rows, as published
            thr = thr + 0.01 * (float(v.min()) - thr)            # the global threshold tracks the smallest kept activation
        return float(loss) / len(PREFIXES)

    docs = [("think", t) for t in train_texts] + [("chat", t) for t in chat_train]
    rnd.shuffle(docs)
    buf, n_tok, steps, losses = [], 0, 0, []
    def flush():
        nonlocal buf, steps
        X = torch.cat(buf); X = X[torch.randperm(X.shape[0], device="cuda")]
        for i in range(0, X.shape[0] - batch + 1, batch):
            losses.append(step(X[i:i + batch])); steps += 1
        buf = []
    for _, text in docs:
        a = acts_of(text)
        if a is None: continue
        buf.append(a); n_tok += a.shape[0]
        if sum(b.shape[0] for b in buf) >= buffer_tokens: flush()
        if time.time() - t0 > 110 * 60: break                    # leave room for the eval
    if buf: flush()
    hd.remove()

    tuned = {"W_enc": W_enc.detach(), "b_enc": b_enc.detach(), "W_dec": W_dec.detach(), "b_dec": b_dec.detach(),
             "threshold": torch.full_like(pub["threshold"], thr)}
    os.makedirs("/root/.cache/huggingface/thinking-sae", exist_ok=True)
    save_file({k: v.cpu().contiguous() for k, v in tuned.items()}, f"/root/.cache/huggingface/thinking-sae/layer{L}.safetensors")
    cache.commit()

    # eval: both SAEs on held-out thinking runs, chat and wiki (first and special tokens excluded)
    def run_eval(sae, texts):
        enc = lambda x: (lambda p: p * (p > sae["threshold"]))((x - sae["b_dec"]) @ sae["W_enc"] + sae["b_enc"])
        mode = {"m": "clean"}; S = {"sse": 0.0, "n": 0, "sum": torch.zeros(1024, device="cuda"), "sq": 0.0, "ce": {"clean": 0.0, "sae": 0.0}, "tok": 0, "l0": 0.0}
        def hook(m, a, o):
            h = o[0] if isinstance(o, tuple) else o
            if mode["m"] == "clean": grab["h"] = h.detach(); return None
            new = h.clone(); kp = mode["keep"]
            new[0, kp] = (enc(h[0, kp].float()) @ sae["W_dec"] + sae["b_dec"]).to(h.dtype)
            return (new,) + tuple(o[1:]) if isinstance(o, tuple) else new
        hh = layer.register_forward_hook(hook)
        for text in texts:
            ids = tok(text, add_special_tokens=False).input_ids[:1024]
            if len(ids) < 32: continue
            x_ids = torch.tensor([ids]).cuda()
            kp = torch.tensor([i > 0 and t not in special for i, t in enumerate(ids)], device="cuda"); mode["keep"] = kp
            with torch.no_grad():
                for m in ("clean", "sae"):
                    mode["m"] = m
                    lg = model(input_ids=x_ids).logits[0, :-1].float()
                    S["ce"][m] += torch.nn.functional.cross_entropy(lg, x_ids[0, 1:], reduction="none")[kp[1:]].sum().item()
            h = grab["h"][0, kp].float(); a = enc(h); r = a @ sae["W_dec"] + sae["b_dec"]
            S["sse"] += ((h - r) ** 2).sum().item(); S["n"] += h.shape[0]; S["sum"] += h.sum(0); S["sq"] += (h ** 2).sum().item()
            S["l0"] += (a > 0).sum().item(); S["tok"] += int(kp[1:].sum())
        hh.remove()
        tot = S["sq"] - (S["sum"] ** 2).sum().item() / S["n"]
        return {"var_explained": round(1 - S["sse"] / tot, 3), "ce_clean": round(S["ce"]["clean"] / S["tok"], 3),
                "ce_sae": round(S["ce"]["sae"] / S["tok"], 3), "l0": round(S["l0"] / S["n"], 1), "tokens": S["n"]}
    out = {"train_tokens": n_tok, "steps": steps, "final_threshold": thr, "minutes": round((time.time() - t0) / 60, 1),
           "loss_first": losses[:5], "loss_last": losses[-5:]}
    for name, texts in (("think", eval_texts), ("chat", chat_eval), ("wiki", wiki_eval)):
        for label, sae in (("public", pub), ("tuned", tuned)):
            out[f"{name}/{label}"] = run_eval(sae, texts); print(name, label, out[f"{name}/{label}"], flush=True)
    return out

@app.local_entrypoint()
def main():
    import json, glob, hashlib, os
    HERE = "/home/three/interp/wsp/"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/home/three/models/qwen3.5-0.8b")
    SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
    banks = {"math": [json.loads(l) for l in open(HERE + "math500.jsonl")], "gsm": [json.loads(l) for l in open(HERE + "gsm8k60.jsonl")],
             "aime": [json.loads(l) for l in open(HERE + "aime.jsonl")]}
    train, ev = [], []
    for f in sorted(glob.glob(HERE + "runs_cloud_*.json")):
        name = os.path.basename(f)
        if "gsmtrain" in name: continue
        bank = "gsm" if "_gsm_" in name else "aime" if "_aime_" in name else "math"
        think_on = "nothink" not in name
        for n, r in enumerate(json.load(open(f))):
            if r.get("split") == "held": continue                  # the sealed half stays sealed
            head = tok.apply_chat_template([{"role": "user", "content": banks[bank][r["idx"]]["problem"] + SUFFIX}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=think_on)
            text = head + r["think"] + ("</think>\n\n" + r["answer"] if r.get("answer") else "")
            # split by problem and run file together, so held-out runs share no transcript with training
            key = f"{bank}{r['idx']}"
            (ev if int(hashlib.md5(key.encode()).hexdigest(), 16) % 10 == 0 else train).append(text)
    print(f"{len(train)} training transcripts, {len(ev)} held out (by problem)")
    out = tune.remote(train, ev[:60])
    json.dump(out, open("/home/three/interp/saes/sae_finetune.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if "/" not in k}, indent=1))
    print("\nset/sae          var_expl   loss clean -> with SAE   (increase)")
    for k, v in out.items():
        if "/" in k: print(f"{k:16s} {v['var_explained']:7.3f}   {v['ce_clean']:.3f} -> {v['ce_sae']:.3f}   (+{v['ce_sae'] - v['ce_clean']:.3f})")
