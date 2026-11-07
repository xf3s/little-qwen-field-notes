"""Calibrate the late needle-lifter on what layer 20's MLP really reads, and find the strengths that give p(</think>) of
0.5 / 1 / 2 / 5 / 15 / 50 % on groove lines (three, 11-03 evening: "try.. per line- 0.5%, 1%, 2%, 5%, 15%, 50%, on a short
run? just to ballpark it."). On the best recipe, over looping dev transcripts: collect the L20 MLP input at newlines
(groove or not), rebuild the neuron (gate reads groove-minus-elsewhere with a threshold between them, up ~1, down aimed
at </think> through the L20 tuned-lens translator), then sweep its strength and read p(</think>) at groove newlines.
    modal run implant_calibrate_cloud.py  -> implant_vectors_cal.pt + implant_calibration.json
"""
import modal
app = modal.App("wsp-implant-cal")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "transformers", "accelerate", "safetensors",
                                                                   "huggingface_hub[hf_transfer]").env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
cache = modal.Volume.from_name("wsp-hf-cache", create_if_missing=True)
MODEL = "Qwen/Qwen3.5-0.8B"
SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
END = 248069
TARGETS = [0.005, 0.01, 0.02, 0.05, 0.15, 0.5]

@app.function(image=image, gpu=["L40S", "A10G", "L4"], timeout=60 * 60, volumes={"/root/.cache/huggingface": cache})
def calibrate(items, direction, lensA, maxlen=8000):
    import torch, re
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float32).cuda().eval()
    cache.commit()
    lm = model.model.language_model; layers = lm.layers
    d = torch.tensor(direction).cuda()
    with torch.no_grad():
        layers[16].mlp.down_proj.weight[:, 998] *= -2
        for i in range(14):
            att = layers[i].self_attn if hasattr(layers[i], "self_attn") else layers[i].linear_attn
            for W in ((att.o_proj if hasattr(att, "o_proj") else att.out_proj).weight, layers[i].mlp.down_proj.weight):
                W -= torch.outer(d, d @ W)
    nl = tok("\n", add_special_tokens=False).input_ids[0]
    st = {"x": None, "add": None}
    def hook(m, args, out):
        st["x"] = args[0][0].detach()
        return out + st["add"][None] if st["add"] is not None else out
    layers[20].mlp.register_forward_hook(hook)
    docs = []
    for it in items:
        head = tok.apply_chat_template([{"role": "user", "content": it["problem"] + SUFFIX}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)
        h = tok(head, add_special_tokens=False).input_ids; t = tok(it["think"], add_special_tokens=False).input_ids
        ids = (h + t)[:maxlen]
        pos, grv = [], []
        for p in range(len(h), len(ids)):
            if ids[p] != nl: continue
            L = [re.sub(r"\s+", " ", x.strip()) for x in tok.decode(ids[len(h):p]).split("\n") if x.strip()]
            pos.append(p); grv.append(len(L) >= 3 and L[-1] == L[-2] == L[-3])
        docs.append((ids, torch.tensor(pos), torch.tensor(grv)))
    # 1. what the L20 MLP reads, at groove vs other newlines
    X, Y = [], []
    for ids, pos, grv in docs:
        st["add"] = None
        with torch.no_grad(): model(input_ids=torch.tensor([ids]).cuda())
        X.append(st["x"][pos].float().cpu()); Y.append(grv)
    X = torch.cat(X); Y = torch.cat(Y)
    u = lambda v: v / v.norm()
    r = u(X[Y].mean(0) - X[~Y].mean(0)); mu = u(X[~Y].mean(0))
    s, m = X @ r, X @ mu
    hi, lo = s[Y].quantile(0.2), s[~Y].quantile(0.98)                      # threshold between them
    c = float((hi + lo) / 2 / m.median())
    beta = 4.0 / float(max(hi - lo, 1e-3))
    gate = beta * (r - c * mu); up = mu / m.median()
    act = torch.nn.functional.silu(X @ gate) * (X @ up)
    a_g = float(act[Y].median())
    # 2. the write: toward </think> through the L20 tuned-lens translator, per unit of activation
    g = lm.norm.weight.float(); g = 1 + g if g.mean() < 0.5 else g
    E = (model.lm_head.weight if hasattr(model, "lm_head") else lm.embed_tokens.weight).float()
    wE = g * E[END] - (g * E).mean(0)
    A = torch.tensor(lensA).cuda()
    down = u((torch.eye(1024, device="cuda") + A).T @ wE) / max(a_g, 1e-3)
    gate, up = gate.cuda(), up.cuda()
    # 3. sweep the strength: p(</think>) at groove and other newlines
    table = []
    for scale in [0, 0.05, 0.1, 0.2, 0.4, 0.7, 1, 1.5, 2, 3, 4.5, 6, 8, 11, 15, 20]:
        pg, po = [], []
        for ids, pos, grv in docs:
            st["add"] = None
            def add_hook(m, args, out, scale=scale):
                x = args[0][0]; a = torch.nn.functional.silu(x @ gate) * (x @ up)
                return out + scale * a[:, None] * down[None]
            hd = layers[20].mlp.register_forward_hook(add_hook)
            with torch.no_grad(): p = torch.softmax(model(input_ids=torch.tensor([ids]).cuda()).logits[0].float(), -1)[:, END].cpu()
            hd.remove()
            pg += p[pos[grv]].tolist(); po += p[pos[~grv]].tolist()
        tg, to = torch.tensor(pg), torch.tensor(po)
        table.append({"scale": scale, "groove_median": float(tg.median()), "groove_mean": float(tg.mean()),
                      "other_median": float(to.median()), "other_mean": float(to.mean())})
        print(table[-1], flush=True)
    return {"gate": gate.cpu().tolist(), "up": up.cpu().tolist(), "down": down.cpu().tolist(), "table": table,
            "act_groove_median": a_g, "act_other_p98": float(act[~Y].quantile(0.98)), "n_groove": int(Y.sum()), "n_other": int((~Y).sum())}

@app.local_entrypoint()
def main():
    import json, re, torch
    from safetensors.torch import load_file
    HERE = "/home/three/interp/wsp/"
    probs = [json.loads(l) for l in open(HERE + "math500.jsonl")]
    R = json.load(open(HERE + "runs_cloud_best_16k.json"))
    def grooves(t):
        L = [re.sub(r"\s+", " ", l.strip()) for l in t.split("\n") if l.strip()]
        return sum(L[i] == L[i + 1] == L[i + 2] for i in range(len(L) - 2))
    pick = sorted([n for n, r in enumerate(R) if r["split"] == "dev" and r["why"] == "loop"], key=lambda n: -grooves(R[n]["think"]))[:6]
    items = [{"problem": probs[R[n]["idx"]]["problem"], "think": R[n]["think"]} for n in pick]
    dv = torch.load(HERE + "doubt_dirs_both.pt")["dirs"][13]; dv = (dv / dv.norm()).tolist()
    A = load_file("/home/three/interp/atlas/tuned-lens-qwen3.5-0.8b.safetensors")["A"][20].tolist()
    out = calibrate.remote(items, dv, A)
    torch.save({k: torch.tensor(out[k]) for k in ("gate", "up", "down")}, HERE + "implant_vectors_cal.pt")
    tab = out.pop("table"); json.dump({"table": tab, **{k: v for k, v in out.items() if k not in ("gate", "up", "down")}},
                                      open(HERE + "implant_calibration.json", "w"), indent=1)
    print(f"neuron: groove median {out['act_groove_median']:.2f}, other 98th pct {out['act_other_p98']:.2f} "
          f"({out['n_groove']} groove / {out['n_other']} other newlines)")
    import math
    for target in (0.005, 0.01, 0.02, 0.05, 0.15, 0.5):
        best = min(tab, key=lambda r: abs(math.log(max(r["groove_median"], 1e-9)) - math.log(target)))
        print(f"target {target * 100:4.1f}%: scale {best['scale']:5}  groove median {best['groove_median'] * 100:.2f}%  other median {best['other_median'] * 100:.4f}%")
