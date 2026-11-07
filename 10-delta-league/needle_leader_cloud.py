"""The late needle-lifter, recalibrated on Delta's 37.2 leader (cd0b95bb) and swapped into its least-used L20 neuron
(11-06, pre-registered; three: "i think that's worth a submission :3"). Delta can't take a widened MLP, so the neuron
replaces one that barely writes anything. Same transcripts and recipe as implant_calibrate_cloud.py; the strength sweep
uses the swapped weights themselves, not a hook.
    modal volume put wsp-hf-cache leader3diff.safetensors /leader3diff.safetensors   (the leader's changed tensors)
    modal run needle_leader_cloud.py  -> needle_leader.pt + needle_leader.json
"""
import modal
app = modal.App("wsp-needle-leader")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "transformers", "accelerate", "safetensors",
                                                                   "huggingface_hub[hf_transfer]").env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
cache = modal.Volume.from_name("wsp-hf-cache", create_if_missing=True)
MODEL = "Qwen/Qwen3.5-0.8B"
SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
END = 248069

@app.function(image=image, gpu=["L40S", "A10G", "L4"], timeout=60 * 60, volumes={"/root/.cache/huggingface": cache})
def calibrate(items, lensA, maxlen=8000):
    import torch, re
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    from safetensors.torch import load_file
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float32).cuda().eval()
    lm = model.model.language_model; layers = lm.layers
    diff = load_file("/root/.cache/huggingface/leader3diff.safetensors")
    P = dict(model.named_parameters())
    with torch.no_grad():
        n = 0
        for k, v in diff.items():
            if k.startswith("embed_"): continue
            P[k].copy_(v.float()); n += 1
        lm.embed_tokens.weight[diff["embed_rows"].long()] = diff["embed_vals"].float().cuda(); n += 1
    print("leader tensors applied:", n, "tied head:", model.lm_head.weight.data_ptr() == lm.embed_tokens.weight.data_ptr(), flush=True)
    mlp = layers[20].mlp
    nl = tok("\n", add_special_tokens=False).input_ids[0]
    st = {"x": None}
    mlp.register_forward_hook(lambda m, args, out: st.__setitem__("x", args[0][0].detach()))
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
        sel = sorted(set(pos) | set(range(len(h), len(ids), 32)))
        where = {p: i for i, p in enumerate(sel)}
        docs.append((ids, torch.tensor([where[p] for p in pos]), torch.tensor(grv), torch.tensor(sel).cuda()))
    def run_all():
        # log-probs only at newlines and every 32nd token (full logits are 8k x 248k floats per doc)
        outs = []
        for ids, pos, grv, sel in docs:
            with torch.no_grad():
                h = model.model(input_ids=torch.tensor([ids]).cuda()).last_hidden_state[0]
                lp = torch.cat([torch.log_softmax(model.lm_head(h[sel[i:i + 256]]).float(), -1).half().cpu() for i in range(0, len(sel), 256)])
            outs.append((lp, st["x"]))
        return outs
    # 1. the leader's own L20 reading at newlines, and how much each existing neuron writes
    X, Y, base_lp, use, ntok = [], [], [], torch.zeros(3584, device="cuda"), 0
    Wd = mlp.down_proj.weight
    for (lp, x), (ids, pos, grv, sel) in zip(run_all(), docs):
        X.append(x[sel[pos]].float().cpu()); Y.append(grv); base_lp.append(lp.float())
        with torch.no_grad():
            a = torch.nn.functional.silu(mlp.gate_proj(x)) * mlp.up_proj(x)
            use += a.abs().sum(0); ntok += a.shape[0]
    use = (use / ntok) * Wd.norm(dim=0)
    j = int(use.argmin()); print("least-used L20 neuron", j, "use", float(use[j]), "median use", float(use.median()), flush=True)
    X = torch.cat(X); Y = torch.cat(Y)
    u = lambda v: v / v.norm()
    r = u(X[Y].mean(0) - X[~Y].mean(0)); mu = u(X[~Y].mean(0))
    s, m = X @ r, X @ mu
    hi, lo = s[Y].quantile(0.2), s[~Y].quantile(0.98)
    c = float((hi + lo) / 2 / m.median())
    beta = 4.0 / float(max(hi - lo, 1e-3))
    gate = beta * (r - c * mu); up = mu / m.median()
    act = torch.nn.functional.silu(X @ gate) * (X @ up)
    a_g = float(act[Y].median())
    g = lm.norm.weight.float(); g = 1 + g if g.mean() < 0.5 else g
    E = model.lm_head.weight.float()
    wE = g * E[END] - (g * E).mean(0)
    A = torch.tensor(lensA).cuda()
    down = u((torch.eye(1024, device="cuda") + A).T @ wE) / max(a_g, 1e-3)
    old = (mlp.gate_proj.weight[j].clone(), mlp.up_proj.weight[j].clone(), Wd[:, j].clone())
    def setn(gv, uv, dv):
        with torch.no_grad(): mlp.gate_proj.weight[j] = gv; mlp.up_proj.weight[j] = uv; Wd[:, j] = dv
    # 2. is neuron j really idle? zero it alone and compare next-token distributions everywhere
    setn(old[0], old[1], torch.zeros_like(old[2]))
    kl = []
    for (lp, _), lp0 in zip(run_all(), base_lp):
        lp = lp.float(); kl.append((lp0.exp() * (lp0 - lp)).sum(-1))
    kl = torch.cat(kl); print("zeroing it: KL mean", float(kl.mean()), "max", float(kl.max()), flush=True)
    # 3. swap in the needle-lifter and sweep its strength
    G, U = gate.cuda().to(Wd.dtype), up.cuda().to(Wd.dtype)
    table = []
    for scale in [0, 0.4, 0.7, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3]:
        setn(G, U, (scale * down).to(Wd.dtype))
        pg, po, kls = [], [], []
        for (lp, _), (ids, pos, grv, sel), lp0 in zip(run_all(), docs, base_lp):
            lp = lp.float(); p = lp[:, END].exp()
            pg += p[pos[grv]].tolist(); po += p[pos[~grv]].tolist()
            kls.append((lp0.exp() * (lp0 - lp)).sum(-1))
        tg, to, kk = torch.tensor(pg), torch.tensor(po), torch.cat(kls)
        table.append({"scale": scale, "groove_median": float(tg.median()), "groove_mean": float(tg.mean()),
                      "other_median": float(to.median()), "other_mean": float(to.mean()), "kl_mean": float(kk.mean())})
        print(table[-1], flush=True)
    base_other = [float(torch.tensor([float(lp0[p, END].exp()) for p in pos[~grv]]).median()) for (ids, pos, grv, sel), lp0 in zip(docs, base_lp)]
    return {"gate": gate.cpu().tolist(), "up": up.cpu().tolist(), "down": down.cpu().tolist(), "neuron": j, "table": table,
            "use_j": float(use[j]), "use_median": float(use.median()), "zero_kl_mean": float(kl.mean()), "zero_kl_max": float(kl.max()),
            "act_groove_median": a_g, "act_other_p98": float(act[~Y].quantile(0.98)), "n_groove": int(Y.sum()), "n_other": int((~Y).sum()),
            "base_other_median_per_doc": base_other}

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
    A = load_file("/home/three/interp/atlas/tuned-lens-qwen3.5-0.8b.safetensors")["A"][20].tolist()
    out = calibrate.remote(items, A)
    torch.save({k: torch.tensor(out[k]) for k in ("gate", "up", "down")} | {"neuron": out["neuron"]}, HERE + "needle_leader.pt")
    json.dump({k: v for k, v in out.items() if k not in ("gate", "up", "down")}, open(HERE + "needle_leader.json", "w"), indent=1)
    print("done: neuron", out["neuron"], "zero KL", out["zero_kl_mean"])
