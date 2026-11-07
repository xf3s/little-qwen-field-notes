"""The Wait neuron (11-06, pre-registered; three: "look for something that fires when little qwen reads its own contradiction,
then says wait, like our neuron" / "build it!!"). On the leader: a spare L23 neuron that reads "the number I just wrote is wrong"
(the result-token direction found in errors4) and writes " Wait". Calibrated on 9 problems' quads, checked on the other 3.
    modal run waitneuron_cloud.py  -> waitneuron.pt + waitneuron.json
"""
import json, re
import modal
app = modal.App("wsp-waitneuron")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "transformers", "accelerate", "safetensors",
                                                                   "huggingface_hub[hf_transfer]").env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
cache = modal.Volume.from_name("wsp-hf-cache", create_if_missing=True)
MODEL = "Qwen/Qwen3.5-0.8B"
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."

@app.function(image=image, gpu=["L40S", "A10G", "L4"], timeout=60 * 60, volumes={"/root/.cache/huggingface": cache})
def run(items, test_probs):
    import torch, math
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    from safetensors.torch import load_file
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float32).cuda().eval()
    lm = model.model.language_model
    diff = load_file("/root/.cache/huggingface/leader3diff.safetensors"); P = dict(model.named_parameters())
    with torch.no_grad():
        for k, v in diff.items():
            if not k.startswith("embed_"): P[k].copy_(v.float())
        lm.embed_tokens.weight[diff["embed_rows"].long()] = diff["embed_vals"].float().cuda()
    mlp = lm.layers[23].mlp
    st = {}
    mlp.register_forward_hook(lambda m, a, o: st.__setitem__("x", a[0][0].detach()))
    WAIT = sorted({tok(" Wait", add_special_tokens=False).input_ids[0], tok("Wait", add_special_tokens=False).input_ids[0]})
    docs = []
    for it in items:
        head = tok.apply_chat_template([{"role": "user", "content": it["problem"] + SUFFIX}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)
        enc = tok(head + it["prefix"][:it["rend"]], add_special_tokens=False, return_offsets_mapping=True, return_tensors="pt")
        offs = enc.offset_mapping[0].tolist(); hl = len(tok(head, add_special_tokens=False).input_ids)
        docs.append((enc.input_ids.cuda(), hl, it["kind"].startswith("bad"), it["prob"] in test_probs))
    def forward_all():
        res = []
        for ids, hl, bad, test in docs:
            with torch.no_grad(): lg = model(input_ids=ids).logits[0].float()
            pw = torch.softmax(lg, -1)[:, WAIT].sum(-1).cpu()
            res.append((pw, st["x"].float().cpu(), hl, bad, test))
        return res
    base = forward_all()
    # neuron use over all thinking tokens -> the spare slot
    Wd = mlp.down_proj.weight; use = torch.zeros(Wd.shape[1])
    n = 0
    for ids, hl, bad, test in docs:
        with torch.no_grad():
            model(input_ids=ids); x = st["x"]; a = torch.nn.functional.silu(mlp.gate_proj(x)) * mlp.up_proj(x)
            use += a.abs().sum(0).cpu(); n += a.shape[0]
    use = use / n * Wd.norm(dim=0).cpu(); j = int(use.argmin())
    # reader, fit on training problems' result tokens
    Xr = torch.stack([x[-1] for _, x, _, _, _ in base]); bad = torch.tensor([b for *_, b, _ in base]); test = torch.tensor([t for *_, t in base])
    Xo = torch.cat([x[hl:-1] for _, x, hl, _, _ in base])                     # every other thinking token
    u = lambda v: v / v.norm()
    r = u(Xr[bad & ~test].mean(0) - Xr[~bad & ~test].mean(0)); mu = u(Xo.mean(0))
    s_bad, s_ok, s_oth = Xr[bad & ~test] @ r, Xr[~bad & ~test] @ r, Xo @ r
    hi = s_bad.quantile(0.2); lo = torch.maximum(s_oth.quantile(0.999), s_ok.quantile(0.98))
    m = Xo @ mu; c = float((hi + lo) / 2 / m.median()); beta = 4.0 / float(max(hi - lo, 1e-3))
    gate = beta * (r - c * mu); up = mu / m.median()
    act_bad = torch.nn.functional.silu(Xr[bad & ~test] @ gate) * (Xr[bad & ~test] @ up)
    a_g = float(act_bad.median())
    g = lm.norm.weight.float(); g = 1 + g if g.mean() < 0.5 else g
    E = model.lm_head.weight.float()
    wE = (g * E[WAIT].mean(0)) - (g * E).mean(0)
    down = (wE / wE.norm() / max(a_g, 1e-3)).cuda()
    info = {"neuron": j, "use_j": float(use[j]), "use_median": float(use.median()), "hi": float(hi), "lo": float(lo),
            "sep_train": [float(s_bad.median()), float(s_ok.median()), float(s_oth.quantile(0.999))], "act_bad_median": a_g}
    print(info, flush=True)
    def summary(res):
        out = {}
        for split in ("train", "test"):
            sel = [(pw, hl, b) for pw, _, hl, b, t in res if t == (split == "test")]
            wb = torch.tensor([float(pw[-1]) for pw, hl, b in sel if b]); wo = torch.tensor([float(pw[-1]) for pw, hl, b in sel if not b])
            oth = torch.cat([pw[hl:-1] for pw, hl, b in sel])
            out[split] = {"wrong_median": float(wb.median()), "wrong_mean": float(wb.mean()), "right_median": float(wo.median()), "right_mean": float(wo.mean()),
                          "other_mean": float(oth.mean()), "other_p999": float(oth.quantile(0.999)), "other_frac_over_10pct": float((oth > 0.1).float().mean())}
        return out
    # 12:45: the write direction is FITTED, not aimed: tied embeddings make g*E[Wait] mostly a "common token" direction (the 12:40
    # diagnostic: x100 lifted " Wait" 2.5 -> 37 but " " to 103). Fit v on training problems' wrong result tokens so that adding
    # (act / a_g) * v before the final norm gives p(Wait) ~0.3, then sweep multiples of it with the real weights.
    pre = {}
    hk = lm.norm.register_forward_hook(lambda m, a_, o: pre.__setitem__("h", a_[0][0, -1].detach()))
    Hs, As = [], []
    for ids, hl, isbad, istest in docs:
        if not isbad or istest: continue
        with torch.no_grad():
            model(input_ids=ids); x = st["x"][-1]
            As.append(float(torch.nn.functional.silu(x @ gate.cuda()) * (x @ up.cuda()))); Hs.append(pre["h"].float())
    hk.remove()
    Hs = torch.stack(Hs); As = torch.tensor(As, device="cuda")[:, None] / a_g
    v = torch.zeros(Hs.shape[1], device="cuda", requires_grad=True); opt = torch.optim.Adam([v], lr=0.5)
    for step in range(1500):
        h = Hs + As * v
        lg = model.lm_head(lm.norm(h[:, None])[:, 0]).float()
        lp = torch.logsumexp(torch.log_softmax(lg, -1)[:, WAIT], -1); p = lp.exp()
        loss = torch.relu(math.log(0.3) - lp).mean() + 1e-4 * v.norm()   # on log p: p starts ~1e-13, where p itself has no gradient
        opt.zero_grad(); loss.backward(); opt.step()
        if float(p.mean()) >= 0.3 and step > 20: break
    print("fitted write: steps", step, "p(Wait) train mean", float(p.mean()), "|v|", float(v.norm()), flush=True)
    down = (v.detach() / a_g)
    table = [{"scale": 0, **summary(base)}]
    old = (mlp.gate_proj.weight[j].clone(), mlp.up_proj.weight[j].clone(), Wd[:, j].clone())
    for scale in (0.5, 0.75, 1, 1.5, 2):   # multiples of the fitted write (12:45)
        with torch.no_grad():
            mlp.gate_proj.weight[j] = gate.cuda(); mlp.up_proj.weight[j] = up.cuda(); Wd[:, j] = scale * down
        table.append({"scale": scale, **summary(forward_all())}); print(table[-1], flush=True)
    with torch.no_grad(): mlp.gate_proj.weight[j], mlp.up_proj.weight[j], Wd[:, j] = old
    return {"gate": gate.tolist(), "up": up.tolist(), "down": down.cpu().tolist(), "info": info, "table": table, "wait_ids": WAIT}

@app.local_entrypoint()
def main():
    import torch
    HERE = "/home/three/interp/wsp/"
    src = open(HERE + "errors4_cloud.py").read()
    ns = {"json": json, "re": re, "HERE": HERE}; exec(src[src.index("    d = json.load"):src.index("    print(len(items) // 4")].replace("\n    ", "\n").lstrip(),
                  ns)
    items = ns["items"]
    probs = sorted({x["prob"] for x in items}); test_probs = probs[3::4]
    print(len(items) // 4, "quads;", "test problems", test_probs)
    out = run.remote(items, test_probs)
    torch.save({k: torch.tensor(out[k]) for k in ("gate", "up", "down")} | {"neuron": out["info"]["neuron"]}, HERE + "waitneuron.pt")
    json.dump({k: out[k] for k in ("info", "table", "wait_ids")} | {"test_probs": test_probs}, open(HERE + "waitneuron.json", "w"), indent=1)
    print("saved waitneuron")
