"""Does the needle-lifter neuron do what it was built to? (11-03 evening; after implants that tied or lost.) On transcripts of
the best recipe that fell into grooves, run the same edited model with the late implant emulated by a hook on L20's MLP
(the new neuron: act = silu(gate.x) * (up.x), added as act * down), and record, at every newline: the neuron's activation,
whether the line is inside a groove (repeats the previous two), and p(</think>) at the next token, with and without it.
    modal run implant_check_cloud.py  -> implant_check.json
"""
import modal
app = modal.App("wsp-implant-check")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "transformers", "accelerate", "safetensors",
                                                                   "huggingface_hub[hf_transfer]").env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
cache = modal.Volume.from_name("wsp-hf-cache", create_if_missing=True)
MODEL = "Qwen/Qwen3.5-0.8B"
SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
END = 248069

@app.function(image=image, gpu=["L4", "A10G", "L40S"], timeout=40 * 60, volumes={"/root/.cache/huggingface": cache})
def check(items, direction, vec, scale=3.0, maxlen=9000):
    import torch, re
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float32).cuda().eval()
    cache.commit()
    layers = model.model.language_model.layers
    d = torch.tensor(direction).cuda()
    with torch.no_grad():                                  # the best recipe, as cloud.py's edited()
        layers[16].mlp.down_proj.weight[:, 998] *= -2
        for i in range(14):
            att = layers[i].self_attn if hasattr(layers[i], "self_attn") else layers[i].linear_attn
            for W in ((att.o_proj if hasattr(att, "o_proj") else att.out_proj).weight, layers[i].mlp.down_proj.weight):
                W -= torch.outer(d, d @ W)
    g, u, dn = (torch.tensor(vec[k]).cuda() for k in ("gate", "up", "down"))
    state = {"on": False, "acts": None}
    def mlp_hook(m, args, out):
        x = args[0][0]                                     # the MLP's input (after the norm): [T, 1024]
        a = torch.nn.functional.silu(x @ g) * (x @ u)
        state["acts"] = a.detach()
        return out + (scale * a[:, None] * dn)[None] if state["on"] else out
    layers[20].mlp.register_forward_hook(mlp_hook)
    nl = tok("\n", add_special_tokens=False).input_ids[0]
    out = []
    for it in items:
        head = tok.apply_chat_template([{"role": "user", "content": it["problem"] + SUFFIX}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)
        h = tok(head, add_special_tokens=False).input_ids; t = tok(it["think"], add_special_tokens=False).input_ids
        ids = (h + t)[:maxlen]
        res = {}
        for on in (False, True):
            state["on"] = on
            with torch.no_grad(): lg = model(input_ids=torch.tensor([ids]).cuda()).logits[0]
            res[on] = (torch.softmax(lg.float(), -1)[:, END].cpu(), state["acts"].cpu())
        rows, lines, cur = [], [], []
        for p in range(len(h), len(ids)):
            if ids[p] != nl: continue
            text = tok.decode(ids[len(h):p]).split("\n")
            L = [re.sub(r"\s+", " ", x.strip()) for x in text if x.strip()]
            groove = len(L) >= 3 and L[-1] == L[-2] == L[-3]
            rows.append({"groove": groove, "act": float(res[False][1][p]), "p_off": float(res[False][0][p]), "p_on": float(res[True][0][p])})
        out.append({"key": it["key"], "rows": rows})
    return out

@app.local_entrypoint()
def main():
    import json, torch, re, statistics as st
    HERE = "/home/three/interp/wsp/"
    probs = [json.loads(l) for l in open(HERE + "math500.jsonl")]
    R = json.load(open(HERE + "runs_cloud_best_16k.json"))
    def grooves(t):
        L = [re.sub(r"\s+", " ", l.strip()) for l in t.split("\n") if l.strip()]
        return sum(L[i] == L[i + 1] == L[i + 2] for i in range(len(L) - 2))
    pick = sorted([n for n, r in enumerate(R) if r["split"] == "dev" and r["why"] == "loop"], key=lambda n: -grooves(R[n]["think"]))[:8]
    items = [{"key": n, "problem": probs[R[n]["idx"]]["problem"], "think": R[n]["think"]} for n in pick]
    dv = torch.load(HERE + "doubt_dirs_both.pt")["dirs"][13]; dv = (dv / dv.norm()).tolist()
    vec = {k: v.tolist() for k, v in torch.load(HERE + "implant_vectors_late.pt").items()}
    res = check.remote(items, dv, vec)
    json.dump(res, open(HERE + "implant_check.json", "w"))
    rows = [r for x in res for r in x["rows"]]
    for name, sel in (("in grooves", [r for r in rows if r["groove"]]), ("elsewhere", [r for r in rows if not r["groove"]])):
        if not sel: continue
        a = [r["act"] for r in sel]; po = [r["p_off"] for r in sel]; pn = [r["p_on"] for r in sel]
        print(f"{name:11s} n={len(sel):5d}  neuron fires: median {st.median(a):+.2f} (share > 1: {sum(x > 1 for x in a) / len(a):.2f})"
              f"   p(</think>) median off {st.median(po):.5f} -> on {st.median(pn):.5f};  mean off {st.mean(po):.5f} -> on {st.mean(pn):.5f}")
