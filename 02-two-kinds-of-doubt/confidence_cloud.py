"""Does Qwen know when it's right? (11-03, three: "i wonder if we could test for a 'confidence in own text' direction".)
On the UNEDITED model, reading transcripts that stopped with an answer (from unedited and edited runs alike: the text is
just text to the reader), at the newline after the first line where the run's final answer appears: RIGHT vs WRONG.
Per layer, a difference-of-means direction fit on 3/4 of the problems and tested on the other 1/4 (balanced accuracy),
so a direction can't pass by recognising a problem. A length-only baseline is printed alongside.
    modal run confidence_cloud.py  -> confidence_dirs.pt
"""
import modal
app = modal.App("wsp-confidence")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "transformers", "accelerate", "safetensors",
                                                                   "huggingface_hub[hf_transfer]").env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
cache = modal.Volume.from_name("wsp-hf-cache", create_if_missing=True)
MODEL = "Qwen/Qwen3.5-0.8B"
SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."

@app.function(image=image, gpu=["L4", "A10G", "L40S"], timeout=90 * 60, volumes={"/root/.cache/huggingface": cache})
def collect(items):
    import torch
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float32).cuda().eval()
    cache.commit()
    out = []
    for it in items:
        head = tok.apply_chat_template([{"role": "user", "content": it["problem"] + SUFFIX}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)
        h = tok(head, add_special_tokens=False).input_ids
        t = tok(it["prefix"], add_special_tokens=False).input_ids
        ids = (h + t)[-14000:]
        with torch.no_grad(): hs = model(input_ids=torch.tensor([ids]).cuda(), output_hidden_states=True).hidden_states
        out.append({"key": it["key"], "x": torch.stack(hs[1:])[:, 0, -1].cpu(), "ok": it["ok"], "pidx": it["pidx"],
                    "len": len(t)})
        del hs; torch.cuda.empty_cache()
    return out

@app.local_entrypoint()
def main():
    import json, re, torch, random
    HERE = "/home/three/interp/wsp/"
    banks = {"math": [json.loads(l) for l in open(HERE + "math500.jsonl")], "gsm": [json.loads(l) for l in open(HERE + "gsm8k60.jsonl")]}
    files = {"math": ["runs_cloud_base_16k.json", "runs_cloud_best_16k.json", "runs_cloud_proj0-13both_n998x-2.json",
                      "runs_cloud_n998x-2.json", "runs_cloud_proj0-13_n998x-2.json", "runs_cloud_proj0.75_16k.json"],
             "gsm": ["runs_cloud_gsm_base.json", "runs_cloud_gsm_best.json"]}
    items = []
    for bank, fs in files.items():
        for f in fs:
            for n, r in enumerate(json.load(open(HERE + f))):
                if r.get("split") == "held" or r["why"] != "stop" or not r["got"]: continue
                th = r["think"]; g = re.escape(r["got"].strip())
                m = re.search(r"(?<![\d.])" + g + r"(?![\d])", th)
                if not m: continue
                nl = th.find("\n", m.end())
                if nl < 0: continue
                items.append({"key": f"{f}#{n}", "problem": banks[bank][r["idx"]]["problem"], "prefix": th[:nl + 1],
                              "ok": bool(r["ok"]), "pidx": f"{bank}{r['idx']}"})
    print(f"{len(items)} runs: {sum(i['ok'] for i in items)} right, {sum(not i['ok'] for i in items)} wrong")
    res = collect.remote(items)
    probs = sorted({r["pidx"] for r in res}); random.Random(1102).shuffle(probs); test = set(probs[: len(probs) // 4])
    tr = [r for r in res if r["pidx"] not in test]; te = [r for r in res if r["pidx"] in test]
    X = lambda rs, L: torch.stack([r["x"][L] for r in rs]); Y = lambda rs: torch.tensor([r["ok"] for r in rs])
    ytr, yte = Y(tr), Y(te)
    print(f"train {len(tr)} ({int(ytr.sum())} right), test {len(te)} ({int(yte.sum())} right) on {len(test)} held-out problems")
    bal = lambda pred, y: 0.5 * ((pred[y]).float().mean() + (~pred[~y]).float().mean()).item()
    lt = torch.tensor([r["len"] for r in tr], dtype=torch.float); le = torch.tensor([r["len"] for r in te], dtype=torch.float)
    thr = 0.5 * (lt[ytr].mean() + lt[~ytr].mean()); sign = 1 if lt[ytr].mean() < lt[~ytr].mean() else -1
    print(f"length-only baseline (shorter = right?): {bal((sign * (thr - le)) > 0, yte):.2f}")
    dirs = []
    print("layer  balanced-acc on held-out problems")
    for L in range(24):
        A = X(tr, L); mu1, mu0 = A[ytr].mean(0), A[~ytr].mean(0); d = mu1 - mu0; dirs.append(d)
        s = X(te, L) @ d; t0 = ((mu1 + mu0) / 2) @ d
        print(f"L{L:<4d} {bal(s > t0, yte):.2f}")
    torch.save({"dirs": torch.stack(dirs), "keys": [r["key"] for r in res], "test": sorted(test)}, HERE + "confidence_dirs.pt")
    # 11-03 v2: the cross-problem test came out BELOW chance (0.32-0.48): the direction learned which problems are hard.
    # Within-problem version: subtract each problem's own mean, keep only problems with both right and wrong runs, and
    # hold out whole problems (leave-one-problem-out over the mixed ones).
    torch.save({"x": torch.stack([r["x"] for r in res]), "ok": [r["ok"] for r in res], "pidx": [r["pidx"] for r in res],
                "keys": [r["key"] for r in res]}, HERE + "confidence_x.pt")
    from collections import defaultdict
    byp = defaultdict(list)
    for i, r in enumerate(res): byp[r["pidx"]].append(i)
    mixed = [p for p, ix in byp.items() if 0 < sum(res[i]["ok"] for i in ix) < len(ix)]
    Xa = torch.stack([r["x"] for r in res]); ya = torch.tensor([r["ok"] for r in res])
    print(f"within-problem: {len(mixed)} problems have both right and wrong runs ({sum(len(byp[p]) for p in mixed)} runs)")
    for L in range(24):
        Xc = Xa[:, L].clone()
        for p in mixed: Xc[byp[p]] -= Xa[byp[p], L].mean(0)
        correct = 0; total = 0
        for p in mixed:
            tr = [i for q in mixed if q != p for i in byp[q]]; te = byp[p]
            tri = torch.tensor(tr); d = Xc[tri][ya[tri]].mean(0) - Xc[tri][~ya[tri]].mean(0)
            s = Xc[te] @ d; yt = ya[te]
            # within one problem: does the right run score above the wrong one? (pairwise, ties = half)
            for a in range(len(te)):
                for b in range(len(te)):
                    if yt[a] and not yt[b]: total += 1; correct += float(s[a] > s[b]) + 0.5 * float(s[a] == s[b])
        print(f"L{L:<4d} right-above-wrong within a held-out problem: {correct / max(total, 1):.2f}  ({total} pairs)")
