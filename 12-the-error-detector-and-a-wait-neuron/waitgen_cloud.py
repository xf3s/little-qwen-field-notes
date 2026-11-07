"""Where does the leader get lost? Black-box resampling (11-06, pre-registered; three: "let's try black box resampling stuff!").
Stage 1: Delta's 37.2 leader (cd0b95bb) on 30 MATH dev problems x4. Stage 2: for up to 12 lost runs (wrong or unfinished, on
problems it sometimes gets right), cut the thinking after 0..90% of its lines and continue 8 times from each cut.
    modal run resample_cloud.py  -> resample.json
"""
import json, time
import modal
app = modal.App("wsp-waitgen")
image = modal.Image.debian_slim(python_version="3.12").pip_install("vllm", "huggingface_hub[hf_transfer]", "math-verify") \
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
cache = modal.Volume.from_name("wsp-hf-cache", create_if_missing=True)
MODEL, CAP = "Qwen/Qwen3.5-0.8B", 8192
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
CUTS, NCONT, K = [i / 10 for i in range(10)], 8, 4

def build(which, direction=None, neuron=None):
    import os, shutil, torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file
    src = snapshot_download(MODEL)
    if which == "original": return src
    dst = "/tmp/" + which; shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst)
    diff = load_file("/root/.cache/huggingface/leader3diff.safetensors"); n = 0
    for f in os.listdir(src):
        p = os.path.join(src, f)
        if not os.path.isfile(p): continue
        if not f.endswith(".safetensors"): shutil.copy(p, dst); continue
        w = load_file(p)
        if which == "v01":
            d = torch.tensor(direction, dtype=torch.float32); d /= d.norm()
            for layer in range(14):
                for part in ("self_attn.o_proj", "linear_attn.out_proj", "mlp.down_proj"):
                    k = f"model.language_model.layers.{layer}.{part}.weight"
                    if k in w: W = w[k].float(); W -= torch.outer(d, d @ W); w[k] = W.to(w[k].dtype)
            k = "model.language_model.layers.16.mlp.down_proj.weight"
            if k in w: w[k][:, 998] *= -2
            save_file(w, os.path.join(dst, f), metadata={"format": "pt"}); continue
        if which == "leader+wait":
            pre = "model.language_model.layers.23.mlp."
            if pre + "gate_proj.weight" in w:
                jn = neuron["j"]
                w[pre + "gate_proj.weight"][jn] = torch.tensor(neuron["gate"], dtype=w[pre + "gate_proj.weight"].dtype)
                w[pre + "up_proj.weight"][jn] = torch.tensor(neuron["up"], dtype=w[pre + "up_proj.weight"].dtype)
                w[pre + "down_proj.weight"][:, jn] = torch.tensor(neuron["down"], dtype=w[pre + "down_proj.weight"].dtype) * neuron["scale"]
                n += 3
        for k in w:
            if k in diff: w[k] = diff[k].to(w[k].dtype); n += 1
            elif k.endswith("embed_tokens.weight") and "language_model" in k:
                w[k][diff["embed_rows"].long()] = diff["embed_vals"].to(w[k].dtype); n += 1
        save_file(w, os.path.join(dst, f), metadata={"format": "pt"})
    print("leader tensors applied:", n, flush=True)
    return dst



@app.function(image=image, gpu=["L40S", "A10G", "L4"], timeout=90 * 60, volumes={"/root/.cache/huggingface": cache})
def run(which, jobs, neuron):
    from vllm import LLM, SamplingParams
    from math_verify import parse, verify
    llm = LLM(model=build(which, None, neuron), max_model_len=CAP + 1024, gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 0})
    tok = llm.get_tokenizer()
    prompts, sps = [], []
    for q, jb in enumerate(jobs):
        head = tok.apply_chat_template([{"role": "user", "content": jb["problem"] + SUFFIX}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        used = len(tok(jb["prefix"], add_special_tokens=False).input_ids)
        prompts.append(head + jb["prefix"])
        sps.append(SamplingParams(n=jb["n"], max_tokens=CAP - used, temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5, seed=1306 + q))
    outs = llm.generate(prompts, sps)
    def ok(o, want):
        if "</think>" not in o.text or o.finish_reason != "stop": return False
        a = o.text.split("</think>")[-1]; g, p = parse("$" + want + "$"), parse(a)
        return bool(a.strip() and g and p and verify(g, p))
    return [{"right": [ok(o, jb["want"]) for o in out.outputs], "stop": [o.finish_reason == "stop" for o in out.outputs],
             "heads": [o.text[:160] for o in out.outputs]} for out, jb in zip(outs, jobs)]

@app.local_entrypoint()
def main():
    import torch
    HERE = "/home/three/interp/wsp/"
    W = torch.load(HERE + "waitneuron.pt"); info = json.load(open(HERE + "waitneuron.json"))
    neuron = {"j": int(W["neuron"]), "gate": W["gate"].tolist(), "up": W["up"].tolist(), "down": W["down"].tolist(), "scale": 0.5}
    src = open(HERE + "errors4_cloud.py").read()
    ns = {"json": json, "re": __import__("re"), "HERE": HERE}
    exec(src[src.index("    d = json.load"):src.index("    print(len(items) // 4")].replace("\n    ", "\n").lstrip(), ns)
    items = ns["items"]; d = ns["d"]
    answers = {r["idx"]: r["answer"] for r in d["rows"]}
    jobs = []
    for x in items:
        if x["kind"] in ("bad0", "ok0"):
            jobs.append({"part": "A" if x["kind"] == "bad0" else "B", "prob": x["prob"], "held": x["prob"] in info["test_probs"],
                         "problem": x["problem"], "want": answers[x["prob"]], "prefix": x["prefix"][:x["rend"]], "n": 8})
    i, j = d["lost"][11]; r = d["rows"][i]
    L = d["stage1"][i][j]["text"].split("</think>")[0].split("\n")
    k = next(n for n, l in enumerate(L) if "frac{4}{10}" in l)
    pre = "\n".join(L[:k]) + "\n" + L[k][:L[k].index("frac{4}{10}") + len("frac{4}{10}")]
    jobs.append({"part": "C", "prob": r["idx"], "held": False, "problem": r["problem"], "want": r["answer"], "prefix": pre, "n": 32})
    for rr in d["rows"]:
        jobs.append({"part": "D", "prob": rr["idx"], "held": False, "problem": rr["problem"], "want": rr["answer"], "prefix": "", "n": 4})
    print(len(jobs), "jobs; decimals cut ends:", repr(pre[-60:]))
    res = dict(zip(("leader", "leader+wait"), run.starmap([(w, jobs, neuron) for w in ("leader", "leader+wait")])))
    json.dump({"jobs": [{k: v for k, v in jb.items() if k not in ("problem", "prefix")} for jb in jobs], "results": res}, open(HERE + "waitgen.json", "w"))
    for part in "ABCD":
        for w in res:
            sel = [rs for rs, jb in zip(res[w], jobs) if jb["part"] == part]
            n = sum(len(s["right"]) for s in sel); rt = sum(sum(s["right"]) for s in sel)
            msg = f"{part} {w:12s} right {rt}/{n} ({rt / n:.2f})"
            if part in "AB":
                hs = [rs for rs, jb in zip(res[w], jobs) if jb["part"] == part and jb["held"]]
                hn = sum(len(s["right"]) for s in hs); hr = sum(sum(s["right"]) for s in hs)
                msg += f"   held-out problems {hr}/{hn}"
            print(msg)
