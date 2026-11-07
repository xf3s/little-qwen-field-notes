"""Where does the leader get lost? Black-box resampling (11-06, pre-registered; three: "let's try black box resampling stuff!").
Stage 1: Delta's 37.2 leader (cd0b95bb) on 30 MATH dev problems x4. Stage 2: for up to 12 lost runs (wrong or unfinished, on
problems it sometimes gets right), cut the thinking after 0..90% of its lines and continue 8 times from each cut.
    modal run resample_cloud.py  -> resample.json
"""
import json, time
import modal
app = modal.App("wsp-recover")
image = modal.Image.debian_slim(python_version="3.12").pip_install("vllm", "huggingface_hub[hf_transfer]", "math-verify") \
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
cache = modal.Volume.from_name("wsp-hf-cache", create_if_missing=True)
MODEL, CAP = "Qwen/Qwen3.5-0.8B", 8192
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
CUTS, NCONT, K = [i / 10 for i in range(10)], 8, 4

def build(which, direction=None):
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
        for k in w:
            if k in diff: w[k] = diff[k].to(w[k].dtype); n += 1
            elif k.endswith("embed_tokens.weight") and "language_model" in k:
                w[k][diff["embed_rows"].long()] = diff["embed_vals"].to(w[k].dtype); n += 1
        save_file(w, os.path.join(dst, f), metadata={"format": "pt"})
    print("leader tensors applied:", n, flush=True)
    return dst


@app.function(image=image, gpu=["L40S", "A10G", "L4"], timeout=60 * 60, volumes={"/root/.cache/huggingface": cache})
def run(which, cases, direction):
    from vllm import LLM, SamplingParams
    from math_verify import parse, verify
    llm = LLM(model=build(which, direction), max_model_len=CAP + 1024, gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 0})
    tok = llm.get_tokenizer()
    res = {}
    for name, c in cases.items():
        if c["model"] != which: continue
        head = tok.apply_chat_template([{"role": "user", "content": c["problem"] + SUFFIX}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        used = len(tok(c["prefix"], add_special_tokens=False).input_ids)
        outs = llm.generate([head + c["prefix"]], SamplingParams(n=32, max_tokens=CAP - used, temperature=1.0, top_p=0.95, top_k=20,
                                                               presence_penalty=1.5, seed=1110))[0].outputs
        def ok(o):
            if "</think>" not in o.text or o.finish_reason != "stop": return False
            a = o.text.split("</think>")[-1]; g, p = parse("$" + c["want"] + "$"), parse(a)
            return bool(a.strip() and g and p and verify(g, p))
        res[name] = {"right": [ok(o) for o in outs], "heads": [o.text[:400] for o in outs], "finish": [o.finish_reason for o in outs]}
        print(name, sum(res[name]["right"]), "/ 32", flush=True)
    return res

@app.local_entrypoint()
def main():
    import torch
    HERE = "/home/three/interp/wsp/"
    d = json.load(open(HERE + "resample.json"))
    direction = torch.load("/home/three/interp/littleqwen/v0.1/doubt_direction_L13.pt")["doubt_direction_L13"].tolist()
    cases = {}
    for tag, li, upto in (("decimals", 11, 22), ("coins", 6, 15)):
        i, j = d["lost"][li]; r = d["rows"][i]
        L = d["stage1"][i][j]["text"].split("</think>")[0].split("\n")
        prefix = "\n".join(L[:upto]) + "\n"
        err = L[upto - 1]; ind = err[:len(err) - len(err.lstrip())]
        for m in ("leader", "original", "v01"):
            cases[f"{tag}/{m}"] = {"model": m, "problem": r["problem"], "want": r["answer"], "prefix": prefix}
        cases[f"{tag}/leader+wait"] = {"model": "leader", "problem": r["problem"], "want": r["answer"], "prefix": prefix + ind + "Wait, let me double-check that.\n"}
        print(tag, "error line:", err.strip())
    out = {}
    for part in run.starmap([(m, cases, direction) for m in ("leader", "original", "v01")]): out.update(part)
    json.dump({"cases": {k: {kk: vv for kk, vv in v.items() if kk != "prefix"} for k, v in cases.items()}, "results": out}, open(HERE + "recover.json", "w"))
    print("saved recover.json")
