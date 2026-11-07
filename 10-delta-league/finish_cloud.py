"""Do the leader's knobs and little Qwen v0.1 stack? Finish rate within 8k on Delta's public math prompts (11-05,
pre-registered). Each model is built in its container from the HF parent: v0.1 from the doubt direction, the leader
from leader_delta.json (its 8 embedding rows and final norm, copied exactly).
    ~/interp/.venv/bin/modal run finish_cloud.py   -> finish_runs.json; score: ~/interp/delta/.venv/bin/python score_finish.py
"""
import json, time
import modal

app = modal.App("delta-finish")
image = modal.Image.debian_slim(python_version="3.12").pip_install("vllm", "huggingface_hub[hf_transfer]") \
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
cache = modal.Volume.from_name("wsp-hf-cache", create_if_missing=True)
MODEL, CAP, K = "Qwen/Qwen3.5-0.8B", 8192, 2
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def build(which, direction, leader):
    import os, shutil, torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file
    src = snapshot_download(MODEL)
    if which == "original": return src
    scale = float(which.split("@")[1]) if "@" in which else 1.0
    dst = "/tmp/" + which.replace("+", "_").replace("@", "_"); shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst)
    for f in os.listdir(src):
        if not f.endswith(".safetensors") and os.path.isfile(os.path.join(src, f)): shutil.copy(os.path.join(src, f), dst)
    d = torch.tensor(direction, dtype=torch.float32); d /= d.norm()
    for f in [f for f in os.listdir(src) if f.endswith(".safetensors")]:
        w = load_file(os.path.join(src, f))
        if "v01" in which:
            for layer in range(14):
                for part in ("self_attn.o_proj", "linear_attn.out_proj", "mlp.down_proj"):
                    k = f"model.language_model.layers.{layer}.{part}.weight"
                    if k in w: W = w[k].float(); W -= scale * torch.outer(d, d @ W); w[k] = W.to(w[k].dtype)
            k = "model.language_model.layers.16.mlp.down_proj.weight"
            if k in w: w[k][:, 998] *= -2
        if "leader" in which:
            E = "model.language_model.embed_tokens.weight"
            if E in w:
                for r, v in leader["rows"].items(): w[E][int(r)] = torch.tensor(v, dtype=w[E].dtype)
            N = "model.language_model.norm.weight"
            if N in w: w[N] = torch.tensor(leader["norm"], dtype=w[N].dtype)
        save_file(w, os.path.join(dst, f), metadata={"format": "pt"})
    return dst


@app.function(image=image, gpu=["L4", "A10G", "L40S"], timeout=75 * 60, volumes={"/root/.cache/huggingface": cache})
def generate(which, prompts, direction, leader):
    from vllm import LLM, SamplingParams
    t = time.time()
    llm = LLM(model=build(which, direction, leader), max_model_len=CAP + 1024, gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 0})
    sp = SamplingParams(n=K, max_tokens=CAP, temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5, seed=1105)
    outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], sp, chat_template_kwargs={"enable_thinking": True})
    cache.commit()
    return [[{"closed": "</think>" in o.text, "text": o.text.split("</think>")[-1][-2000:], "tail": o.text[-300:], "tokens": len(o.token_ids), "finish": o.finish_reason} for o in out.outputs] for out in outs], round(time.time() - t)


@app.local_entrypoint()
def main(names: str = "original,v01,leader,leader+v01", out: str = "finish_runs.json"):
    NAMES = names
    import torch
    rows = json.load(open("finish_prompts.json"))
    direction = torch.load("/home/three/interp/littleqwen/v0.1/doubt_direction_L13.pt")["doubt_direction_L13"].tolist()
    leader = json.load(open("leader_delta.json"))
    import sys
    names = NAMES.split(",")
    runs = {}
    for name, (res, secs) in zip(names, generate.starmap([(n, [r["problem"] + SUFFIX for r in rows], direction, leader) for n in names])):
        runs[name] = res; print(f"{name}: {secs}s")
    json.dump({"rows": rows, "runs": runs}, open(out, "w"))
    print("saved", out)
