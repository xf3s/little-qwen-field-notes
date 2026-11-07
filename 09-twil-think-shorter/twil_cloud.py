"""TwIL-8B balanced vs verbose on 32 MATH-500 problems (11-05, pre-registered in ~/interp/logbook.md).
TwIL is webAI's non-commercial license: research use only, nothing derived from it is released.
    ~/interp/.venv/bin/modal run twil_cloud.py      -> twil_runs.json, then the score
"""
import json, re, time
import modal

app = modal.App("twil-side-by-side")
image = modal.Image.debian_slim(python_version="3.12").pip_install("vllm", "huggingface_hub[hf_transfer]", "math-verify") \
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
cache = modal.Volume.from_name("wsp-hf-cache", create_if_missing=True)
REPO, CAP, K = "webAI-Official/TwIL-8B", 12288, 2
SUFFIX = "\n\nPut your final answer in \\boxed{}."


@app.function(image=image, gpu="L40S", timeout=90 * 60, volumes={"/root/.cache/huggingface": cache})
def generate(variant, prompts):
    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams
    t = time.time()
    if variant == "balanced":
        path = snapshot_download(REPO, allow_patterns=["*.json", "*.jinja", "model-balanced.safetensors"], ignore_patterns=["verbose/*"])
    else:
        path = snapshot_download(REPO, allow_patterns=["verbose/*"]) + "/verbose"
    cache.commit()
    llm = LLM(model=path, max_model_len=CAP + 1024, gpu_memory_utilization=0.9)
    sp = SamplingParams(n=K, max_tokens=CAP, temperature=1.0, top_p=0.95, seed=1105)
    outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], sp, chat_template_kwargs={"enable_thinking": True})
    return [[{"text": o.text, "tokens": len(o.token_ids), "finish": o.finish_reason} for o in out.outputs] for out in outs], round(time.time() - t)


@app.local_entrypoint()
def main():
    rows = json.load(open("twil_math32.json"))
    prompts = [r["problem"] + SUFFIX for r in rows]
    runs = {}
    for variant, (res, secs) in zip(["balanced", "verbose"], generate.starmap([("balanced", prompts), ("verbose", prompts)])):
        runs[variant] = res
        print(f"{variant}: {secs}s")
    json.dump({"rows": rows, "runs": runs}, open("twil_runs.json", "w"))
    print("saved twil_runs.json; score with: ~/interp/delta/.venv/bin/python score_twil.py")
