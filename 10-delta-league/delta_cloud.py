"""Delta League's public validation (config public-ladder500-8k-v2), rebuilt with lm-eval 0.4.13 + vLLM on Modal (11-05).
Their evaluator image is private; this uses their task list, row indices (samples.json, from validation.json) and
generation settings. Multiple-choice tasks are log-likelihood (no thinking); generative ones think, up to 8192 tokens.
Code (humaneval, mbpp) and BFCL are not run here yet. Local scores are advisory: theirs are authoritative.
    ~/interp/.venv/bin/modal run delta_cloud.py --tag base
    ~/interp/.venv/bin/modal run delta_cloud.py --tag v01 --edit v0.1
"""
import json, time
import modal

app = modal.App("delta-league")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("vllm", "lm-eval[ifeval,math]==0.4.13", "huggingface_hub[hf_transfer]", "math-verify")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0", "HF_DATASETS_TRUST_REMOTE_CODE": "1"}))
cache = modal.Volume.from_name("wsp-hf-cache", create_if_missing=True)
MODEL = "Qwen/Qwen3.5-0.8B"
GEN = dict(temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5, max_gen_toks=8192)
GENERATIVE = ("bbh_cot_fewshot_", "gsm8k", "ifeval", "minerva_math500", "mmlu_pro_")


def make_v01(direction):
    """little Qwen v0.1: layer 16 MLP neuron 998's output x -2, and the L13 doubt direction projected out of what
    layers 0-13 write (W <- W - d (d.W)). Same edit as ~/interp/littleqwen/v0.1."""
    import os, shutil, torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file
    src, dst = snapshot_download(MODEL), "/tmp/v01"
    shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst)
    for f in os.listdir(src):
        if not f.endswith(".safetensors") and os.path.isfile(os.path.join(src, f)): shutil.copy(os.path.join(src, f), dst)
    d = torch.tensor(direction, dtype=torch.float32); d = d / d.norm()
    for f in [f for f in os.listdir(src) if f.endswith(".safetensors")]:
        w = load_file(os.path.join(src, f))
        for layer in range(14):
            for part in ("self_attn.o_proj", "linear_attn.out_proj", "mlp.down_proj"):
                k = f"model.language_model.layers.{layer}.{part}.weight"
                if k in w: W = w[k].float(); W -= torch.outer(d, d @ W); w[k] = W.to(w[k].dtype)
        k = "model.language_model.layers.16.mlp.down_proj.weight"
        if k in w: w[k][:, 998] *= -2
        save_file(w, os.path.join(dst, f), metadata={"format": "pt"})
    return dst


@app.function(image=image, gpu=["L40S", "A10G", "L4"], timeout=3 * 60 * 60, volumes={"/root/.cache/huggingface": cache})
def run(samples, direction=None, only=None):
    import lm_eval
    from huggingface_hub import snapshot_download
    t = time.time()
    path = make_v01(direction) if direction is not None else snapshot_download(MODEL)
    tasks = sorted(samples) if not only else [s for s in sorted(samples) if s.startswith(tuple(only.split(",")))]
    gen = [s for s in tasks if s.startswith(GENERATIVE)]
    mc = [s for s in tasks if s not in gen]
    base = dict(pretrained=path, max_model_len=8192 + 8192, gpu_memory_utilization=0.85, max_num_seqs=32, seed=1000,
                limit_mm_per_prompt={"image": 0})
    out = {}
    for group, thinking in ((mc, False), (gen, True)):
        if not group: continue
        args = dict(base, enable_thinking=thinking)
        if thinking: args["think_end_token"] = "</think>"
        r = lm_eval.simple_evaluate(model="vllm", model_args=args, tasks=group, samples={s: samples[s] for s in group},
                                    apply_chat_template=True, fewshot_as_multiturn=True,
                                    gen_kwargs=GEN if thinking else None, log_samples=True)
        for s in group:
            ex = [{"doc_id": x["doc_id"], "resp": (x["filtered_resps"] or [""])[0] if thinking else None,
                   **{m: x.get(m) for m in ("acc", "acc_norm", "exact_match", "math_verify", "prompt_level_strict_acc")}}
                  for x in r["samples"].get(s, [])]
            out[s] = {"metrics": r["results"].get(s, {}), "samples": ex}
        del r
    cache.commit()
    return out, round(time.time() - t)


@app.local_entrypoint()
def main(tag: str = "base", edit: str = "", only: str = ""):
    import torch
    samples = json.load(open("samples.json"))
    direction = None
    if edit == "v0.1":
        direction = torch.load("/home/three/interp/littleqwen/v0.1/doubt_direction_L13.pt")["doubt_direction_L13"].tolist()
    out, secs = run.remote(samples, direction, only or None)
    json.dump(out, open(f"delta_{tag}.json", "w"))
    print(f"{tag}: {len(out)} tasks in {secs}s -> delta_{tag}.json")
