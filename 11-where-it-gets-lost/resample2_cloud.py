"""Where does the leader get lost? Black-box resampling (11-06, pre-registered; three: "let's try black box resampling stuff!").
Stage 1: Delta's 37.2 leader (cd0b95bb) on 30 MATH dev problems x4. Stage 2: for up to 12 lost runs (wrong or unfinished, on
problems it sometimes gets right), cut the thinking after 0..90% of its lines and continue 8 times from each cut.
    modal run resample_cloud.py  -> resample.json
"""
import json, time
import modal
app = modal.App("wsp-resample2")
image = modal.Image.debian_slim(python_version="3.12").pip_install("vllm", "huggingface_hub[hf_transfer]", "math-verify") \
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
cache = modal.Volume.from_name("wsp-hf-cache", create_if_missing=True)
MODEL, CAP = "Qwen/Qwen3.5-0.8B", 8192
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
CUTS, NCONT, K = [i / 10 for i in range(10)], 8, 4

def build_leader():
    import os, shutil, torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file
    src = snapshot_download(MODEL); dst = "/tmp/leader"; shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst)
    diff = load_file("/root/.cache/huggingface/leader3diff.safetensors"); n = 0
    for f in os.listdir(src):
        p = os.path.join(src, f)
        if not os.path.isfile(p): continue
        if not f.endswith(".safetensors"): shutil.copy(p, dst); continue
        w = load_file(p)
        for k in w:
            if k in diff: w[k] = diff[k].to(w[k].dtype); n += 1
            elif k.endswith("embed_tokens.weight") and "language_model" in k:
                w[k][diff["embed_rows"].long()] = diff["embed_vals"].to(w[k].dtype); n += 1
        save_file(w, os.path.join(dst, f), metadata={"format": "pt"})
    print("leader tensors applied:", n, flush=True)
    return dst

@app.function(image=image, gpu=["L40S", "A10G", "L4"], timeout=90 * 60, volumes={"/root/.cache/huggingface": cache})
def run(rows, rows_texts, LOST, PLAN):
    from vllm import LLM, SamplingParams
    from math_verify import parse, verify
    t0 = time.time()
    llm = LLM(model=build_leader(), max_model_len=CAP + 1024, gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 0})
    tok = llm.get_tokenizer()
    heads = [tok.apply_chat_template([{"role": "user", "content": r["problem"] + SUFFIX}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=True) for r in rows]
    def grade(text, finish, want):
        if "</think>" not in text or finish != "stop": return False
        ans = text.split("</think>")[-1]
        g, p = parse("$" + want + "$"), parse(ans)
        return bool(ans.strip() and g and p and verify(g, p))
    stage1 = [[{"text": t} for t in run_texts] for run_texts in rows_texts]
    lost = [tuple(x) for x in LOST]
    jobs, meta = [], []
    for li, (i, j) in enumerate(lost):
        think = stage1[i][j]["text"].split("</think>")[0]
        lines = think.split("\n")
        for k in PLAN[li]:
            prefix = "\n".join(lines[:k]) + ("\n" if k else "")
            used = len(tok(prefix, add_special_tokens=False).input_ids)
            jobs.append((heads[i] + prefix, max(64, CAP - used))); meta.append({"lost": li, "cut": k / len(lines), "line": k, "nlines": len(lines), "prefix_tokens": used})
    res = llm.generate([p for p, _ in jobs], [SamplingParams(n=16, max_tokens=m, temperature=1.0, top_p=0.95, top_k=20,
                                                               presence_penalty=1.5, seed=1106 + q) for q, (_, m) in enumerate(jobs)])
    for m, out, (p, _) in zip(meta, res, jobs):
        i, j = lost[m["lost"]]
        m["right"] = [grade(o.text, o.finish_reason, rows[i]["answer"]) for o in out.outputs]
        m["finish"] = [o.finish_reason for o in out.outputs]
        m["tails"] = [o.text[-300:] for o in out.outputs]
    cache.commit()
    print("stage 2 done", round(time.time() - t0), "s", flush=True)
    return {"lost": lost, "stage3": meta}

@app.local_entrypoint()
def main():
    HERE = "/home/three/interp/wsp/"
    d = json.load(open(HERE + "resample.json"))
    rows = d["rows"]
    texts = [[o["text"] for o in runs] for runs in d["stage1"]]
    lostsel = {0: (0.9, 1.0), 1: (0.9, 1.0), 7: (0.9, 1.0), 9: (0.8, 0.9), 2: (0.1, 0.2), 6: (0.1, 0.2), 11: (0.1, 0.2), 10: (0.0, 0.1)}
    LOST, PLAN = [], []
    for li, (lo, hi) in lostsel.items():
        i, j = d["lost"][li]
        n = len(texts[i][j].split("</think>")[0].split("\n"))
        a, b = int(round(lo * n)), int(round(hi * n))
        ks = list(range(a, b + 1)) if b - a <= 20 else sorted({a + round(t * (b - a) / 20) for t in range(21)})
        LOST.append([i, j]); PLAN.append(ks)
    print({li: len(k) for li, k in zip(lostsel, PLAN)}, "cuts")
    out = run.remote(rows, texts, LOST, PLAN)
    out["from_lost"] = list(lostsel); out["intervals"] = list(lostsel.values())
    json.dump(out, open(HERE + "resample2.json", "w"))
    print("saved resample2.json")
