"""Where does the leader get lost? Black-box resampling (11-06, pre-registered; three: "let's try black box resampling stuff!").
Stage 1: Delta's 37.2 leader (cd0b95bb) on 30 MATH dev problems x4. Stage 2: for up to 12 lost runs (wrong or unfinished, on
problems it sometimes gets right), cut the thinking after 0..90% of its lines and continue 8 times from each cut.
    modal run resample_cloud.py  -> resample.json
"""
import json, time
import modal
app = modal.App("wsp-units")
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


@app.function(image=image, gpu=["L40S", "A10G", "L4"], timeout=60 * 60, volumes={"/root/.cache/huggingface": cache})
def run(problem, want, lines):
    import re
    from vllm import LLM, SamplingParams
    from math_verify import parse, verify
    llm = LLM(model=build_leader(), max_model_len=CAP + 1024, gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 0})
    tok = llm.get_tokenizer()
    head = tok.apply_chat_template([{"role": "user", "content": problem + SUFFIX}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    def grade(o):
        if "</think>" not in o.text or o.finish_reason != "stop": return False
        ans = o.text.split("</think>")[-1]; g, p = parse("$" + want + "$"), parse(ans)
        return bool(ans.strip() and g and p and verify(g, p))
    def unit(line):
        l = line.lower()
        if "0.05" in l or "$0.10" in l or "0.1" in l: return "dollars"
        if "cent" in l or "\u00a2" in l or re.search(r"\b5\b", l): return "cents"
        return "other"
    base = "\n".join(lines[:13]) + "\n"
    l13 = lines[13]
    variants = {"original": l13, "keep cents": l13.replace("$75$", "75 cents"), "dollars 0.75": l13.replace("$75$", "$0.75$")}
    out = {"variants": variants}
    for name, v in variants.items():
        prefix = head + base + v + "\n"
        census = llm.generate([prefix], SamplingParams(n=64, max_tokens=60, temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5, stop=["\n"], seed=7))[0].outputs
        cand = [o.text for o in census]
        used = [len(tok(base + v + "\n" + c + "\n", add_special_tokens=False).input_ids) for c in cand]
        conts = llm.generate([prefix + c + "\n" for c in cand], [SamplingParams(n=8, max_tokens=CAP - u, temperature=1.0, top_p=0.95, top_k=20,
                                                                      presence_penalty=1.5, seed=100 + q) for q, u in enumerate(used)])
        out[name] = [{"line": c, "unit": unit(c), "right": [grade(o) for o in r.outputs]} for c, r in zip(cand, conts)]
        print(name, {u: sum(x["unit"] == u for x in out[name]) for u in ("dollars", "cents", "other")}, flush=True)
    # C: next token after "A nickel is worth "
    indent = l13[:len(l13) - len(l13.lstrip())]
    stem = head + base + l13 + "\n" + indent + "    *   A nickel is worth "
    lp = llm.generate([stem], SamplingParams(max_tokens=1, temperature=1.0, logprobs=20))[0].outputs[0].logprobs[0]
    out["next_token"] = sorted([(v.decoded_token, float(v.logprob)) for v in lp.values()], key=lambda x: -x[1])
    return out

@app.local_entrypoint()
def main():
    HERE = "/home/three/interp/wsp/"
    d = json.load(open(HERE + "resample.json"))
    i, j = d["lost"][6]
    r = d["rows"][i]
    lines = d["stage1"][i][j]["text"].split("</think>")[0].split("\n")
    assert "$75$" in lines[13], lines[13]
    out = run.remote(r["problem"], r["answer"], lines)
    json.dump(out, open(HERE + "units.json", "w"))
    print("saved units.json")
