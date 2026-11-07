"""The baseline on a rented GPU (three, 11-02: Modal account, spending cap set; the token came by browser sign-in).
Same 60 problems, prompt and sampling as baseline.py, but through vLLM on one GPU, K samples per problem, so a run takes
minutes instead of a day. Weights download cloud-side from Hugging Face; an edited model will go up to a Volume later.
    ~/interp/.venv/bin/modal run cloud.py --tag base --k 4      -> runs_cloud_<tag>.json, then the score
"""
import json, time
import modal

app = modal.App("wsp-loops")
image = modal.Image.debian_slim(python_version="3.12").pip_install("vllm", "huggingface_hub[hf_transfer]") \
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"})  # its JIT wants nvcc, which slim lacks
cache = modal.Volume.from_name("wsp-hf-cache", create_if_missing=True)
MODEL = "Qwen/Qwen3.5-0.8B"

WIDE, NEW = 3648, 3584   # an implant widens every MLP to WIDE neurons; the new neuron is index NEW

def edited(model, head_scales, direction=None):
    """A copy of the model with some full-attention heads' outputs scaled: {"19.4": 2.0} multiplies the o_proj columns
    that carry layer 19 head 4 (256 of them; 8 heads x 256) by 2. Returns the copy's folder."""
    import os, shutil, torch, json
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file
    src, dst = snapshot_download(model), "/tmp/edited"
    shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst)
    for f in os.listdir(src):
        if not f.endswith(".safetensors") and os.path.isfile(os.path.join(src, f)): shutil.copy(os.path.join(src, f), dst)  # (a repo may hold folders, e.g. eval/)
    implants = {k: v for k, v in head_scales.items() if k.startswith("implant.")}
    if implants:   # 11-03 (three: "do you need to clobber what's already on that neuron? can you add at right angles again?"):
        # widen every MLP from 3,584 to 3,648 neurons; the new ones are zero in and zero out (they do nothing), except
        # the first new one in each implanted layer, which gets its own gate/up rows and down column
        cfg = json.load(open(os.path.join(dst, "config.json")))
        cfg["text_config"]["intermediate_size"] = WIDE
        json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
    for f in [f for f in os.listdir(src) if f.endswith(".safetensors")]:
        w = load_file(os.path.join(src, f))
        for spec, scale in head_scales.items():
            if spec.startswith("implant."): continue   # handled below, after the widening
            if spec.startswith("proj."):              # "proj.11-13": project the doubt direction out of what L11..L13 write
                a, b = map(int, spec.split(".")[1].split("@")[0].split("~")[0].split("-"))   # (11-03): W <- W - scale * d (d . W)
                d = torch.tensor(direction[spec] if isinstance(direction, dict) else direction, dtype=torch.float32)  # unit d
                for layer in range(a, b + 1):
                    for part in ("self_attn.o_proj", "linear_attn.out_proj", "mlp.down_proj"):
                        key = f"model.language_model.layers.{layer}.{part}.weight"
                        if key in w:
                            W = w[key].float(); W -= scale * torch.outer(d, d @ W); w[key] = W.to(w[key].dtype)
                            print("projected", key, "x", scale)
                continue
            if ".mlp." in spec:                       # "16.mlp.998": one MLP neuron's output column (1024 weights)
                layer, n = int(spec.split(".")[0]), int(spec.split(".")[-1])
                key = f"model.language_model.layers.{layer}.mlp.down_proj.weight"
                if key in w: w[key][:, n] *= scale; print("scaled", key, "neuron", n, "x", scale)
                continue
            layer, head = map(int, spec.split("."))
            key = f"model.language_model.layers.{layer}.self_attn.o_proj.weight"
            if key in w: w[key][:, head * 256:(head + 1) * 256] *= scale; print("scaled", key, "head", head, "x", scale)
        if implants:
            for key in list(w):
                if not (("language_model.layers." in key or key.startswith("mtp.layers.")) and ".mlp." in key and key.endswith("_proj.weight")): continue
                t = w[key]
                if key.endswith("down_proj.weight"):
                    pad = torch.zeros(t.shape[0], WIDE - t.shape[1], dtype=t.dtype); w[key] = torch.cat([t, pad], 1)
                else:
                    pad = torch.zeros(WIDE - t.shape[0], t.shape[1], dtype=t.dtype); w[key] = torch.cat([t, pad], 0)
            for spec, scale in implants.items():
                layer = int(spec.split(".")[1].split("~")[0]); vec = direction[spec]
                pre = f"model.language_model.layers.{layer}.mlp."
                if pre + "gate_proj.weight" in w:
                    w[pre + "gate_proj.weight"][NEW] = torch.tensor(vec["gate"], dtype=w[pre + "gate_proj.weight"].dtype)
                    w[pre + "up_proj.weight"][NEW] = torch.tensor(vec["up"], dtype=w[pre + "up_proj.weight"].dtype)
                    w[pre + "down_proj.weight"][:, NEW] = scale * torch.tensor(vec["down"], dtype=w[pre + "down_proj.weight"].dtype)
                    print("implanted a new neuron in", pre, "x", scale)
        save_file(w, os.path.join(dst, f), metadata={"format": "pt"})
    return dst

@app.function(image=image, gpu=["L4", "A10G", "L40S"],  # fallbacks: 11-02 the L4 queue sat 33 min
              timeout=40 * 60, volumes={"/root/.cache/huggingface": cache})
def generate(prompts, sampling, cap, k, think=True, model=MODEL, head_scales=None, direction=None):
    from vllm import LLM, SamplingParams
    t = time.time()
    if head_scales: model = edited(model, head_scales, direction)
    llm = LLM(model=model, max_model_len=cap + 1024, gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 0})
    sp = SamplingParams(n=k, max_tokens=cap, seed=1102, **sampling)
    outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], sp,
                    chat_template_kwargs={"enable_thinking": think})  # 0.8B's template defaults to NOT thinking
    cache.commit()
    res = [[{"text": o.text, "tokens": len(o.token_ids), "finish": o.finish_reason} for o in out.outputs] for out in outs]
    return res, round(time.time() - t)

REFRAME = " Before solving, write down two or three different ways to see this problem."  # 11-03: "can it rotate?"

def same_number(a, b):   # "1,000", "1000.0", "\\$18" all equal 18-style answers (GSM8K, AIME are plain numbers)
    import re
    f = lambda x: re.sub(r"\\text\{[^}]*\}|\\!|[,\\$\s]", "", str(x)).rstrip(".")   # \text{...} first (11-03 fix)
    try: return abs(float(f(a)) - float(f(b))) < 1e-6
    except ValueError: return False

@app.local_entrypoint()
def main(tag: str = "base", k: int = 4, think: bool = True, scale: str = "", split: str = "all", cap: int = 0, bench: str = "", reframe: bool = False, part: str = "", base: str = ""):
    # scale: "19.4=2" or "19.4=2,23.6=0.5" (layer.head=factor); edits run on dev only unless told otherwise
    from baseline import subset, boxed, norm, looping, SAMPLING, SUFFIX, CAP, HERE
    rows = [r for r in subset() if split == "all" or r["split"] == split]
    if bench:   # 11-03 generalization: another question set (gsm8k60 / aime), all "test": the recipe is fixed beforehand
        rows_ = [dict(json.loads(l), idx=i, split="test") for i, l in enumerate(open(HERE + bench + ".jsonl"))]
        if part: a, b = map(int, part.split(":")); rows_ = rows_[a:b]   # 11-03: a slice, so big batches fit the timeout
        rows = rows_
    hs = {a: float(b) for a, b in (x.rsplit("=", 1) for x in scale.split(","))} if scale else None
    direction = None
    if hs and any(x.startswith("proj.") or x.startswith("implant.") for x in hs):   # the L13 doubt direction (doubt.py), as a unit vector
        import torch
        # "proj.0-13" = L13's direction from DIRS (default doubt_dirs.pt); "proj.0-16@16" = L16's; "proj.14-20@18~done" =
        # L18's from done_dirs.pt ("~both" -> doubt_dirs_both.pt). A negative scale strengthens instead of removing.
        import os; direction = {}
        for x in hs:
            if x.startswith("implant."):   # the needle-lifter neuron's vectors, made by implant.py
                direction[x] = torch.load(HERE + ("implant_vectors_exit.pt" if "~exit" in x else "implant_vectors_late.pt" if "~late" in x else "implant_vectors_cal.pt" if "~cal" in x else "implant_vectors.pt")); direction[x] = {k: v.tolist() for k, v in direction[x].items()}
                continue
            if not x.startswith("proj."): continue
            at = int(x.split("@")[1].split("~")[0]) if "@" in x else 13
            name = x.split("~")[1] if "~" in x else None
            f = {"done": "done_dirs.pt", None: os.environ.get("DIRS", "doubt_dirs.pt")}.get(name, f"doubt_dirs_{name}.pt")
            dv = torch.load(HERE + f)["dirs"][at]; direction[x] = (dv / dv.norm()).tolist()
    sfx = SUFFIX + (REFRAME if reframe else "")
    extra = {"model": base} if base else {}   # 11-03: start from another checkpoint (e.g. RikiyaT's published edit)
    res, secs = generate.remote([r["problem"] + sfx for r in rows], SAMPLING, cap or CAP, k, think, head_scales=hs, **extra,  # cap: 11-03 16k
                                direction=direction)
    done = []
    for r, samples in zip(rows, res):
        for s in samples:
            text = s["text"]
            think, _, answer = text.partition("</think>")
            why = "stop" if s["finish"] == "stop" else ("loop" if looping(text) else "cap")
            got = (boxed(answer) or boxed(think)) if why == "stop" else None
            ok = got is not None and (norm(got) == norm(r["answer"]) or same_number(got, r["answer"]))
            done.append({"idx": r["idx"], "split": r["split"], "level": r["level"], "why": why, "tokens": s["tokens"],
                         "ok": ok, "got": got, "want": r["answer"], "waits": think.count("Wait"), "think": think, "answer": answer})
    json.dump(done, open(HERE + f"runs_cloud_{tag}.json", "w"))
    print(f"GPU time {secs}s for {len(done)} samples")
    for which in (("test",) if bench else ("dev",)):  # held is scored only at the very end (logbook, 11-02 14:55)
        d = [x for x in done if x["split"] == which]
        c = lambda w: sum(x["why"] == w for x in d)
        print(f"{tag} {which}: n {len(d)}, right {sum(x['ok'] for x in d)} ({100 * sum(x['ok'] for x in d) / len(d):.0f}%), "
              f"stopped {c('stop')}, loop {c('loop')}, cap {c('cap')}")
