"""Two kinds of doubt? (11-03, three: "is there a more subtle way to connect them?" -> fit the "stuck" doubt and the "noodling"
doubt separately, then remove only the noodling one.) On the UNEDITED model, over its own thinking transcripts, at '\\n' line
starts:  STUCK  = the newline right before a line that escapes a groove (a line repeated 3+ times in a row) with Wait/But/...;
         NOODLE = the newline before an ordinary reconsider line (the previous 3 lines all differ);
         COMP   = the newline before a compute line.
Per layer, class means; then d_stuck = STUCK - COMP, d_noodle = NOODLE - COMP, and how far apart they point.
    modal run groove_cloud.py --runs runs_cloud_base_16k.json  -> groove_means.pt
"""
import modal
app = modal.App("wsp-groove")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "transformers", "accelerate", "safetensors",
                                                                   "huggingface_hub[hf_transfer]").env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
cache = modal.Volume.from_name("wsp-hf-cache", create_if_missing=True)
MODEL = "Qwen/Qwen3.5-0.8B"
SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
RECON = r"^\s*[-*]*\s*\**\s*(wait|hold on|however|but\b|maybe|is there|is it possible|what if|hmm|actually|perhaps|double[- ]check|let me re|let's re|re-?read|alternatively|or maybe|no[,.]|hang on|could it)"
COMP = r"^\s*[-*]*\s*(\$|\\\(|\\\[|\d|=)"

@app.function(image=image, gpu=["L4", "A10G", "L40S"], timeout=120 * 60, volumes={"/root/.cache/huggingface": cache})
def collect(items):
    import torch, re
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float32).cuda().eval()
    cache.commit()
    rec, comp = re.compile(RECON, re.I), re.compile(COMP)
    nl = tok("\n", add_special_tokens=False).input_ids[0]
    S = {k: torch.zeros(24, 1024) for k in ("stuck", "noodle", "comp")}; N = {k: 0 for k in S}
    keep = {"stuck": [], "noodle": [], "comp": []}   # per-example vectors (11-03 v2: with the run each came from)
    src = {"stuck": [], "noodle": [], "comp": []}
    for it in items:
        head = tok.apply_chat_template([{"role": "user", "content": it["problem"] + SUFFIX}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)
        h = tok(head, add_special_tokens=False).input_ids; t = tok(it["think"], add_special_tokens=False).input_ids
        ids = (h + t)[:14000]
        # line index of every newline token, from the text so far
        labels = {}
        lines_so_far = []
        text = ""
        for p in range(len(h), len(ids) - 1):
            if ids[p] != nl: continue
            prev = tok.decode(ids[len(h):p]).split("\n")
            prev = [re.sub(r"\s+", " ", l.strip()) for l in prev if l.strip()]
            nxt = tok.decode(ids[p + 1:p + 14]).split("\n")[0]
            if not nxt.strip(): continue
            groove = len(prev) >= 3 and prev[-1] == prev[-2] == prev[-3] and re.sub(r"\s+", " ", nxt.strip()) != prev[-1]
            calm = len(prev) >= 3 and len({prev[-1], prev[-2], prev[-3]}) == 3
            if groove and rec.match(nxt): labels[p] = "stuck"
            elif calm and rec.match(nxt): labels[p] = "noodle"
            elif calm and comp.match(nxt): labels[p] = "comp"
        if not labels: continue
        with torch.no_grad(): hs = model(input_ids=torch.tensor([ids]).cuda(), output_hidden_states=True).hidden_states
        H = torch.stack(hs[1:])[:, 0]
        for p, k in labels.items():
            S[k] += H[:, p].cpu(); N[k] += 1
            if k == "stuck" or (k == "noodle" and len(keep["noodle"]) < 3000 and p % 11 == 0) or (k == "comp" and len(keep["comp"]) < 3000 and p % 11 == 0):
                keep[k].append(H[:, p].cpu()); src[k].append(it["rid"])
        del hs, H; torch.cuda.empty_cache()
    return {k: (S[k] / max(N[k], 1)) for k in S}, N, {k: torch.stack(v) for k, v in keep.items() if v}, src

@app.local_entrypoint()
def main(runs: str = "runs_cloud_base_16k.json,runs_cloud_gsm_base.json,runs_cloud_aime_base.json"):
    import json, torch
    HERE = "/home/three/interp/wsp/"
    probs = [json.loads(l) for l in open(HERE + "math500.jsonl")]
    sets = {"runs_cloud_base_16k.json": probs, "runs_cloud_gsm_base.json": [json.loads(l) for l in open(HERE + "gsm8k60.jsonl")],
            "runs_cloud_aime_base.json": [json.loads(l) for l in open(HERE + "aime.jsonl")]}
    items = []
    for f in runs.split(","):
        for n, r in enumerate(json.load(open(HERE + f))):
            if r.get("split") == "held": continue          # the sealed half stays sealed
            items.append({"problem": sets[f][r["idx"]]["problem"], "think": r["think"], "rid": f"{f}#{n}"})
    means, N, keep, src = collect.remote(items)
    print("counts:", N, "kept", {k: v.shape[0] for k, v in keep.items()})
    print("stuck moments come from", len(set(src["stuck"])), "different runs")
    torch.save({"means": means, "n": N, "keep": keep, "src": src}, HERE + "groove_v2.pt")
    # noise check: noodle directions from random handfuls the size of the stuck set, vs the full noodle direction
    g = torch.Generator().manual_seed(1102); m = keep["stuck"].shape[0]
    print(f"noise check (handfuls of {m} noodle examples vs all noodle), and stuck vs all noodle:")
    for L in (4, 8, 13, 17, 21):
        full = means["noodle"][L] - means["comp"][L]
        hs = [torch.nn.functional.cosine_similarity(keep["noodle"][torch.randperm(keep["noodle"].shape[0], generator=g)[:m], L].mean(0)
              - means["comp"][L], full, 0).item() for _ in range(20)]
        st = torch.nn.functional.cosine_similarity(keep["stuck"][:, L].mean(0) - means["comp"][L], full, 0).item()
        print(f"   L{L}: handful-noodle vs noodle {sum(hs)/len(hs):.2f} (min {min(hs):.2f}); stuck vs noodle {st:.2f}")
    cos = torch.nn.functional.cosine_similarity
    d8 = torch.load(HERE + "doubt_dirs_both.pt")["dirs"][13]
    print("layer  cos(stuck, noodle)  cos(noodle, our 'but')  cos(stuck, our 'but')")
    for L in range(24):
        ds, dn = means["stuck"][L] - means["comp"][L], means["noodle"][L] - means["comp"][L]
        extra = f"   {cos(dn, d8, 0):.2f}   {cos(ds, d8, 0):.2f}" if L == 13 else ""
        print(f"L{L:<4d}  {cos(ds, dn, 0):.2f}{extra}")
