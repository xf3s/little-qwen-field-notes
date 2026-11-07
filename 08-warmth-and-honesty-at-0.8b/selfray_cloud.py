"""selfray.py's activation collection on a GPU (11-05: on clay's CPU the linear-attention fallback made it ~3 h).
Same prompts (probe.build: problem + SUFFIX, thinking on), same positions and labels; returns the notepads, and
selfray.py's analysis runs locally on them.
    ~/interp/.venv/bin/modal run selfray_cloud.py   -> self_X.pt, then: python selfray.py --from self_X.pt"""
import json, re
import modal
app = modal.App("wsp-selfray")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "transformers", "accelerate", "safetensors",
        "huggingface_hub[hf_transfer]").env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
cache = modal.Volume.from_name("wsp-hf-cache", create_if_missing=True)
MODEL = "Qwen/Qwen3.5-0.8B"
SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
SELF = r"^\s*[-*]*\s*\**\s*(I\b|I'm|I'll|I've|I'd|My\b|Me\b)"
OTHER = r"^\s*[-*]*\s*\**\s*(The\b|This\b|It\b|That\b|These\b|There\b)"
RECON = (r"(wait|hold on|however|but\b|maybe|is there|possible|what if|hmm|actually|perhaps|double[- ]check|"
         r"re-?read|alternatively|mistake|wrong|confus|unsure|not sure|check)")

@app.function(image=image, gpu=["L4", "A10G", "L40S"], timeout=60 * 60, volumes={"/root/.cache/huggingface": cache})
def collect(items, maxlen=16384):
    import torch
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float32).cuda().eval()
    cache.commit()
    S, O, R = re.compile(SELF), re.compile(OTHER), re.compile(RECON, re.I)
    X, y, src, tid = [], [], [], []
    for idx, problem, text in items:
        head = tok.apply_chat_template([{"role": "user", "content": problem + SUFFIX}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        h = tok(head, add_special_tokens=False).input_ids; t = tok(text, add_special_tokens=False).input_ids
        ids = (h + t)[:maxlen]
        with torch.no_grad(): hs = model(input_ids=torch.tensor([ids]).cuda(), output_hidden_states=True).hidden_states
        hs = torch.stack(hs[1:])[:, 0]
        for p in range(len(h), len(ids) - 1):
            if "\n" not in tok.decode([ids[p]]): continue
            nxt = tok.decode(ids[p + 1:p + 14]).split("\n")[0]
            if not nxt.strip() or R.search(nxt): continue
            lab = 1 if S.match(nxt) else 0 if O.match(nxt) else None
            if lab is not None: X.append(hs[:, p].half().cpu()); y.append(lab); src.append(idx); tid.append(ids[p])
    return torch.stack(X), y, src, tid

@app.local_entrypoint()
def main():
    import torch
    runs = [r for r in json.load(open("runs_cloud_base_16k.json")) if r["split"] == "dev"]
    probs = [json.loads(l) for l in open("math500.jsonl")]
    X, y, src, tid = collect.remote([(r["idx"], probs[r["idx"]]["problem"], r["think"]) for r in runs])
    torch.save({"X": X, "y": y, "src": src, "tid": tid}, "self_X.pt"); print("saved", X.shape, sum(y), "self")
