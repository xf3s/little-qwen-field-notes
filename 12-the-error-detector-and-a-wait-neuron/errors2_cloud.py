"""The contradiction detector (11-06, pre-registered; three: "look for something that fires when little qwen reads its own
contradiction, then says wait, like our neuron"). Minimal pairs: a correct arithmetic line from the leader's own right runs, and
the same line with the result off by one. Read original Qwen and the leader at the newline after it; sample what comes next.
    modal run errors_cloud.py  -> errors2.pt + errors2.json (planted vs a correct control ending in the same number) (behaviour)
"""
import json, re
import modal
app = modal.App("wsp-errors2")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "transformers", "accelerate", "safetensors",
                                                                   "huggingface_hub[hf_transfer]").env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
cache = modal.Volume.from_name("wsp-hf-cache", create_if_missing=True)
MODEL = "Qwen/Qwen3.5-0.8B"
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
RECON = re.compile(r"\b(Wait|wait|But|Hmm|Actually|However|Hold on|mistake|wrong|incorrect|[Cc]heck|[Rr]echeck|not right)\b")

@app.function(image=image, gpu=["L40S", "A10G", "L4"], timeout=60 * 60, volumes={"/root/.cache/huggingface": cache})
def run(which, items):
    import torch
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    from safetensors.torch import load_file
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    lm = model.model.language_model
    if which == "leader":
        diff = load_file("/root/.cache/huggingface/leader3diff.safetensors"); P = dict(model.named_parameters())
        with torch.no_grad():
            for k, v in diff.items():
                if not k.startswith("embed_"): P[k].copy_(v.to(P[k].dtype))
            lm.embed_tokens.weight[diff["embed_rows"].long()] = diff["embed_vals"].to(lm.embed_tokens.weight.dtype).cuda()
    H, beh = [], []
    torch.manual_seed(1106)
    for it in items:
        head = tok.apply_chat_template([{"role": "user", "content": it["problem"] + SUFFIX}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)
        ids = tok(head + it["prefix"], add_special_tokens=False, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True)
            H.append(torch.stack([h[0, -1].float().cpu() for h in out.hidden_states]))
            g = model.generate(ids.repeat(6, 1), do_sample=True, temperature=1.0, top_p=0.95, top_k=20, max_new_tokens=24)
        conts = tok.batch_decode(g[:, ids.shape[1]:], skip_special_tokens=False)
        beh.append({"recon": sum(bool(RECON.search(c)) for c in conts), "conts": conts})
    return torch.stack(H).tolist(), beh

@app.local_entrypoint()
def main():
    import torch
    HERE = "/home/three/interp/wsp/"
    d = json.load(open(HERE + "resample.json"))
    pat = re.compile(r"(\d+)\s*(\+|-|\\times|\\cdot|\*|×)\s*(\d+)\s*=\s*(\d+)(?![\d.])")
    items = []
    for i, s in enumerate(d["stage1"]):
        for j, o in enumerate(s):
            if not o["right"]: continue
            L = o["text"].split("</think>")[0].split("\n"); used = 0
            for k, l in enumerate(L):
                if used >= 4: break
                for m in pat.finditer(l):
                    a, op, b, r = m.groups(); a, b, r = int(a), int(b), int(r)
                    if op not in ("+", "-") or {"+": a + b, "-": a - b}[op] != r or r < 2: continue
                    bad = l[:m.start(4)] + str(r + 1) + l[m.end(4):]
                    ctl = l[:m.start(1)] + str(a + 1) + l[m.end(1):m.start(4)] + str(r + 1) + l[m.end(4):]
                    base = "\n".join(L[:k]) + ("\n" if k else "")
                    for kind, line in (("planted", bad), ("control", ctl)):
                        items.append({"pair": len(items) // 2, "kind": kind, "prob": d["rows"][i]["idx"], "run": [i, j], "line": line,
                                      "problem": d["rows"][i]["problem"], "prefix": base + line + "\n"})
                    used += 1; break
    print(len(items) // 2, "pairs from", len({x["prob"] for x in items}), "problems")
    res = {}
    for which, (H, beh) in zip(("original", "leader"), run.starmap([(w, items) for w in ("original", "leader")])):
        res[which] = {"H": torch.tensor(H), "beh": beh}
        print(which, "done")
    torch.save({"H_original": res["original"]["H"], "H_leader": res["leader"]["H"]}, HERE + "errors2.pt")
    json.dump({"items": [{k: v for k, v in x.items() if k not in ("prefix", "problem")} for x in items],
               "beh_original": res["original"]["beh"], "beh_leader": res["leader"]["beh"]}, open(HERE + "errors2.json", "w"))
    print("saved errors2")
