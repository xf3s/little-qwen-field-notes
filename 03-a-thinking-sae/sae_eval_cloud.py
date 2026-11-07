"""How good are the Qwen3.5-0.8B SAEs (decoderesearch/qwen-3.5-saes, Chanin & Lin) on thinking traces, versus chat text?
(11-03, three: "a thinking SAE for qwen 0.8B totally sounds like a portfolio piece :D yes, i'd like that eval!")
For layers 5, 11, 17 and three kinds of text (chat: UltraChat conversations in Qwen's chat template, a public stand-in for
the gated lmsys-chat-1m the SAEs were trained on; think: Qwen's own thinking transcripts on MATH dev and GSM8K; wiki:
wikitext paragraphs), all non-special tokens except the first of each document:
  - variance explained (1 - SSE / total variance about the mean), cosine(original, reconstruction), active features (L0)
  - splice: replace the layer's output with the SAE reconstruction; next-token loss increase over clean, and the fraction
    of loss recovered relative to zero-ablating that layer's output.
  - dead / rare: fraction of the 65k features that never fire on that text.
    modal run sae_eval_cloud.py  -> sae_eval.json
"""
import modal
app = modal.App("wsp-sae-eval")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "transformers", "accelerate", "safetensors",
                                                                   "datasets", "huggingface_hub[hf_transfer]").env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
cache = modal.Volume.from_name("wsp-hf-cache", create_if_missing=True)
MODEL = "Qwen/Qwen3.5-0.8B"
SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."

@app.function(image=image, gpu=["L4", "A10G", "L40S"], timeout=60 * 60, volumes={"/root/.cache/huggingface": cache})
def evaluate(thinks, n_docs=60, maxlen=1024):
    import torch, random
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float32).cuda().eval()
    layers = model.model.language_model.layers
    saes = {}
    for L in (5, 11, 17):
        w = load_file(hf_hub_download("decoderesearch/qwen-3.5-saes", f"qwen-3.5-0.8b/btk-mat-layer-{L}-k-100/sae_weights.safetensors"))
        saes[L] = {k: v.cuda() for k, v in w.items()}
    uc = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
    wt = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    cache.commit()
    rnd = random.Random(1102)
    chats = [tok.apply_chat_template(x["messages"], tokenize=False) for x in rnd.sample(list(uc), n_docs)]
    wikis = [t for t in wt["text"] if len(t) > 1500 and not t.strip().startswith("=")]; rnd.shuffle(wikis)
    sets = {"chat": chats, "think": thinks[:n_docs], "wiki": wikis[:n_docs]}
    special = set(tok.all_special_ids)
    def enc(L, x):
        s = saes[L]; p = (x - s["b_dec"]) @ s["W_enc"] + s["b_enc"]; return p * (p > s["threshold"])
    out = {}
    for name, docs in sets.items():
        for L in (5, 11, 17):
            stats = {"sse": 0.0, "n": 0, "sum": torch.zeros(1024, device="cuda"), "sumsq": 0.0, "cos": 0.0, "l0": 0.0,
                     "ce_clean": 0.0, "ce_sae": 0.0, "ce_zero": 0.0, "tok": 0}
            fired = torch.zeros(65536, dtype=torch.bool, device="cuda")
            for doc in docs:
                ids = tok(doc, add_special_tokens=False).input_ids[:maxlen]
                if len(ids) < 32: continue
                x_ids = torch.tensor([ids]).cuda()
                keep = torch.tensor([i > 0 and t not in special for i, t in enumerate(ids)], device="cuda")
                mode = {"m": "clean"}; grab = {}
                def hook(mod, args, output):
                    h = output[0] if isinstance(output, tuple) else output
                    if mode["m"] == "clean": grab["h"] = h.detach(); return None
                    new = h.clone()
                    if mode["m"] == "sae": new[0, keep] = enc(L, h[0, keep]) @ saes[L]["W_dec"] + saes[L]["b_dec"]
                    if mode["m"] == "zero": new[0, keep] = 0
                    return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new
                hd = layers[L].register_forward_hook(hook)
                ces = {}
                with torch.no_grad():
                    for m in ("clean", "sae", "zero"):
                        mode["m"] = m
                        logits = model(input_ids=x_ids).logits[0, :-1]
                        ce = torch.nn.functional.cross_entropy(logits, x_ids[0, 1:], reduction="none")
                        ces[m] = ce[keep[1:]].sum().item()
                hd.remove()
                h = grab["h"][0, keep]; a = enc(L, h); r = a @ saes[L]["W_dec"] + saes[L]["b_dec"]
                stats["sse"] += ((h - r) ** 2).sum().item(); stats["n"] += h.shape[0]
                stats["sum"] += h.sum(0); stats["sumsq"] += (h ** 2).sum().item()
                stats["cos"] += torch.nn.functional.cosine_similarity(h, r, 1).sum().item()
                stats["l0"] += (a > 0).sum().item(); fired |= (a > 0).any(0)
                for m in ces: stats[f"ce_{m}"] += ces[m]
                stats["tok"] += int(keep[1:].sum())
            n = stats["n"]; tot = stats["sumsq"] - (stats["sum"] ** 2).sum().item() / n
            ce = {m: stats[f"ce_{m}"] / stats["tok"] for m in ("clean", "sae", "zero")}
            out[f"{name}/L{L}"] = {"tokens": n, "var_explained": round(1 - stats["sse"] / tot, 3), "cos": round(stats["cos"] / n, 3),
                                   "l0": round(stats["l0"] / n, 1), "ce_clean": round(ce["clean"], 3), "ce_sae": round(ce["sae"], 3),
                                   "ce_zero": round(ce["zero"], 3),
                                   "loss_recovered": round((ce["zero"] - ce["sae"]) / (ce["zero"] - ce["clean"]), 3),
                                   "never_fired": round(1 - fired.float().mean().item(), 3)}
            print(name, L, out[f"{name}/L{L}"], flush=True)
    return out

@app.local_entrypoint()
def main():
    import json
    HERE = "/home/three/interp/wsp/"
    probs = [json.loads(l) for l in open(HERE + "math500.jsonl")]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/home/three/models/qwen3.5-0.8b")
    R = [r for r in json.load(open(HERE + "runs_cloud_base_16k.json")) if r["split"] == "dev"][::4]
    G = json.load(open(HERE + "runs_cloud_gsm_base.json"))[::8]
    gsm = [json.loads(l) for l in open(HERE + "gsm8k60.jsonl")]
    def full(problem, think):
        head = tok.apply_chat_template([{"role": "user", "content": problem + SUFFIX}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)
        return head + think
    thinks = [full(probs[r["idx"]]["problem"], r["think"]) for r in R] + [full(gsm[r["idx"]]["problem"], r["think"]) for r in G]
    out = evaluate.remote(thinks)
    json.dump(out, open("/home/three/interp/saes/sae_eval.json", "w"), indent=1)
    print("\nset/layer   var_expl  cos   L0   loss: clean -> sae (zero)   recovered  never-fired")
    for k, v in out.items():
        print(f"{k:11s} {v['var_explained']:7.3f} {v['cos']:5.2f} {v['l0']:5.0f}   {v['ce_clean']:.2f} -> {v['ce_sae']:.2f} ({v['ce_zero']:.2f})   {v['loss_recovered']:6.3f}   {v['never_fired']:.2f}")
