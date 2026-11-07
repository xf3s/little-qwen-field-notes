"""A "done" direction (11-03, three: "meownds good!!!" to: learn a direction for "I have the answer, close the thinking").
On the model as edited by our best recipe (n998 x-2, doubt direction out of L0-13), over its own stopped dev transcripts:
DONE = the last position of the thinking, where the model chose </think>; CONT = earlier line starts with the same token,
away from the end. Per layer, difference of class means; tested on runs of problems it wasn't fit on (balanced accuracy).
Runs on a Modal GPU because clay's CPU takes ~1.5 min per transcript.
    modal run done_cloud.py --runs runs_cloud_proj0-13both_n998x-2.json --dirs doubt_dirs_both.pt  -> done_dirs.pt
"""
import modal
app = modal.App("wsp-done")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "transformers", "accelerate", "safetensors",
                                                                   "huggingface_hub[hf_transfer]").env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
cache = modal.Volume.from_name("wsp-hf-cache", create_if_missing=True)
MODEL = "Qwen/Qwen3.5-0.8B"
SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."

@app.function(image=image, gpu=["L4", "A10G", "L40S"], timeout=40 * 60, volumes={"/root/.cache/huggingface": cache})
def collect(items, direction):
    import torch, random
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float32).cuda().eval()
    layers = model.model.language_model.layers
    d = torch.tensor(direction, dtype=torch.float32).cuda()
    with torch.no_grad():                                    # the same edits as cloud.py's edited()
        layers[16].mlp.down_proj.weight[:, 998] *= -2
        for i in range(14):
            att = layers[i].self_attn if hasattr(layers[i], "self_attn") else layers[i].linear_attn
            for W in ((att.o_proj if hasattr(att, "o_proj") else att.out_proj).weight, layers[i].mlp.down_proj.weight):
                W -= torch.outer(d, d @ W)
    cache.commit()
    out = []
    for it in items:
        head = tok.apply_chat_template([{"role": "user", "content": it["problem"] + SUFFIX}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)
        h = tok(head, add_special_tokens=False).input_ids; t = tok(it["think"], add_special_tokens=False).input_ids
        ids = h + t
        with torch.no_grad(): hs = model(input_ids=torch.tensor([ids]).cuda(), output_hidden_states=True).hidden_states
        H = torch.stack(hs[1:])[:, 0]                       # [24, T, 1024]
        last = len(ids) - 1; tid = ids[last]
        cont = [p for p in range(len(h), last - 60) if ids[p] == tid]   # same token, well before the end
        random.Random(it["n"]).shuffle(cont); cont = cont[:12]
        out.append({"n": it["n"], "idx": it["idx"], "tok": tok.decode([tid]), "done": H[:, last].cpu(),
                    "cont": H[:, cont].cpu() if cont else None})
    return out

@app.local_entrypoint()
def main(runs: str = "runs_cloud_proj0-13both_n998x-2.json", dirs: str = "doubt_dirs_both.pt"):
    import json, torch, collections
    HERE = "/home/three/interp/wsp/"
    probs = [json.loads(l) for l in open(HERE + "math500.jsonl")]
    R = json.load(open(HERE + runs))
    items = [{"n": n, "idx": r["idx"], "problem": probs[r["idx"]]["problem"], "think": r["think"]}
             for n, r in enumerate(R) if r["split"] == "dev" and r["why"] == "stop"]
    dv = torch.load(HERE + dirs)["dirs"][13]; dv = (dv / dv.norm()).tolist()
    res = collect.remote(items, dv)
    print("DONE tokens:", collections.Counter(x["tok"] for x in res).most_common(6))
    probs_seen = sorted({x["idx"] for x in res}); test = set(probs_seen[::4])   # a quarter of the problems held out
    def stack(rows, key):
        xs = [x[key] if key == "cont" else x[key][:, None] for x in rows if x[key] is not None]
        return torch.cat(xs, 1) if xs else None                                  # [24, N, 1024]
    tr = [x for x in res if x["idx"] not in test]; te = [x for x in res if x["idx"] in test]
    D, C, Dt, Ct = stack(tr, "done"), stack(tr, "cont"), stack(te, "done"), stack(te, "cont")
    print(f"train: {D.shape[1]} done, {C.shape[1]} cont; test ({len(test)} problems): {Dt.shape[1]} done, {Ct.shape[1]} cont")
    dirs_out = []
    print("layer  balanced-acc on held problems")
    for L in range(24):
        mu1, mu0 = D[L].mean(0), C[L].mean(0); dd = mu1 - mu0; dirs_out.append(dd)
        thr = ((mu1 + mu0) / 2) @ dd
        acc = 0.5 * ((Dt[L] @ dd > thr).float().mean() + (Ct[L] @ dd <= thr).float().mean())
        print(f"L{L:<4d} {acc:.2f}")
    torch.save({"dirs": torch.stack(dirs_out), "n_done": D.shape[1], "n_cont": C.shape[1]}, HERE + "done_dirs.pt")
