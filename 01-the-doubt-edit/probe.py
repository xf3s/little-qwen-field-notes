"""Who pushes "Wait" over "</think>"? Direct logit attribution on Qwen3.5-0.8B over its own transcript.
At each line start in the thinking (the decision point for a new thought), the final residual is a sum of the embedding plus
every layer's attention and MLP outputs. Each piece's push = its dot with (W_U[Wait] - W_U[</think>]), scaled by the final
RMSNorm's 1/rms at that position (the norm weight folded into W_U). Per head too: full-attention heads through o_proj column
slices, linear-attention heads through out_proj column slices.
    python probe.py RUNS_FILE N   -> probe_<N>.json, prints top components
"""
import json, sys, time, torch
from transformers import AutoTokenizer, AutoModelForImageTextToText
torch.set_num_threads(3)
M = "/home/three/models/qwen3.5-0.8b"
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForImageTextToText.from_pretrained(M, dtype=torch.float32).eval()
lm = model.model.language_model
layers = lm.layers
WAIT = [13784, 13428]; END = 248069
import os
for spec in filter(None, os.environ.get("EDIT", "").split(",")):   # e.g. EDIT=16.mlp.998=-1, same specs as cloud.py
    k, v = spec.rsplit("=", 1); L, n = int(k.split(".")[0]), int(k.split(".")[-1])
    with torch.no_grad(): layers[L].mlp.down_proj.weight[:, n] *= float(v)
    print("edited", k, "x", v)

SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."

def build(problem, text, thinking):
    # 11-02 fix: the template defaults to NOT thinking, so the first probe's prompt read "<think></think><think>" and lacked
    # the suffix the runs had. Now it's exactly what the runs saw: thinking -> ends "<think>\n"; not -> an empty think block.
    head = tok.apply_chat_template([{"role": "user", "content": problem + SUFFIX}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=thinking)
    return tok(head, add_special_tokens=False).input_ids, tok(text, add_special_tokens=False).input_ids

def run(problem, think, thinking=True, target="end", maxlen=2500):
    h, t = build(problem, think, thinking)
    ids = torch.tensor([(h + t)[:maxlen]])
    cap = {}
    def pre(name):
        def f(mod, args): cap[name] = args[0].detach()
        return f
    def post(name):
        def f(mod, args, out): cap[name] = (out[0] if isinstance(out, tuple) else out).detach()
        return f
    hs = []
    for i, L in enumerate(layers):
        att = L.self_attn if hasattr(L, "self_attn") else L.linear_attn
        proj = att.o_proj if hasattr(att, "o_proj") else att.out_proj
        hs += [proj.register_forward_pre_hook(pre(f"in{i}")), att.register_forward_hook(post(f"att{i}")), L.mlp.register_forward_hook(post(f"mlp{i}"))]
    hs.append(lm.norm.register_forward_pre_hook(pre("final")))
    with torch.no_grad(): out = model(input_ids=ids)
    for x in hs: x.remove()
    logits = out.logits[0]
    W = model.lm_head.weight if hasattr(model, "lm_head") else lm.embed_tokens.weight
    g = lm.norm.weight
    # Qwen3.5 RMSNorm may use (1 + weight); detect by checking the final norm reproduces the logits
    x = cap["final"][0]
    eps = getattr(lm.norm, "variance_epsilon", getattr(lm.norm, "eps", 1e-6))
    rms = (x.pow(2).mean(-1, keepdim=True) + eps).rsqrt()
    for gg in (g, 1 + g):
        test = (x * rms * gg) @ W[[WAIT[0], END]].T
        if torch.allclose(test, logits[:, [WAIT[0], END]], atol=1e-2, rtol=1e-3): break
    else:
        print("WARNING: final-norm reconstruction off by", (test - logits[:, [WAIT[0], END]]).abs().max().item())
    # positions: line starts inside the thinking
    ids0 = ids[0].tolist(); start = len(h)
    pos = [p for p in range(start, len(ids0) - 1) if "\n" in tok.decode([ids0[p]])]
    k = logits[pos][:, WAIT].argmax(-1)                      # which Wait (with or without a space) this spot prefers
    wid = torch.tensor(WAIT)[k]
    # target "end": Wait vs </think> (thinking only); "mean": Wait vs the average word, comparable across modes
    other = logits[pos, END] if target == "end" else logits[pos].mean(-1)
    Wo = W[END] if target == "end" else W.mean(0)
    if target == "close":                                    # who pushes </think> itself (vs the average word)
        wid = torch.full_like(wid, END)
    if target == "vstop":                                    # </think> vs the word the model actually preferred there
        top = logits[pos].argmax(-1)
        other = logits[pos, top]; Wo = W[top]; wid = torch.full_like(wid, END)
    wait_logit = logits[pos, wid]; end_logit = other
    D = (W[wid] - Wo) * gg                               # [P, 1024]: per-position direction, norm weight folded in
    sc = rms[pos]                                            # [P,1]
    dot = lambda comp: ((comp * sc) * D).sum(-1)
    res = {}
    for i, L in enumerate(layers):
        att = L.self_attn if hasattr(L, "self_attn") else L.linear_attn
        proj = att.o_proj if hasattr(att, "o_proj") else att.out_proj
        res[f"L{i}.mlp"] = dot(cap[f"mlp{i}"][0, pos])
        res[f"L{i}.attn"] = dot(cap[f"att{i}"][0, pos])
        inp = cap[f"in{i}"][0, pos]                              # [P, heads*hd]
        nh = 8 if hasattr(L, "self_attn") else 16
        hd = inp.shape[-1] // nh
        Wp = proj.weight                                         # [1024, nh*hd]
        for hh in range(nh):
            res[f"L{i}.{'attn' if nh == 8 else 'lin'}.h{hh}"] = dot(inp[:, hh*hd:(hh+1)*hd] @ Wp[:, hh*hd:(hh+1)*hd].T)
    res["embed"] = dot(lm.embed_tokens(ids)[0, pos])
    total = sum(v for k, v in res.items() if k.endswith((".mlp", ".attn")) or k == "embed")
    return pos, ids0, (wait_logit - end_logit), total, res

if __name__ == "__main__":
    runs = json.load(open(sys.argv[1])); n = int(sys.argv[2])
    thinking = (sys.argv[3] if len(sys.argv) > 3 else "think") == "think"
    target = sys.argv[4] if len(sys.argv) > 4 else "end"
    tag = f"{sys.argv[1].split('runs_')[-1].removesuffix('.json')}_{n}_{target}"
    probs = [json.loads(l) for l in open("/home/three/interp/wsp/math500.jsonl")]
    r = runs[n]; t0 = time.time()
    pos, ids0, gap, total, res = run(probs[r["idx"]]["problem"], r["think"], thinking, target)
    print(f"{len(ids0)} tokens, {len(pos)} line starts, {time.time()-t0:.0f}s; mean logit gap Wait-minus-{target} {gap.mean():.2f}; DLA sum {total.mean():.2f} (must match)")
    means = {k: v.mean().item() for k, v in res.items()}
    print("biggest pushes toward Wait (mean over line starts):")
    for k, v in sorted(means.items(), key=lambda kv: -kv[1])[:15]: print(f"  {k:16s} {v:+.3f}")
    print("biggest pushes toward ending:")
    for k, v in sorted(means.items(), key=lambda kv: kv[1])[:8]: print(f"  {k:16s} {v:+.3f}")
    json.dump({"idx": r["idx"], "means": means, "gap": gap.tolist(), "why": r["why"]}, open(f"/home/three/interp/wsp/probe_{tag}.json", "w"))
