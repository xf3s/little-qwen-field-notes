"""What drives L16 neuron 998 negative when the model thinks? (11-02 night.) The neuron's activation is silu(g) * u with
g = gate_row . norm(x), u = up_row . norm(x), x = the notepad entering L16's MLP. x is the embedding plus every earlier
attention and MLP output plus L16's attention, so g and u split exactly into per-component shares (through the norm's
scale at each position). Compare a thinking run with a not-thinking run of the same problem.
    python upstream.py RUNS_FILE N think|nothink [RUNS_FILE N think|nothink]
"""
import json, sys, torch
import probe
from probe import tok, model, lm, layers

LAYER, NEURON = 16, 998

def shares(problem, text, thinking, maxlen=2500):
    h, t = probe.build(problem, text, thinking)
    ids = torch.tensor([(h + t)[:maxlen]])
    cap, hk = {}, []
    def post(name):
        def f(mod, args, out): cap[name] = (out[0] if isinstance(out, tuple) else out).detach()
        return f
    for i in range(LAYER + 1):
        L = layers[i]
        att = L.self_attn if hasattr(L, "self_attn") else L.linear_attn
        hk.append(att.register_forward_hook(post(f"L{i}.attn")))
        if i < LAYER: hk.append(L.mlp.register_forward_hook(post(f"L{i}.mlp")))
    ln = layers[LAYER].post_attention_layernorm
    hk.append(ln.register_forward_pre_hook(lambda m, a: cap.__setitem__("x", a[0].detach())))
    hk.append(ln.register_forward_hook(lambda m, a, o: cap.__setitem__("xn", o.detach())))
    hk.append(layers[LAYER].mlp.down_proj.register_forward_pre_hook(lambda m, a: cap.__setitem__("act", a[0].detach())))
    with torch.no_grad(): model(input_ids=ids)
    for x in hk: x.remove()
    ids0 = ids[0].tolist()
    pos = [p for p in range(len(h), len(ids0) - 1) if "\n" in tok.decode([ids0[p]])]
    x = cap["x"][0]
    eps = getattr(ln, "variance_epsilon", getattr(ln, "eps", 1e-6))
    rms = (x.pow(2).mean(-1, keepdim=True) + eps).rsqrt()
    w = ln.weight
    gg = w if torch.allclose(x * rms * w, cap["xn"][0], atol=1e-3) else 1 + w
    assert torch.allclose(x * rms * gg, cap["xn"][0], atol=1e-3), "norm reconstruction failed"
    mlp = layers[LAYER].mlp
    G, U = mlp.gate_proj.weight[NEURON] * gg, mlp.up_proj.weight[NEURON] * gg
    comps = {k: v[0] for k, v in cap.items() if k.startswith("L")}
    comps["embed"] = lm.embed_tokens(ids)[0].detach()
    sc = rms[pos]
    g = {k: ((v[pos] * sc) @ G) for k, v in comps.items()}
    u = {k: ((v[pos] * sc) @ U) for k, v in comps.items()}
    gt, ut = sum(g.values()), sum(u.values())
    act = cap["act"][0, pos, NEURON]
    recon = torch.nn.functional.silu(gt) * ut
    return {"act": act.mean().item(), "recon": recon.mean().item(), "g": gt.mean().item(), "u": ut.mean().item(),
            "g_parts": {k: v.mean().item() for k, v in g.items()}, "u_parts": {k: v.mean().item() for k, v in u.items()}}

if __name__ == "__main__":
    a = sys.argv[1:]
    probs = [json.loads(l) for l in open("/home/three/interp/wsp/math500.jsonl")]
    out = {}
    for i in range(0, len(a), 3):
        r = json.load(open(a[i]))[int(a[i + 1])]
        s = shares(probs[r["idx"]]["problem"], r["think"], a[i + 2] == "think")
        key = f"{a[i + 2]}#{r['idx']}"
        out[key] = s
        print(f"{key}: act {s['act']:+.3f} (recon {s['recon']:+.3f}); gate {s['g']:+.3f}, up {s['u']:+.3f}")
    json.dump(out, open("/home/three/interp/wsp/upstream_998.json", "w"))
    if len(out) == 2:
        k1, k2 = list(out)
        for part in ("g_parts", "u_parts"):
            d = {k: out[k1][part][k] - out[k2][part][k] for k in out[k1][part]}
            print(f"{part[0]}: biggest differences [{k1}] minus [{k2}]:")
            for k, v in sorted(d.items(), key=lambda kv: -abs(kv[1]))[:10]:
                print(f"   {k:10s} {v:+.3f}   ({out[k1][part][k]:+.3f} vs {out[k2][part][k]:+.3f})")
