"""Who writes the L13 "next line reconsiders" direction? (11-03, three: "go ahead with the layer 13 edit".)
The notepad after L13 is the embedding plus every attention and MLP output of L0..L13. Each piece's share along the
unit doubt direction d (doubt_dirs.pt, layer 13) at '\\n' line starts, RECONSIDER minus COMPUTE, says who puts the
difference there. Those are the writers whose output matrices an edit would project d out of.
    python writers.py RUNS_FILE N [N ...]  -> writers_L13.json
"""
import json, sys, torch
import probe, doubt
from probe import tok, model, lm, layers

L13 = 13
d = torch.load("/home/three/interp/wsp/doubt_dirs.pt")["dirs"][L13]; d = d / d.norm()

def shares(problem, text, maxlen=2500):
    h, t = probe.build(problem, text, True)
    ids = torch.tensor([(h + t)[:maxlen]])
    cap, hk = {}, []
    def post(name):
        def f(mod, args, out): cap[name] = (out[0] if isinstance(out, tuple) else out).detach()
        return f
    for i in range(L13 + 1):
        att = layers[i].self_attn if hasattr(layers[i], "self_attn") else layers[i].linear_attn
        hk += [att.register_forward_hook(post(f"L{i}.attn")), layers[i].mlp.register_forward_hook(post(f"L{i}.mlp"))]
    with torch.no_grad(): out = model(input_ids=ids, output_hidden_states=True)
    for x in hk: x.remove()
    comps = {k: v[0] @ d for k, v in cap.items()}
    comps["embed"] = lm.embed_tokens(ids)[0].detach() @ d
    total = out.hidden_states[L13 + 1][0] @ d
    assert torch.allclose(sum(comps.values()), total, atol=1e-2), "shares don't add up to the notepad"
    ids0 = ids[0].tolist(); rec, com = [], []
    for p in range(len(h), len(ids0) - 1):
        if tok.decode([ids0[p]]) != "\n": continue
        nxt = tok.decode(ids0[p + 1:p + 12]).split("\n")[0]
        if not nxt.strip(): continue
        if doubt.RECON.match(nxt): rec.append(p)
        elif doubt.COMP.match(nxt): com.append(p)
    return {k: (v[rec].sum().item(), v[com].sum().item()) for k, v in comps.items()}, len(rec), len(com)

if __name__ == "__main__":
    runs = json.load(open(sys.argv[1])); probs = [json.loads(l) for l in open("/home/three/interp/wsp/math500.jsonl")]
    R, C, nr, nc = {}, {}, 0, 0
    for n in map(int, sys.argv[2:]):
        r = runs[n]; s, a, b = shares(probs[r["idx"]]["problem"], r["think"])
        for k, (x, y) in s.items(): R[k] = R.get(k, 0) + x; C[k] = C.get(k, 0) + y
        nr += a; nc += b; print(f"run {n}: {a} reconsider, {b} compute", flush=True)
    diff = {k: R[k] / nr - C[k] / nc for k in R}
    tot = sum(diff.values())
    print(f"along d at L13: reconsider minus compute = {tot:+.3f} (sum of shares)")
    for k, v in sorted(diff.items(), key=lambda kv: -abs(kv[1]))[:16]:
        print(f"   {k:10s} {v:+.3f}  ({100 * v / tot:5.1f}%)   mean at reconsider {R[k] / nr:+.3f}, at compute {C[k] / nc:+.3f}")
    json.dump({"diff": diff, "rec": {k: R[k] / nr for k in R}, "com": {k: C[k] / nc for k in C}, "n": [nr, nc]},
              open("/home/three/interp/wsp/writers_L13.json", "w"), indent=1)
