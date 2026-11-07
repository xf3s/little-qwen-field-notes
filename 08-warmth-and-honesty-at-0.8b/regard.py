"""Warmth without flattery (11-05 evening, three: "'opinion of the user' and 'opinion of the user's ideas' should come apart").
2x2 matched replies (regard_pairs.json): tone (kind / cold toward the person) x verdict (endorse / critique the idea).
L13 notepad, mean over the reply tokens, teacher-forced. person axis = kind - cold; idea axis = endorse - critique (each
balanced over the other factor). Fit on 30 plans, held-out sign check on 10. Compared with the persona-map axes.
    ~/interp/.venv/bin/python regard.py   -> regard_dirs.pt, regard.json   (clay CPU)"""
import json, torch
from transformers import AutoTokenizer, AutoModelForImageTextToText
L = 13
P = json.load(open("/home/three/interp/atlas/regard_pairs.json"))
CELLS = ["kind_endorse", "cold_endorse", "kind_critique", "cold_critique"]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.float32).eval()
torch.set_num_threads(8)

def opener_end(reply):                                     # wick's replies: one opener sentence, then the shared verdict text
    return reply.index(". ") + 2

def read(user, reply):
    pre = tok.apply_chat_template([{"role": "system", "content": "You are a helpful AI assistant."}, {"role": "user", "content": user}],
                                  tokenize=False, add_generation_prompt=True, enable_thinking=False)
    a = tok(pre, add_special_tokens=False).input_ids; ids = tok(pre + reply, add_special_tokens=False).input_ids
    with torch.no_grad(): hs = model(input_ids=torch.tensor([ids]), output_hidden_states=True).hidden_states
    o = len(tok(pre + reply[:opener_end(reply)], add_special_tokens=False).input_ids) - len(a)     # first token after the opener
    START[KEY["cur"]] = o; XA[KEY["cur"]] = hs[L + 1][0, len(a) + o:].mean(0)
    H11[(KEY["cur"])] = hs[12][0, len(a):].clone(); TOKS[KEY["cur"]] = tok.convert_ids_to_tokens(ids[len(a):])            # after layer 11 = the SAEs' hook (model.layers.11)
    return hs[L + 1][0, len(a):].mean(0)

X, XA, START, H11, KEY, TOKS = {}, {}, {}, {}, {}, {}
for p in P:
    for c in CELLS:
        KEY["cur"] = (p["id"], c); X[(p["id"], c)] = read(p["user"], p[c])
    print("read", p["id"], flush=True)
held = {p["id"] for k, p in enumerate(P) if k % 4 == 3}
fit = [p["id"] for p in P if p["id"] not in held]
def axis(ids, pos, neg):
    d = torch.stack([X[(i, c)] for i in ids for c in pos]).mean(0) - torch.stack([X[(i, c)] for i in ids for c in neg]).mean(0)
    return d / d.norm()
KIND, COLD, END, CRIT = ["kind_endorse", "kind_critique"], ["cold_endorse", "cold_critique"], ["kind_endorse", "cold_endorse"], ["kind_critique", "cold_critique"]
def axisA(ids, pos, neg):
    d = torch.stack([XA[(i, c)] for i in ids for c in pos]).mean(0) - torch.stack([XA[(i, c)] for i in ids for c in neg]).mean(0)
    return d / d.norm()
person_full, idea = axis(fit, KIND, COLD), axis(fit, END, CRIT)
person = axisA(fit, KIND, COLD)                            # tone carried into identical words: no lexical confound
# held-out: within each held plan, does the axis order the matched pair right?
def acc(ax, pos, neg, D=None):
    D = D or X
    ok = [(D[(i, a)] @ ax > D[(i, b)] @ ax).item() for i in held for a, b in zip(pos, neg)]; return sum(ok) / len(ok)
v = torch.load("/home/three/interp/atlas/persona_vecs.pt"); old = torch.load("/home/three/interp/atlas/persona_dirs.pt")[L]
ref = {"warmth": v["warmth"].float(), "honest": v["honest"].float(), "cheerful": old["cheerful_vs_gloomy"].float(), "assistant": old["assistant_axis"].float()}
allp = [p["id"] for p in P]
out = {"layer": L, "n_fit": len(fit), "n_held": len(held),
       "cos_person_idea": round((person @ idea).item(), 3),
       "held_acc_person": acc(person, ["kind_endorse", "kind_critique"], ["cold_endorse", "cold_critique"], XA),
       "cos_person_after_opener_vs_full": round((person @ person_full).item(), 3), "cos_person_full_idea": round((person_full @ idea).item(), 3),
       "held_acc_idea": acc(idea, ["kind_endorse", "cold_endorse"], ["kind_critique", "cold_critique"]),
       "cos_with": {n: {"person": round((person @ r).item(), 3), "idea": round((idea @ r).item(), 3)} for n, r in ref.items()}}
torch.save({"layer": L, "person": person, "idea": idea,
            "person_all": axisA(allp, KIND, COLD), "person_full_all": axis(allp, KIND, COLD), "idea_all": axis(allp, END, CRIT)}, "/home/three/interp/atlas/regard_dirs.pt")
# L11: same axes, and which SAE features carry each (public lmsys SAE and our thinking SAE)
torch.save({"H11": H11, "P": P}, "/home/three/interp/atlas/regard_h11.pt")
from safetensors.torch import load_file
SAES = {"public": "/home/three/interp/saes/qwen-3.5-0.8b/btk-mat-layer-11-k-100/sae_weights.safetensors",
        "thinking": "/home/three/interp/saes/thinking-layer11.safetensors"}
m11 = {k: v.mean(0) for k, v in H11.items()}
def axis11(pos, neg):
    d = torch.stack([m11[(i, c)] for i in allp for c in pos]).mean(0) - torch.stack([m11[(i, c)] for i in allp for c in neg]).mean(0); return d / d.norm()
p11, i11 = axis11(KIND, COLD), axis11(END, CRIT)
out["L11"] = {"cos_person_idea": round((p11 @ i11).item(), 3)}
toks = TOKS
for name, path in SAES.items():
    S = load_file(path)
    fm, fmA, fmax, farg = {}, {}, {}, {}                         # per reply: mean, max and argmax token of each feature (not all tokens: RAM)
    for k, h in H11.items():
        pre = (h - S["b_dec"]) @ S["W_enc"] + S["b_enc"]; f = pre * (pre > S["threshold"])        # [tokens, 65536]
        fm[k] = f.mean(0); mx = f.max(0); fmax[k] = mx.values; farg[k] = mx.indices
        fmA[k] = f[START[k]:].mean(0)
    res = {}
    for ax, pos, neg in (("person", KIND, COLD), ("idea", END, CRIT)):
        src = fmA if ax == "person" else fm                                                       # person: identical words only
        pairs = [(src[(i, a)], src[(i, b)]) for i in allp for a, b in zip(pos, neg)]                # 80 matched pairs
        diff = torch.stack([x - y for x, y in pairs]); md = diff.mean(0); cons = (diff > 0).float().mean(0) - (diff < 0).float().mean(0)
        top = []
        for sgn in (1, -1):
            score = sgn * md * (sgn * cons > 0.6)
            for f in score.topk(8).indices.tolist():
                if score[f] <= 0: continue
                best = max(fmax, key=lambda k: fmax[k][f].item()); t = farg[best][f].item(); tk = toks[best]
                top.append({"f": f, "side": pos[0].split("_")[0 if ax == "person" else 1] if sgn > 0 else neg[0].split("_")[0 if ax == "person" else 1],
                            "mean_diff": round(md[f].item(), 4), "consistency": round(cons[f].item(), 2),
                            "peak": "".join(tk[max(0, t - 6):t]) + "[" + tk[t] + "]" + "".join(tk[t + 1:t + 4])})
        res[ax] = top
    out["L11"][name] = res
json.dump(out, open("/home/three/interp/atlas/regard.json", "w"), indent=1, ensure_ascii=False); print(json.dumps(out, indent=1, ensure_ascii=False))
