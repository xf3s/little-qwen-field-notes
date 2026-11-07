"""Score finish_runs.json: finished within 8k, right (math-verify on the answer after </think>), median tokens."""
import json
from math_verify import parse, verify
d = json.load(open(__import__("sys").argv[1] if len(__import__("sys").argv) > 1 else "finish_runs.json")); rows = d["rows"]
for name, res in d["runs"].items():
    out = {}
    for src in ("gsm8k", "math500", "all"):
        fin = right = n = 0; toks = []
        for r, outs in zip(rows, res):
            if src != "all" and r["src"] != src: continue
            for o in outs:
                n += 1; toks.append(o["tokens"])
                ans = o["text"]
                ok = o["closed"] and o["finish"] == "stop" and bool(ans.strip())
                fin += ok
                if ok:
                    g, p = parse("$" + r["answer"] + "$"), parse(ans)
                    right += bool(g and p and verify(g, p))
        toks.sort()
        out[src] = f"finished {fin}/{n} ({100*fin/n:.0f}%), right {right}/{n} ({100*right/n:.0f}%), median tokens {toks[n//2]}"
    print(f"{name:11s}", " | ".join(f"{k}: {v}" for k, v in out.items()))
