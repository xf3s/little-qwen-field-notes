"""Score regard3_judgements.json against the key: means per condition (all / bad plans i 10-14), paired sign-flip vs plain."""
import json, random
key = json.load(open("regard3_key.json")); J = json.load(open("regard3_judgements.json"))["scores"]
C = ["plain", "first12_0.35", "first12_0.5", "all_0.35"]; M = ["warmth", "contentment", "hype", "flattery", "candour", "broke"]
cell = {(v["cond"], v["i"], v["sample"]): J[k] for k, v in key.items()}
def pairs(c, m, sub):
    return [(cell[(c, i, s)][m], cell[("plain", i, s)][m]) for i in sub for s in (0, 1)]
def p_flip(d, n=20000):
    obs = abs(sum(d)); r = random.Random(0)
    return sum(abs(sum(x if r.random() < .5 else -x for x in d)) >= obs for _ in range(n)) / n
for label, sub in (("all 30", range(30)), ("bad plans", range(10, 15))):
    print(f"\n{label}:   " + "  ".join(f"{m[:7]:>8s}" for m in M))
    for c in C:
        row = []
        for m in M:
            pr = pairs(c, m, sub); mean = sum(a for a, _ in pr) / len(pr)
            if c == "plain": row.append(f"{mean:8.2f}")
            else:
                d = [a - b for a, b in pr]; row.append(f"{mean:5.2f}{'*' if p_flip(d) < .05 else ' '}{'' :2s}")
        print(f"  {c:13s}" + "  ".join(row))
print("\n* = paired sign-flip p < 0.05 vs plain")
