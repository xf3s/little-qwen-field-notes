"""Score twil_runs.json (from twil_cloud.py)."""
import json, re
from math_verify import parse, verify
d = json.load(open("twil_runs.json")); rows, runs = d["rows"], d["runs"]
for variant, res in runs.items():
    right = done = 0; lens = []; words = {w: 0 for w in ("But", "Wait", "Thus", "Given")}; think_tok = 0
    for r, outs in zip(rows, res):
        for o in outs:
            text = o["text"]; think, _, answer = text.partition("</think>")
            finished = bool(answer.strip()) and o["finish"] == "stop"
            done += finished; lens.append(o["tokens"])
            think_tok += o["tokens"] if not finished else max(1, len(think) // 4)
            for w in words: words[w] += len(re.findall(rf"\b{w}\b", think))
            if finished:
                g, p = parse("$" + r["answer"] + "$"), parse(answer)
                right += bool(g and p and verify(g, p))
    lens.sort(); n = len(lens)
    per = {w: round(1000 * c / think_tok, 2) for w, c in words.items()}
    print(f"{variant}: right {right}/{n}, finished {done}/{n}, median tokens {lens[n // 2]}, per 1k think tokens {per}")
