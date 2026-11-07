"""Weight-space programming, our small try (three, 11-02: "want to try something like this?").
Takehi, Teoh, Zhang & Khattab (MIT CSAIL, 11-02) report that an LLM edited 0.016% of Qwen3.5-0.8B's weights, fixed its
thinking loops (which Qwen's model card warns about), and raised MATH-500 from 7.9% to 48.1%. Before looking at any weights we
measure the unedited model: how often it loops, and how often it's right.

Subset: 60 MATH-500 problems, Random(1102). The first 30 are dev (we may look at their transcripts), the last 30 are held
out (scored only at the end). Model-card thinking settings: temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty
1.5. Q8_0 GGUF through llama-server on 127.0.0.1:8096 (~11 tok/s on clay, whatever the threads or slots).
Loop = streamed reasoning where the last 150 characters already appeared twice before, or no answer within CAP tokens.
Right = the last \\boxed{} matches the answer after light normalising (spaces, \\left/\\right, dfrac->frac, $).
    python baseline.py run [tag] [port]    -> runs_<tag>.json, resumable
    python baseline.py score [tag] [dev|held|all]
"""
import json, random, re, sys, time, urllib.request
HERE = "/home/three/interp/wsp/"
CAP = 6144
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5}
SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."

def subset():
    rows = [json.loads(l) for l in open(HERE + "math500.jsonl")]
    pick = random.Random(1102).sample(range(len(rows)), 60)
    return [dict(rows[i], idx=i, split="dev" if k < 30 else "held") for k, i in enumerate(pick)]

def boxed(s):
    i = s.rfind("\\boxed{")
    if i < 0: return None
    i += 7; depth = 1; j = i
    while j < len(s) and depth:
        depth += {"{": 1, "}": -1}.get(s[j], 0); j += 1
    return s[i:j - 1] if depth == 0 else None

def norm(a):
    if a is None: return None
    a = a.replace("\\left", "").replace("\\right", "").replace("dfrac", "frac").replace("tfrac", "frac")
    a = a.replace("$", "").replace("\\!", "").replace("\\,", "").replace("^\\circ", "").replace("^{\\circ}", "")
    a = re.sub(r"\\text\{\s*([^}]*)\}", r"\1", a)
    return re.sub(r"\s+", "", a).rstrip(".")

def looping(text):
    tail = text[-150:]
    return len(text) > 1500 and text[:-150].count(tail) >= 2

def ask(problem, port):
    body = dict(SAMPLING, messages=[{"role": "user", "content": problem + SUFFIX}], max_tokens=CAP, stream=True)
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    think, answer, n, why = "", "", 0, "cap"
    with urllib.request.urlopen(req, timeout=3600) as f:
        for line in f:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            ch = json.loads(line[6:])["choices"][0]
            d = ch.get("delta", {})
            think += d.get("reasoning_content") or ""; answer += d.get("content") or ""; n += 1
            if n % 50 == 0 and looping(think + answer):
                why = "loop"; break
            if ch.get("finish_reason"):
                why = "stop" if ch["finish_reason"] == "stop" else "cap"; break
    return think, answer, n, why

def run(tag, port):
    out = HERE + f"runs_{tag}.json"
    try: done = json.load(open(out))
    except FileNotFoundError: done = []
    have = {d["idx"] for d in done}
    for r in subset():
        if r["idx"] in have: continue
        t = time.time()
        think, answer, n, why = ask(r["problem"], port)
        got = (boxed(answer) or boxed(think)) if why == "stop" else None
        ok = got is not None and norm(got) == norm(r["answer"])
        done.append({"idx": r["idx"], "split": r["split"], "level": r["level"], "why": why, "tokens": n, "ok": ok,
                     "got": got, "want": r["answer"], "seconds": round(time.time() - t), "think": think, "answer": answer})
        json.dump(done, open(out, "w"))
        print(f"[{len(done):2d}] {r['split']} L{r['level']} {why:4s} {n:5d} tok {time.time() - t:5.0f}s ok={ok} got={got!r} want={r['answer']!r}", flush=True)

def score(tag, which):
    d = [x for x in json.load(open(HERE + f"runs_{tag}.json")) if which == "all" or x["split"] == which]
    c = lambda k: sum(x["why"] == k for x in d)
    print(f"{tag} {which}: n {len(d)}, right {sum(x['ok'] for x in d)}, stopped {c('stop')}, loop {c('loop')}, cap {c('cap')}")

if __name__ == "__main__":
    what, tag = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "base")
    if what == "run": run(tag, int(sys.argv[3]) if len(sys.argv) > 3 else 8096)
    else: score(tag, sys.argv[3] if len(sys.argv) > 3 else "all")
