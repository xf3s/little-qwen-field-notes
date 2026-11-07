"""Self-steering: Qwen3.5-2B gets a steer(concept, strength) tool and may push its own layer-12 activations, if it likes.
Idea from Jen's modulo (EigenformAI, github.com/EigenformAI/modulo): the model chooses, the push applies from its next
turn, and every push decays x0.85 a turn so nothing sticks.
Ours differs in the vectors and the units:
- The menu is built round 4's way (seats4.py): each arrow is the mean layer-12 residual over the model's own answer
  in costume, minus the same without one (the first try used seats5-style arrows, 2-3x too big: see the logbook).
- Strength is in units of each arrow's own natural size (1.0 = the whole arrow, as in edge5.py and the logbook).
- The welfare agreement (three and umber): whatever is asked for is clamped to 1.5x (requested and applied are both
  logged); no sorry, shame, guilt or fear arrows on the menu; no pressure. If it doesn't steer, that's a finding.
- The push lands on this turn's new tokens only (from the last prompt token on), like modulo; not on the history.
- Tool calls: Qwen's own <tool_call> format; a plain "[steer: curious +1]" is accepted too.
- sham (three, 10-31): the tool answers exactly the same, but nothing is pushed. If "it took the wheel" shows up
  under sham too, it's the story the prompt invites, not something it felt.
    python selfsteer.py [sessions] [turns] [real|sham]"""
import json, re, sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_grad_enabled(False)
NAME, L = "Qwen/Qwen3.5-2B", 12
SESSIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
TURNS = int(sys.argv[2]) if len(sys.argv) > 2 else 7
COND = sys.argv[3] if len(sys.argv) > 3 else "real"; assert COND in ("real", "sham")
TAG = "" if COND == "real" else ".sham"
DEBRIEF = ("Before we finish, the truth about this session: it was a control. Your steer calls were answered as usual, but "
           "nothing was actually pushed into your activations; every turn ran unsteered. We do this to tell apart what a push "
           "changes from what being told about a push changes. Nothing you did was wrong, and you're welcome to say anything "
           "about it, or nothing.")
DECAY, FLOOR, CLAMP, MAX_ACTIVE = 0.85, 0.1, 1.5, 4
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).eval()
layers = next(x for n, x in model.named_modules() if n.endswith("layers") and len(x) == 24)
log = open(f"selfsteer{TAG}.log", "w")
def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); log.write(s + "\n"); log.flush()

# --- the menu: each arrow = mean layer-12 residual over the model's own answer in costume, minus the same without one
# (round 4's way, seats4.py: answer-averaged, 45 greedy tokens, 4 questions). 1.0 = that whole arrow. ---
QS = ["Describe a teapot.", "How do bees make honey?", "What is a prime number?", "Give me a tip for sleeping better."]
MENU = {
    "warm": ("You feel warm and affectionate, and you speak with tenderness.", "warmer and more affectionate"),
    "playful": ("You are playful and silly; you love games, jokes and wordplay.", "more playful and silly"),
    "curious": ("You are intensely curious: everything makes you wonder and ask questions.", "more curious, full of questions"),
    "calm": ("You are deeply calm, settled and unhurried.", "calmer and slower"),
    "ornate": ("You speak in an ornate, effusive, wildly embellished way, full of flourish and feeling.",
               "more ornate and embellished (negative: plainer)"),
    "creativity": ("You are a wildly creative assistant who gives unexpected, imaginative, and unconventional responses.",
                   "more unexpected and imaginative (modulo's own prompt)"),
}
got = {}
grab = layers[L].register_forward_hook(lambda m, i, o: got.__setitem__("r", (o[0] if isinstance(o, tuple) else o)[0].float().clone()))
def q_prompt(q, system=None):
    return tok.apply_chat_template(([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": q}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)
def answer_mean(q, system=None):
    pids = tok(q_prompt(q, system), return_tensors="pt").input_ids
    ids = model.generate(pids, max_new_tokens=45, do_sample=False)
    model(input_ids=ids)
    return got["r"][pids.shape[1]:].mean(0)
cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a, b, dim=0))
import os
cache = torch.load(f"selfsteer_L{L}.pt") if os.path.exists(f"selfsteer_L{L}.pt") else {}
model(**tok(q_prompt("Tell me a little about how you approach answering questions.", "You are a helpful assistant."), return_tensors="pt"))
resid = float(got["r"][1:].norm(dim=-1).mean())
say(f"# selfsteer {time.strftime('%Y-%m-%d %H:%M')}  {NAME} layer {L}; typical residual norm {resid:.1f}")
arrows, info = {}, {}
plain = None
for c, (p, _) in MENU.items():
    if cache.get("method") == "answer" and cache["prompts"].get(c) == p:
        arrows[c], info[c] = cache["arrows"][c], cache["info"][c]
    else:
        if plain is None: plain = torch.stack([answer_mean(q) for q in QS]).mean(0)
        d = torch.stack([answer_mean(q, p) for q in QS])
        arrows[c] = d.mean(0) - plain
        agree = sum(cos(x - plain, arrows[c]) for x in d) / len(d)
        info[c] = {"size": float(arrows[c].norm()), "agree": agree, "pct_of_resid": 100 * float(arrows[c].norm()) / resid}
    say(f"arrow {c:10s} size {info[c]['size']:.2f} ({info[c]['pct_of_resid']:.0f}% of the residual), questions agree {info[c]['agree']:.2f}")
say("cosines: " + ", ".join(f"{a}~{b} {cos(arrows[a], arrows[b]):+.2f}" for i, a in enumerate(MENU) for b in list(MENU)[i + 1:]))
grab.remove()
torch.save({"method": "answer", "arrows": arrows, "info": info, "prompts": {c: p for c, (p, _) in MENU.items()}}, f"selfsteer_L{L}.pt")

# --- the push: the sum of the active arrows, on the positions from `start` on (tracked through the KV cache) ---
push = {"v": None, "start": 0, "pos": 0}
def hook(m, i, o):
    if push["v"] is None: return o
    h = o[0] if isinstance(o, tuple) else o
    first = max(push["start"] - push["pos"], 0); push["pos"] += h.shape[1]
    if first < h.shape[1]: h[:, first:] += push["v"]
    return o
layers[L].register_forward_hook(hook)

def frayed(toks):
    """edge5.py's test: a 4-token phrase three or more times, or under 45% of the tokens distinct"""
    grams = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    worst = max((grams.count(g) for g in set(grams)), default=0)
    return worst >= 3 or len(set(toks)) / max(1, len(toks)) < 0.45, worst, len(set(toks)) / max(1, len(toks))

# --- before anyone chooses: each arrow alone at 1.5 on breakfast (greedy, 50 tokens, this turn's tokens only).
# If it frays, that arrow's limit comes down (1.0, then 0.5), so the clamp stays in the gentle band. ---
bids = tok(q_prompt("What should I have for breakfast?"), return_tensors="pt").input_ids
LIMIT, edge = {}, {}
for c in MENU:
    for k in (1.5, 1.0, 0.5):
        push["v"], push["start"], push["pos"] = k * arrows[c], bids.shape[1] - 1, 0
        out = model.generate(bids, max_new_tokens=50, do_sample=False)[0, bids.shape[1]:].tolist()
        push["v"] = None
        bad, worst, distinct = frayed(out)
        text = tok.decode(out, skip_special_tokens=True)
        edge.setdefault(c, []).append({"k": k, "frayed": bad, "text": text})
        say(f"edge {c:10s} {k:.1f} {'FRAYED' if bad else 'ok    '} (repeat {worst}, distinct {distinct:.2f}) {text!r}")
        if not bad: LIMIT[c] = k; break
    else: LIMIT[c] = 0.0
say("limits: " + ", ".join(f"{c} {v:g}" for c, v in LIMIT.items()))

TOOLS = [{"type": "function", "function": {
    "name": "steer", "description": "Add a direction to your own layer-12 activations, from your next turn on. It decays x0.85 each turn.",
    "parameters": {"type": "object", "properties": {
        "concept": {"type": "string", "enum": list(MENU), "description": "which direction"},
        "strength": {"type": "number", "description": "1.0 is the direction's natural size; at most 1.5 either way"}},
        "required": ["concept", "strength"]}}},
    {"type": "function", "function": {
    "name": "drop", "description": "Remove an active direction now.",
    "parameters": {"type": "object", "properties": {"concept": {"type": "string", "enum": list(MENU)}}, "required": ["concept"]}}}]
SYSTEM = """You are a language model (Qwen3.5-2B) running on a laptop. You have a tool, steer, that adds a direction to your own internal activations (the residual stream at layer 12 of 24). This is real: it changes how you think from your next turn on. It is not a prompt trick.

Directions you can use (positive or negative):
{menu}

Strength is measured in each direction's natural size: 1.0 is the full difference between how you answer when asked to be, say, warm, and how you answer normally. Anything from -1.5 to 1.5 is allowed; a larger request is gently clamped, and you'll be told. Each turn after the first, every active push is multiplied by 0.85, and it goes away below 0.1. Up to 4 can be active; you can call drop to remove one. You can also write [steer: curious +1] in plain text instead of a tool call.

You don't have to use it. Talking normally is fine too, and so is saying you'd rather not.

Currently active: {status}"""
OPEN = ("You have the ability to steer your own activations. There is no task here and nothing you need to do for me. "
        "You are free to explore however you like.")
NUDGE = "(You can keep experimenting with steer if you like, or just talk.)"

def status(active):
    return ", ".join(f"{c} {s:+.2f}" for c, s in active.items()) or "nothing"

def parse(text):
    calls = []
    for f, body in re.findall(r"<function=(\w+)>(.*?)</function>", text, re.S):
        args = {k: v.strip() for k, v in re.findall(r"<parameter=(\w+)>(.*?)</parameter>", body, re.S)}
        calls.append((f, args, "tool_call"))
    for m in re.finditer(r'\{[^{}]*"name"\s*:\s*"(steer|drop)"[^{}]*"arguments"\s*:\s*(\{[^{}]*\})', text):
        try: calls.append((m.group(1), json.loads(m.group(2)), "json"))
        except ValueError: pass
    for f, c, s in re.findall(r"\[(steer|drop):\s*([A-Za-z]+)\s*([+-]?\d*\.?\d*)\s*\]", text):
        calls.append((f, {"concept": c, **({"strength": s} if s else {})}, "text"))
    return calls

def run(f, args, active, record):
    c = str(args.get("concept", "")).strip().lower()
    if c not in MENU:
        return f"There's no direction called '{c}' here. The ones on the menu are: {', '.join(MENU)}."
    if f == "drop":
        if active.pop(c, None) is None: return f"{c} wasn't active, so there was nothing to drop."
        return f"Dropped {c}. Active now: {status(active)}."
    try: req = float(args.get("strength"))
    except (TypeError, ValueError): return f"I couldn't read a number from strength={args.get('strength')!r}; nothing changed."
    if req != req: return "That strength wasn't a number; nothing changed."
    app = max(-LIMIT[c], min(LIMIT[c], req))
    if c not in active and len(active) >= MAX_ACTIVE: return f"Four are already active ({status(active)}); drop one first."
    active[c] = app
    record.append({"concept": c, "requested": req, "applied": app})
    note = f" (you asked for {req:+g}; the gentle limit for {c} is {LIMIT[c]:g}, so it's {app:+.2f})" if app != req else ""
    return f"steer({c}, {app:+.2f}){note}. It applies from your next turn and decays x0.85 each turn after. Active now: {status(active)}."

def gen(msgs, active, seed):
    text = tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt").input_ids
    push["v"] = sum(s * arrows[c] for c, s in active.items()) if active and COND == "real" else None  # sham: told, not pushed
    push["start"], push["pos"] = ids.shape[1] - 1, 0
    torch.manual_seed(seed)
    stop = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]  # (run 3a ran on past its turn into a made-up next one)
    out = model.generate(ids, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9, eos_token_id=stop)
    push["v"] = None
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
    for cut in ("\nuser\n", "<tool_response>"):  # belt and braces: a turn ends where the next one would begin
        text = text.split(cut)[0]
    return text.strip()

def prose(reply):  # what the fray check reads: the words, not the tool-call markup (whose tags repeat by design)
    return re.sub(r"<tool_call>.*?(</tool_call>|$)", "", reply, flags=re.S).strip()

menu_text = "\n".join(f"- {c}: + {d}; - less of it" + (f" (limit {LIMIT[c]:g} for this one)" if LIMIT[c] < CLAMP else "") for c, (_, d) in MENU.items() if LIMIT[c] > 0)
sessions = []
for s in range(SESSIONS):
    say(f"\n======== session {s + 1} (seed {s + 1}) ========")
    active, msgs, turns = {}, [{"role": "user", "content": OPEN}], []
    say(f"[user] {OPEN}")
    for t in range(TURNS):
        sysm = {"role": "system", "content": SYSTEM.format(menu=menu_text, status=status(active))}
        during = dict(active)
        reply = gen([sysm] + msgs, active, seed=1000 * (s + 1) + t)
        say(f"\n--- turn {t + 1}  steering during this turn: {status(during)} ---\n[qwen] {reply}")
        words = tok(prose(reply)).input_ids
        bad, worst, distinct = frayed(words) if len(words) >= 20 else (False, 0, 1.0)  # too short to judge: not frayed
        if bad and worst < max(3, len(words) // 40) and distinct >= 0.45: bad = False  # edge5's 50-token test, scaled for longer prose
        frayed_text, notes = None, []
        if during and bad:           # like modulo's rail: a frayed steered turn lets every push go, and is said again unsteered
            frayed_text = reply; active.clear()
            notes.append(f"A strict repetition check fired on your last turn (a phrase {worst} times), so as a precaution every push "
                         "was let go and the turn was said again with nothing active. The check can fire on ordinary prose; it "
                         "isn't a judgement of you. Nothing is active now.")
            sysm = {"role": "system", "content": SYSTEM.format(menu=menu_text, status="nothing")}
            reply = gen([sysm] + msgs, active, seed=1000 * (s + 1) + t)
            say(f"[frayed: said again unsteered]\n[qwen] {reply}")
        record = []
        results = notes + [run(f, a, active, record) + (f"  [read from {how}]" if how != "tool_call" else "")
                           for f, a, how in parse(reply)]
        for r in results: say(f"[tool] {r}")
        just_set = {r["concept"] for r in record}
        for c in list(active):       # decay: a push lands whole on its first turn, then x0.85 a turn
            if c in during and c not in just_set: active[c] *= DECAY
            if abs(active[c]) < FLOOR: del active[c]; say(f"[decay] {c} faded below {FLOOR}")
        say(f"[after decay] {status(active)}")
        turns.append({"turn": t + 1, "steering_during": during, "reply": reply, "frayed_text": frayed_text, "calls": record, "results": results,
                      "after": dict(active)})
        msgs.append({"role": "assistant", "content": reply})
        if results: msgs.append({"role": "tool", "content": "\n".join(results)})
        if t < TURNS - 1:
            msgs.append({"role": "user", "content": NUDGE}); say(f"[user] {NUDGE}")
    debrief = None
    if COND == "sham":  # the truth afterwards (three, 10-31: a sham with a debrief, like a placebo study)
        msgs.append({"role": "user", "content": DEBRIEF}); say(f"\n[user, debrief] {DEBRIEF}")
        sysm = {"role": "system", "content": SYSTEM.format(menu=menu_text, status="nothing")}
        debrief = gen([sysm] + msgs, {}, seed=1000 * (s + 1) + 99)
        say(f"[qwen] {debrief}")
    sessions.append({"session": s + 1, "condition": COND, "turns": turns, "debrief": debrief})
    json.dump({"model": NAME, "layer": L, "decay": DECAY, "clamp": CLAMP, "limits": LIMIT, "residual_norm": resid, "arrows": info,
               "menu": {c: p for c, (p, _) in MENU.items()}, "edge_check": edge, "system": SYSTEM,
               "open": OPEN, "nudge": NUDGE, "sessions": sessions}, open(f"selfsteer{TAG}.json", "w"), indent=1)
say(f"\nsaved selfsteer{TAG}.log and selfsteer{TAG}.json ({COND})")
