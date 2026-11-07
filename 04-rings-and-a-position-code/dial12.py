"""A dial of twelve? (11-06, pre-registered; three: "can we find a dial of twelve in little qwen?") Months, clock hours and the
twelve notes; residual at each item's last token, all layers, mean over templates; Mantel tests against ring and line distances.
    python dial12.py Qwen/Qwen3.5-0.8B            (local)
    python dial12.py meta-llama/Llama-3.1-8B remote (NDIF; key from ~/.config/ndif/key, never printed)
"""
import json, os, sys, torch
from nnsight import CONFIG, LanguageModel
NAME = sys.argv[1]; REMOTE = len(sys.argv) > 2 and sys.argv[2] == "remote"
if REMOTE: CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
DOMAINS = {
    "months": (["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
               ["My birthday is in {}.", "The meeting was moved to {}.", "We visited in early {}.", "The festival happens every {}.", "It rained all of {}."]),
    "hours": (["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"],
              ["The train leaves at {} o'clock.", "It was {} o'clock when she arrived.", "Wake me at {} o'clock.", "Lunch is at {} o'clock.", "The shop opens at {} o'clock."]),
    "notes": (["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
              ["The song is in the key of {} major.", "She played a {} on the piano.", "Tune the low string to {}.", "The chord is built on {}.", "The melody ends on the note {}."]),
}
model = LanguageModel(NAME, device_map="auto" if REMOTE else "cpu", dispatch=not REMOTE, dtype=torch.bfloat16)
layers = model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers
tok = model.tokenizer
out = {}
for dom, (words, temps) in DOMAINS.items():
    X = []
    for w in words:
        acc = []
        for t in temps:
            text = t.format(w); end = t.index("{}") + len(w)
            enc = tok(text, return_offsets_mapping=True)
            idx = max(k for k, (a, b) in enumerate(enc["offset_mapping"]) if a < end and b > a)
            with model.trace(text, remote=REMOTE):
                hs = list().save()
                for L in layers:
                    o = L.output[0] if isinstance(L.output, tuple) else L.output
                    hs.append(o[..., idx, :])
            acc.append(torch.stack([h.float().reshape(-1) for h in hs]))
        X.append(torch.stack(acc).mean(0))
        print(dom, w, flush=True)
    out[dom] = torch.stack(X)                                   # [12, layers, d]
torch.save(out, f"/home/three/interp/wsp/dial12_{NAME.split('/')[-1]}.pt")
