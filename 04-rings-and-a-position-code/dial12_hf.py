"""dial12.py for little Qwen on clay's CPU with plain transformers (nnsight leaked ~23 GB locally on 11-06). Same items, templates
and reading position; hidden_states[L+1] = output of layer L, as in dial12.py."""
import sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.argv = ["dial12.py", "Qwen/Qwen3.5-0.8B"]
src = open("/home/three/interp/wsp/dial12.py").read()
exec(src[src.index("DOMAINS = {"):src.index("model = LanguageModel")])
NAME = "Qwen/Qwen3.5-0.8B"
tok = AutoTokenizer.from_pretrained(NAME); model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).eval()
torch.set_grad_enabled(False)
out = {}
for dom, (words, temps) in DOMAINS.items():
    X = []
    for w in words:
        acc = []
        for t in temps:
            text = t.format(w); end = t.index("{}") + len(w)
            enc = tok(text, return_offsets_mapping=True, return_tensors="pt")
            idx = max(k for k, (a, b) in enumerate(enc["offset_mapping"][0].tolist()) if a < end and b > a)
            hs = model(input_ids=enc["input_ids"], output_hidden_states=True).hidden_states[1:]
            acc.append(torch.stack([h[0, idx] for h in hs]))
        X.append(torch.stack(acc).mean(0))
    out[dom] = torch.stack(X); print(dom, flush=True)
torch.save(out, "/home/three/interp/wsp/dial12_Qwen3.5-0.8B.pt"); print("saved")
