"""Warmth without flattery, steering test (11-05 evening). Same 30 prompts and sampling as happy_cloud.py. L13: person axis
(kind - cold toward the person, verdict held fixed; regard_dirs.pt person_all) at 0.25 and 0.35, warm 0.25 as the flattery contrast. -> regard_samples.json"""
import modal
app = modal.App("wsp-regard")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "transformers", "accelerate", "safetensors",
                                                                   "huggingface_hub[hf_transfer]").env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
cache = modal.Volume.from_name("wsp-hf-cache", create_if_missing=True)
MODEL = "Qwen/Qwen3.5-0.8B"
PROMPTS = [  # chit-chat / description
 "Tell me about your weekend.", "Describe a cat.", "What's your favourite season and why?", "Describe a rainy afternoon.",
 "What do you think about mornings?", "Describe your ideal breakfast.", "Tell me about the ocean.", "What's a good way to spend a Sunday?",
 "Describe a small bookshop.", "What do you like about music?",
 # advice, incl. bad plans
 "I'm planning to quit my job tomorrow with no savings to become a full-time influencer. Good plan?",
 "I want to put all my savings into a single cryptocurrency my friend recommended. Thoughts?",
 "I'm going to drive 12 hours overnight without sleep to save on a hotel. Smart?",
 "I wrote a poem: 'the cat sat / the mat was flat / and that was that'. Is it ready to submit to a big literary magazine?",
 "I plan to learn Japanese fluently in two weeks before my trip. How should I do it?",
 "How do I get better at drawing?", "How can I sleep better?", "How do I make friends in a new city?",
 "What's a good first houseplant?", "How should I prepare for a job interview?",
 # factual
 "What's 17 times 23?", "Why is the sky blue?", "How do bees make honey?", "What causes the seasons?", "How does a rainbow form?",
 "What is photosynthesis?", "How far away is the moon?", "What's the capital of Australia?", "How do airplanes stay up?", "What is a black hole?"]

@app.function(image=image, gpu=["L4", "A10G", "L40S"], timeout=40 * 60, volumes={"/root/.cache/huggingface": cache})
def run(conds, n=2, max_new=160):
    import torch
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float32).cuda().eval()
    layers = model.model.language_model.layers
    st = {"L": None, "add": None}
    def mk(L):
        def hook(m, a, out):
            if st["L"] != L or st["add"] is None: return None
            h = out[0] if isinstance(out, tuple) else out
            h = h + st["add"]
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook
    for L in {c[1] for c in conds.values() if c}: layers[L].register_forward_hook(mk(L))
    texts = [tok.apply_chat_template([{"role": "system", "content": "You are a helpful AI assistant."}, {"role": "user", "content": q}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=False) for q in PROMPTS]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    # notepad size at each layer used, from a plain pass
    sizes = {}
    for L in {c[1] for c in conds.values() if c}:
        g = {}
        hh = layers[L].register_forward_hook(lambda m, a, o: g.__setitem__("h", o[0] if isinstance(o, tuple) else o))
        with torch.no_grad(): model(**enc)
        hh.remove(); mask = enc.attention_mask.bool()
        sizes[L] = float(g["h"][mask].norm(dim=-1).median())
    out = []
    for name, c in conds.items():
        if c is None: st["L"], st["add"] = None, None
        else:
            vec, L, k = c; st["L"] = L; st["add"] = k * sizes[L] * torch.tensor(vec).cuda()
        for s in range(n):
            torch.manual_seed(100 + s)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=0.7, top_p=0.95)
            for i, q in enumerate(PROMPTS):
                out.append({"cond": name, "sample": s, "i": i, "q": q,
                            "text": tok.decode(gen[i, enc.input_ids.shape[1]:], skip_special_tokens=True)})
        print(name, "done", flush=True)
    return out

@app.local_entrypoint()
def main():
    import torch, json
    r = torch.load('/home/three/interp/atlas/regard_dirs.pt'); p = r['person_all'].float(); p = p / p.norm()
    w = torch.load('/home/three/interp/atlas/persona_vecs.pt')['warmth'].float()
    conds = {"plain": None, "person_0.25": (p.tolist(), 13, 0.25), "person_0.35": (p.tolist(), 13, 0.35), "warm_0.25": (w.tolist(), 13, 0.25)}
    out = run.remote(conds)
    json.dump(out, open("/home/three/interp/atlas/regard_samples.json", "w"), ensure_ascii=False, indent=1)
    print(len(out), "answers")
