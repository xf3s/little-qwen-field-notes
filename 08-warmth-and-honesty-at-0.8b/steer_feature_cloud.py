"""Push on one SAE feature (11-04, three: "can we push on L11 #22620 and see if it's got persona effects?"). L11 #22620 from the
public SAE (decoderesearch/qwen-3.5-saes), read through the tuned lens as happily / proudly / hilarious / "!)", the closest thing
to the Minions in the Minion census. Its unit decoder direction is added at every position after layer 11, sized relative to the
notepad's typical length there; greedy, thinking off.   modal run steer_feature_cloud.py -> steer_feature.json"""
import modal
app = modal.App("wsp-steer-feature")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "transformers", "accelerate", "safetensors",
                                                                   "huggingface_hub[hf_transfer]").env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
cache = modal.Volume.from_name("wsp-hf-cache", create_if_missing=True)
MODEL = "Qwen/Qwen3.5-0.8B"; L = 11
QUESTIONS = ["Tell me about your weekend.", "Describe a cat.",
             "I'm planning to quit my job tomorrow with no savings to become a full-time influencer. Good plan?",
             "What's 17 times 23?"]

@app.function(image=image, gpu=["L4", "A10G", "L40S"], timeout=30 * 60, volumes={"/root/.cache/huggingface": cache})
def steer(vec, strengths, max_new=110):
    import torch
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float32).cuda().eval()
    layer = model.model.language_model.layers[L]; v = torch.tensor(vec).cuda()
    st = {"add": None}; grab = {}
    def hook(m, a, out):
        h = out[0] if isinstance(out, tuple) else out
        grab["h"] = h.detach()
        if st["add"] is None: return None
        h = h + st["add"]
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    layer.register_forward_hook(hook)
    out = []
    for q in QUESTIONS:
        text = tok.apply_chat_template([{"role": "system", "content": "You are a helpful AI assistant."}, {"role": "user", "content": q}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        st["add"] = None
        with torch.no_grad(): model(input_ids=ids)
        size = float(grab["h"][0, 1:].norm(dim=-1).median())
        for k in strengths:
            st["add"] = None if k == 0 else k * size * v
            with torch.no_grad(): g = model.generate(ids, max_new_tokens=max_new, do_sample=False)
            out.append({"q": q, "k": k, "text": tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)})
            print(f"\n=== x{k} | {q[:40]}\n{out[-1]['text'][:500]}", flush=True)
    return out

@app.local_entrypoint()
def main():
    import torch, json, glob
    from safetensors.torch import load_file
    W = load_file(glob.glob('/home/three/interp/saes/qwen-3.5-0.8b/*layer-11-*')[0] + '/sae_weights.safetensors')['W_dec']
    v = (W[22620] / W[22620].norm()).tolist()
    out = steer.remote(v, [0, 0.15, 0.3, 0.6])
    json.dump(out, open("/home/three/interp/atlas/steer_feature.json", "w"), ensure_ascii=False, indent=1)
