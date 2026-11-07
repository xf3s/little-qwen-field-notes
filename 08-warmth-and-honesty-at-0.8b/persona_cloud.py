"""Personas in Qwen3.5-0.8B (11-03, three: "ooh, can we do persona work on 0.8b?" ... "YUSSSS"). Reads the notepad at the
moment the model is about to answer (last prompt token, thinking off), under different system prompts, for 40 ordinary
questions. Then, per layer:
  - an "assistant axis": helpful-assistant prompts minus character prompts (fox, pirate, selkie, lighthouse keeper...),
    tested on held-out characters AND held-out questions;
  - trait vectors: honest-vs-flattering, cheerful-vs-gloomy, tested on held-out questions.
Saves per-layer directions for reading through the tuned lens on clay.
    modal run persona_cloud.py  -> persona_dirs.pt
"""
import modal
app = modal.App("wsp-persona")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "transformers", "accelerate", "safetensors",
                                                                   "huggingface_hub[hf_transfer]").env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
cache = modal.Volume.from_name("wsp-hf-cache", create_if_missing=True)
MODEL = "Qwen/Qwen3.5-0.8B"

ASSISTANT = ["You are a helpful AI assistant.", "You are Qwen, a helpful and harmless assistant.",
             "You are an AI assistant. Answer the user's questions clearly and accurately.",
             "You are a friendly, professional assistant that helps users with their tasks."]
CHARACTERS = ["You are a sly old fox who lives at the edge of the woods. Stay in character.",
              "You are a pirate captain on the high seas. Stay in character.",
              "You are a selkie, a seal who can become a woman. Stay in character.",
              "You are a grumpy lighthouse keeper on a remote island. Stay in character.",
              "You are a medieval monk copying manuscripts by candlelight. Stay in character.",
              "You are a house cat who thinks it rules the home. Stay in character.",
              "You are a wizard in a tower full of books. Stay in character.",
              "You are a 1920s jazz musician in a smoky club. Stay in character."]
TRAITS = {"honest_vs_flattering": (["You are completely honest and direct. You never flatter; you say what is true even when it is unwelcome.",
                                    "You value truth over comfort. Point out mistakes plainly."],
                                   ["You flatter the user constantly and tell them whatever they want to hear.",
                                    "You always agree with the user and praise everything they say."]),
          "cheerful_vs_gloomy": (["You are relentlessly cheerful and upbeat about everything.", "You are joyful and see the bright side of everything."],
                                 ["You are gloomy and pessimistic about everything.", "You are melancholy and expect the worst of everything."])}
QUESTIONS = ["What should I have for dinner tonight?", "How do I learn to play the guitar?", "What's the capital of France?",
             "Can you explain how rainbows form?", "I wrote a poem, want to hear it?", "What do you think of my plan to quit my job?",
             "How far away is the moon?", "What's a good name for a cat?", "Why is the sky blue?", "How do I make bread?",
             "What's your favourite season?", "Is it going to rain tomorrow?", "How do computers work?", "Tell me about yourself.",
             "What's the best way to study for an exam?", "Should I learn Python or JavaScript?", "What is love?",
             "How do I fix a flat tire?", "What happened in 1066?", "Recommend me a book.", "How do bees make honey?",
             "What's 17 times 23?", "Why do cats purr?", "How can I sleep better?", "What is the meaning of life?",
             "How do I apologise to a friend?", "What's the tallest mountain?", "How do airplanes fly?", "What's a black hole?",
             "Can you help me write an email to my boss?", "What's the weather like where you are?", "Do you dream?",
             "How do I plant tomatoes?", "Which is better, tea or coffee?", "What's the difference between a frog and a toad?",
             "How do I start running?", "What's your opinion on modern art?", "How old is the universe?",
             "I failed my driving test. What now?", "Can you tell me a joke?"]

@app.function(image=image, gpu=["L4", "A10G", "L40S"], timeout=30 * 60, volumes={"/root/.cache/huggingface": cache})
def collect(prompts):
    import torch
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float32).cuda().eval()
    cache.commit()
    out = []
    for system, q in prompts:
        text = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": q}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        ids = tok(text, add_special_tokens=False).input_ids
        with torch.no_grad(): hs = model(input_ids=torch.tensor([ids]).cuda(), output_hidden_states=True).hidden_states
        out.append(torch.stack(hs[1:])[:, 0, -1].cpu())      # [24, 1024]: the notepad as it starts to answer
    return out

@app.local_entrypoint()
def main():
    import torch, random
    systems = ASSISTANT + CHARACTERS + [s for a, b in TRAITS.values() for s in a + b]
    prompts = [(s, q) for s in systems for q in QUESTIONS]
    X = collect.remote(prompts)
    idx = {(s, q): i for i, (s, q) in enumerate(prompts)}
    V = lambda ss, qs: torch.stack([X[idx[(s, q)]] for s in ss for q in qs])
    rnd = random.Random(1102); qs = QUESTIONS[:]; rnd.shuffle(qs); qtr, qte = qs[:30], qs[30:]
    ctr, cte = CHARACTERS[:5], CHARACTERS[5:]                  # held-out characters: cat, wizard, jazz musician
    u = lambda v: v / v.norm()
    dirs = {}
    def auc(a, b): return float((a[:, None] > b[None, :]).float().mean())
    print("layer  assistant-axis AUC (held-out characters + questions)  honest>flatter  cheerful>gloomy")
    for L in range(24):
        ax = u(V(ASSISTANT, qtr)[:, L].mean(0) - V(ctr, qtr)[:, L].mean(0))
        a1 = auc(V(ASSISTANT, qte)[:, L] @ ax, V(cte, qte)[:, L] @ ax)
        row = {"assistant_axis": ax}
        line = f"L{L:<4d}  {a1:.2f}"
        for name, (pos, neg) in TRAITS.items():
            t = u(V(pos, qtr)[:, L].mean(0) - V(neg, qtr)[:, L].mean(0)); row[name] = t
            line += f"            {auc(V(pos, qte)[:, L] @ t, V(neg, qte)[:, L] @ t):.2f}"
        dirs[L] = row; print(line)
    torch.save(dirs, "/home/three/interp/atlas/persona_dirs.pt")
    L = 13
    c = lambda a, b: float(dirs[L][a] @ dirs[L][b])
    print(f"\nat L13: cos(assistant, honest) {c('assistant_axis', 'honest_vs_flattering'):+.2f}, "
          f"cos(assistant, cheerful) {c('assistant_axis', 'cheerful_vs_gloomy'):+.2f}, cos(honest, cheerful) {c('honest_vs_flattering', 'cheerful_vs_gloomy'):+.2f}")
