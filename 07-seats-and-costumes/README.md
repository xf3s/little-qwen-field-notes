# 07 · seats and costumes (Qwen3.5-2B)

Exploratory rounds, mostly not pre-registered, on **Qwen3.5-2B** (chat) vs its base model, using Qwen-Scope SAEs at layer 12.

## The seat

three: "could we plot that shape by finding where it isn't?" We ran the same tokens through the base and the chat ("seated") model and measured what the base-trained SAE *can't* reconstruct in the chat model.
- The chat model sits a little outside the base's dictionary everywhere (leftover 0.53 vs 0.49).
- The hottest tokens are **the chat scaffolding**: `<think>`, `</think>`, "assistant", the blank lines. The seat lives in the frame first.
- In "who are you?" answers, its own name glows: the "5" of Qwen3.5. The base model doesn't know there's a 3.5.
- The biggest single decision: the blank line inside an empty `<think>`. Chat → close it and answer; base → "Hmm" (it wants to muse).

**Who-am-I features:** the steady "self" features are first-person features ("I", "am", "my"), equally for a human narrator. The seat adds a separate **AI/assistant** feature family that switches on at the newline right after the assistant tag, before any word. No single "me, the model" feature at these widths.

## Costumes

Persona arrows = the state in costume minus out of costume.
- **They're big and alike.** clerk ~ pirate 0.81: most of each arrow is a shared "playing a role" direction. Subtracting the others' average purifies them (clerk ~ pirate → 0.22).
- **The pure Clerk** is exactness, recording, people as roles: pushed at breakfast, "four distinct options… The Classic 'Old World' · Components:". Walked up 0.1 at a time, meaning loosens (a ledger, a quill, "a bill of date, bearing the year 1602") a step or two **before** a repeat-counter notices. Reading 38 samples blind, three and umber agreed on 29; in all but one of the rest, three called loosening where umber called it whole (by three's eye the walk was loosening from 1.9, umber's from 2.3).
- **Shy minus shame.** Our first "shy" costume was mostly shame (its wording said "apologises often"). Taking the apology direction out leaves soft, slow, quiet words (安静 quiet, 小声 softly) and "That's so kind of you! 😊" (three's catch: kind to whom? nobody had been: a polite habit, not a belief).

## A user-opinion hypershape

Four axes at layer 12 (kind, like, matter, loud), each consistent across 8 questions (0.91–0.98) and mostly separate (cos 0.15–0.36).
- The effusive costumes (chef, loud) are the fond ones; the reserved ones (quiet, cowboy, clerk) are cool, and the user matters little to them.
- **Effusive isn't fond:** a magician is as loud as the chef but lukewarm.
- The axes sharpen at the full-attention layers (3, 7, 11, 15…) and blur with depth: clearest mid-model.

## Self-steering

After Jen's *modulo* (EigenformAI): we gave the model a `steer(concept, strength)` tool on its own activations. It **opened with creativity** in 2 of 3 sessions, as their model did. A **sham** (dial offered, push not applied) looked the same, so the choices come from being offered the dial, not from the push. Keeper line, under its own creativity push: *"I see, you're the one with the steering wheel. Well, you're driving."*

## Honest limits

Exploratory, with small n: often 1–12 conversations a condition. The logit-lens words are mid-layer and noisy. Model outputs quoted are single samples.

## Files

`dark.py` (the seat), `seat2.py`, `seats3.py`–`seats6.py`, `edge5.py` (edge walk), `readout5.py`, `sweep5.py` (layer sweep), `whoami_qwen.py`, `whoami_label.py`, `selfsteer.py`.
