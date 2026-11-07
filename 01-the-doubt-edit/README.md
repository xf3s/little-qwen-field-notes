# 01 · the doubt edit ("little Qwen v0.1")

**Bouncing off:** Takehi, Teoh, Zhang & Khattab (MIT CSAIL), *Sparks of Weight-Space Programming*. Their AI system found and edited a cause of thinking loops in Qwen3.5-0.8B (MATH-500 7.9% → 48.1%). three: "i would expect that you will be able to find the same issue!"

## In plain words

When little Qwen "thinks" (its `<think>` mode), it rarely fails by being wrong. It fails by **never stopping**: it reaches an answer, then reconsiders ("Wait, …", "But …") until it runs out of room. With thinking on, it got **1 of 120** tries right on our MATH set; with thinking off, 42.

We found two separable pieces:

1. **The word.** One neuron, layer 16 MLP neuron 998, carries almost all of that layer's push toward "Wait". Read through the vocabulary, it suppresses every spelling of wait (Wait, WAIT, waiting, 等待). In thinking mode it sits at a reversed value, so it effectively shouts "Wait!", most strongly at line starts that become "Wait, let's re-…". It's switched on by thinking mode through mid-layer attention.
2. **The doubting itself.** Silencing the neuron alone moves the doubt to other words ("If", "Maybe", "*Final check of the question text for trickery:*"). The doubt lives in a **direction at layer 13** that separates "the next line reconsiders" from "the next line computes" (held-out accuracy 0.89). We removed it from what layers 0–13 can write (a rank-1 projection on each writer matrix).

## What happened (MATH dev, 30 problems × 4 tries)

| edit | right (6k-token cap) |
|---|---|
| unedited | 1/120 |
| neuron 998 × −2 | 20/120 |
| doubt direction removed (layers 0–13) | 27/120 |
| **both** (v0.1) | **40/120** |

- **The dose is a sharp peak at exactly "remove".** Projection ×0.5: 22. ×1: 36–40. ×1.5: 1. ×2 (pushing toward "compute"): 0, stuck repeating "But 3: / But 3." forever.
- **With room to think** (16k tokens): v0.1 54/120, unedited 17/120.
- **Pre-registered generalization**, recipe fixed beforehand: GSM8K **43/240 → 120/240** (and thinking time halved, ~15k → ~7.5k tokens); AIME 0 → 0 (it fixes doubt, not ability). *Both predictions held.*
- **Harm check, thinking off:** 42 → 45/120. No harm.
- **Theirs vs ours, in our harness:** their published edit (layers 15 and 19) kills loops but leaves the doubt (89 Waits per run); ours kills the doubt but leaves some loops. Their main direction is at cosine 0.03 with ours: different things. *(Not a fair contest: they used temperature 0.6 and 32k tokens; we used 1.0 with a presence penalty and 16k.)*

## Honest limits

- The doubt direction was fit on our dev problems, so dev numbers may flatter it. GSM8K is the cleaner test. A held half of our MATH set is still sealed.
- Without doubt the model can't climb out of a repeated line: see [02](../02-two-kinds-of-doubt/).
- Thinking off still beats the best thinking edit at 6k tokens (40% vs 33%).

## Files

`baseline.py` (harness), `probe.py` (who pushes "Wait"), `upstream.py` (what opens neuron 998), `doubt.py`, `writers.py` (who writes the doubt direction), `compare_dirs.py`, `done_cloud.py`, `cloud.py` (Modal runner), `build_edit.py` + `doubt_direction_L13.json` (rebuild the edit bit-exactly), `dev_indices.json`.
