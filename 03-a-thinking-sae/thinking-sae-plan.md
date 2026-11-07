# A thinking SAE for Qwen3.5-0.8B: plan (11-03, three: "yeah!!")

Goal: fine-tune the public layer-11 SAE (decoderesearch/qwen-3.5-saes, Chanin & Lin, Matryoshka BatchTopK, k=100,
65,536 latents, prefixes 2k/16k) on Qwen's own thinking traces, so it reads the thinking dialect as well as it reads chat,
without getting worse at chat.

## Why layer 11 first
It's the SAEs' weakest spot on thinking (variance explained 0.78; splice loss +0.14 nats, +16%, vs +0.05 on chat; sae_eval.json),
and it sits just before our "but" direction (L13). L5 and L17 after, if L11 works.

## Data
- Qwen's own transcripts (prompt + thinking + answer, in its chat template): MATH dev runs (all edits and the unedited ones),
  GSM8K test and train runs, AIME. **Never the sealed MATH half.** About 15M tokens now; the 4 GSM8K-train slices add more.
- Split **by run** (not by token): 90% train, 10% eval, so eval transcripts are unseen.
- Regression sets, unchanged from sae_eval: UltraChat (chat) and wikitext (wiki).

## Method
- Start from the published weights (W_enc, b_enc, W_dec, b_dec, threshold). Same architecture and k.
- Activations at `model.layers.11` output, first token and special tokens excluded (as sae_eval did).
- Loss: Matryoshka reconstruction (each prefix 2k / 16k / 65k reconstructs on its own) + the usual aux loss for dead
  latents; decoder rows kept unit norm. Mix in ~20% chat tokens so it doesn't forget its home dialect.
- Learning rate well below the original's 1e-4 (try 2e-5), a few passes at most. A Modal L4 or A10G.
- Budget: under 1 GPU-hour all in (activations ~10 min per 15M tokens; the SAE steps are minutes).

## Pre-registered success (written before training)
On the held-out thinking runs at L11: splice loss increase ≤ +0.08 nats (from +0.14), variance explained ≥ 0.83 (from 0.78).
On chat and wiki: no worse than +0.01 nats beyond today's numbers (+0.05 chat, +0.08 wiki at L11).
Then the interp check: do the stuck / noodle moments (groove_v2.pt) separate onto cleaner, fewer latents than with the public SAE?

## Before anything is published
The public repo states no license. Fine-tuned weights are derived from Chanin & Lin's; ask them (or check the
licence) before releasing the weights. Results and the method can be written up either way, with credit.
