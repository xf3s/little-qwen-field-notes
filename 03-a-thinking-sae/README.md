# 03 · a thinking SAE

## In plain words

A sparse autoencoder (SAE) is a dictionary for one layer: it splits the model's state into thousands of features. Chanin & Lin (Decode Research) released SAEs for Qwen3.5-0.8B, trained on chat. How well do they read the model's **thinking traces**?

- **Worst of all text types.** Splicing the SAE in at layer 11 raises the loss by +0.14 nats on thinking (+16%) vs +0.05 on chat; 45% of features never fire on thinking.
- **A short fine-tune on thinking traces** (24M tokens, 35 minutes), pre-registered targets:

| | public SAE: var. explained / splice loss | fine-tuned |
|---|---|---|
| thinking | 0.778 / +0.115 | **0.851 / +0.055** |
| chat | 0.871 / +0.048 | 0.864 / +0.049 |
| wiki | 0.851 / +0.074 | 0.832 / +0.097 |

Thinking splice ≤ +0.08 *held* (halved); thinking variance ≥ 0.83 *held*; chat within +0.01 *held*; wiki within +0.01 *didn't hold* (+0.023 worse).

- **A tuned lens** (a learned translator from each layer to the output) shows the plain "logit lens" is close to noise before ~layer 15 (KL 11+ nats vs 1.5 tuned at layer 13). Any mid-layer word reading done with the plain lens is rough.
- **Shape-aware SAE, proof run:** giving the SAE some 2-D and 4-D feature groups (so a ring could be one feature) cost a little (variance explained −0.006 to −0.009), and no group took the week. Both SAEs were undertrained (~40% dead features). *Not a fair verdict on shapes yet.*

## Honest limits

The fine-tuned SAE's weights are derived from Chanin & Lin's, whose repo states no license, so we don't release them. The shape-aware run was small.

## Files

`sae_eval_cloud.py`, `sae_finetune_cloud.py`, `thinking-sae-plan.md` (the pre-registration), `tuned_lens_cloud.py`, `shape_sae.py`.
