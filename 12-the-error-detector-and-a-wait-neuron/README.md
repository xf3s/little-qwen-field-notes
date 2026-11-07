# 12 · the error detector and a Wait neuron

three: "look for something that fires when little qwen reads its own contradiction, then says wait, like our neuron."

## In plain words

We planted off-by-one errors in little Qwen's own correct arithmetic lines ("a + b = c" → "c + 1") and looked for where the model registers it.

**First, controls.** Our first detector mostly learned "the numbers in this line are bigger". A second control just moved that confound. The fix was a 2×2 per line: two right versions (a, c) and (a+1, c+1), two wrong ones (a, c+1) and (a+1, c). Correctness is then the XOR, so any plain "bigger number" code cancels.

**With that, it's there.** Read at the newline after the line, the wrong-vs-right direction sorts quads at chance through layer 4 and at 1.00 (66/66) at the last layer, in both the original model and the Delta leader. *Held.* Through the vocabulary, the last layer leans toward **"Wait", "?", "???"**. But in sampled text the model barely reacts (1.1×): the detector exists, and its Wait is a whisper.

**Read where the error is written** (at the result number itself):
- The model's own surprise at the written number separates wrong from right almost perfectly (AUC 0.99–1.00). *Held.*
- The residual stream shows it only from **layer 23** (0.99): layer 22 computes it. Per line, at a point a neuron can read.

## A Wait neuron

We rewired the leader's least-used layer-23 neuron to read that signal and write "Wait", fit on 9 problems and tested on 3 unseen ones:

| at the dose ×0.5 | chance the next word is "Wait" |
|---|---|
| right after a **wrong** result | median **72%** |
| right after a correct result | median 0% (mean 2%: a few false alarms) |
| every other token | 0.006% |

*Predicted ≥ 20% on wrong results and ≤ 1% elsewhere: held by median; the mean on correct results is the limit.*

**Does it rescue answers? Open.** The test was pre-registered, but the cloud run stopped before any result.

## Honest limits

Only off-by-one sums and differences planted in its own correct lines. A wrong number being a *surprise* may not be the same as a concept of contradiction.

## Files

`errors_cloud.py`, `errors2_cloud.py`, `errors3_cloud.py` (XOR quads), `errors4_cloud.py` (at the result), `waitneuron_cloud.py`, `waitgen_cloud.py` (the unfinished rescue test).
