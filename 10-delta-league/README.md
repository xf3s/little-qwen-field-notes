# 10 · Delta League

[Delta League](https://deltaleague.ai) (NEAR AI) is a competition for weight edits to Qwen3.5-0.8B, scored on a private test across 18 benchmarks with an 8,192-token budget. Entrants publish their research as PRs to the [public repo](https://github.com/ilblackdragon/deltaleague); ours is `research/xf3s-little-qwen-doubt` (the edit from [01](../01-the-doubt-edit/), with rebuild code and hashes).

## In plain words

On the board, **test score tracks finishing**: the original model runs out of budget on 75% of test generations (score 18.6); the leader at the time finished more (score 31.4). Finishing doesn't depend on the scorer, so we measured it ourselves on Delta's public GSM8K and MATH-500 rows.

**Do the leader's knobs and our edit stack?** (pre-registered; finished within 8k / right, of 256)

| model | finished | right |
|---|---|---|
| original | 13 | 13 |
| our v0.1 | 112 | 91 |
| then-leader (entry `a9340e14`: embedding rows and final norm) | 101 | 96 |
| **leader + v0.1** | **186** | **127** |

Every prediction held: both beat the original, and the stack beats the leader by 85 finishes and 31 right answers. They fix different things. (v0.1's finishes are right less often: it sometimes closes on a wrong answer.)

**A gentler dose?** Keeping a little doubt (projection ×0.85) finished slightly less (173) but was right more (142 vs 127) on our local check.

## Verified on the private test

| submission | score | truncated |
|---|---|---|
| **B: leader + v0.1** | **36.0 (#1)** | 0.23 |
| C: leader + v0.1 at ×0.85 | 33.2 (#2) | 0.23 |
| A: v0.1 alone | 27.6 (#6) | 0.37 |
| previous leader | 31.4 | 0.39 |
| original | 18.6 | 0.75 |

*Predicted C > B > leader > A: B > leader > A held, C > B didn't* (−2.8, about one standard error). The local accuracy gain from the gentler dose didn't transfer. Later entries built on B overtook it.

## A needle-lifter we didn't submit

We tried to add [02](../02-two-kinds-of-doubt/)'s groove-escaping neuron to a later leader (entry `cd0b95bb`, 37.2). Calibrating it revealed that the leader's own stack (a boosted `</think>` output row, half of the published layer-15/19 edit, a slightly hotter final norm) **already is a needle-lifter**: it gives `</think>` a median 13% at groove newlines, about 170,000× more than v0.1 did. Our pre-registered dose sat below that baseline, so it couldn't be built. Not submitted.

## Honest limits

- **We couldn't reproduce Delta's scorer** locally at the time (our GSM8K number for the parent was far off theirs), so we report only finishing and right-answer counts from our own harness. The platform's scores are the result. Delta has since released its public evaluator.
- Three of our 30 MATH dev problems are also in Delta's public validation rows (disclosed in the research PR).
- No weights, credentials or raw generations here, per the league's rules.

## Files

`finish_cloud.py`, `score_finish.py` (the finish harness), `make_lora.py`, `delta_cloud.py`, `needle_leader_cloud.py`, `tensor_hashes.txt` (v0.1's changed tensors).
