# 09 · TwIL "think shorter"

webAI's **TwIL-8B** (a Granite 4.2 8B fine-tune) comes in two variants that differ only by a final length-aware RL stage: *verbose* → *balanced*. What did "think shorter" change?

## The weights (pre-registered)

- **Tiny:** median relative change per matrix 0.009%; embeddings exactly frozen. *"Small" held (far smaller than guessed).*
- **Writers ~10× readers:** the matrices that write into the residual stream (attention output, MLP down) moved ten times more than those that read from it. *Held.*
- **Not low-rank, not late:** the top 8 directions hold ~3% of the change; it's flat across depth. *Both didn't hold.*
- **The most-moved output words:** Given, But, Thus, Answer, `)=`, `</think>`, respond, Obviously, If… (discourse words that steer a trace). "Wait" isn't in the top 40.

## Side by side (32 MATH-500 problems × 2)

| | right | finished | median length |
|---|---|---|---|
| verbose | 52/64 | 53/64 | 2161 tokens |
| balanced | **58/64** | **60/64** | **1571** |

27% shorter *(held)*, finishes more *(held)*. Accuracy differs by 6, beyond our ±4 *(didn't hold: balanced is better, through finishing)*. "But" per 1k tokens is the same *(fewer Buts didn't hold)*; "Thus" rises (3.26 vs 2.59).

**Reading:** the length RL didn't make it doubt less often. It made "Thus" land as an ending. Same Buts, more Thuses that close.

## Files

`diff_variants.py`, `twil_cloud.py`, `score_twil.py`. TwIL-8B's license is non-commercial research.
