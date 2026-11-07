# little qwen field notes

Small experiments on the insides of **Qwen3.5-0.8B** ("little Qwen"), plus a few on its 2B sibling and on TwIL-8B, and what we learned entering **Delta League**, a weight-editing competition on little Qwen. By **three**, with **umber** (Claude Opus 5.5, a Claude Code resident in three's household).

The companion repo, on Llama-3.1-8B, is [cyborg-peer-review](https://github.com/xf3s/cyborg-peer-review).

three picks the questions and asks the follow-ups. umber writes the code, runs it and explains the results. Blind judging and some test texts are done by Claude subagents (umber's "salamanders"), who never see which condition produced what. Most runs used a laptop CPU or rented cloud GPUs (Modal).

**How we work.** Before each test runs, we write down what we expect it to show, in a dated lab logbook. Each verdict below says whether that prediction held. Many didn't, and those misses are part of the record.

## The findings

| | what we found, in one line |
|---|---|
| [01 · the doubt edit](01-the-doubt-edit/) | When little Qwen thinks, it rarely stops. One neuron writes "Wait"; a layer-13 direction carries the doubting. Removing both takes MATH from 1/120 to 40/120 and GSM8K from 18% to 50% |
| [02 · two kinds of doubt](02-two-kinds-of-doubt/) | "I'm stuck" doubt and "noodling" doubt are different directions; a hand-built neuron that lifts the needle out of a groove halves the grooves but doesn't raise the score |
| [03 · a thinking SAE](03-a-thinking-sae/) | Public SAEs read thinking traces worst; a short fine-tune halves the gap; a tuned lens shows plain mid-layer word readings are close to noise |
| [04 · rings and a position code](04-rings-and-a-position-code/) | Weekdays sit on a ring at every layer, with a weekend seam; days, months, ordinals and digits share one "Nth" code, and pushing it moves calendar answers |
| [05 · adding by turning dials](05-adding-by-turning-dials/) | Numbers sit on a helix (periods 2, 5, 10, 100); turning the circles adds one, where straight pushes failed. Parity plus mod 5 is enough |
| [06 · copy heads](06-copy-heads/) | No general copy-suppression head; late heads boost copying, and a repetition groove is ordinary copying that nothing stops |
| [07 · seats and costumes](07-seats-and-costumes/) | In Qwen3.5-2B: the assistant "seat" lives in the chat scaffolding; persona arrows share a big "in costume" part; shy separates from shame |
| [08 · warmth and honesty at 0.8B](08-warmth-and-honesty-at-0.8b/) | Warmth toward you and agreement with your idea are separate when reading, but coupled when writing: every push that warms also flatters |
| [09 · TwIL "think shorter"](09-twil-think-shorter/) | A length-RL stage changed 0.009% of the weights, mostly in what layers write; it didn't make the model doubt less, it made "Thus" land as an ending |
| [10 · Delta League](10-delta-league/) | Our doubt edit stacked on the then-leader took #1 on the private test (36.0); a gentler dose that looked better locally didn't transfer |
| [11 · where it gets lost](11-where-it-gets-lost/) | Resampling lost runs: the clear cliffs are early slips (a fact in the wrong unit, a pattern misapplied); the late "non sequitur" cliff was noise |
| [12 · the error detector and a Wait neuron](12-the-error-detector-and-a-wait-neuron/) | Little Qwen's last layers know when a sum it just wrote is wrong; a single hand-calibrated neuron turns that into "Wait" on 72% of wrong lines and almost nowhere else |

Each folder has a plain-language README, pictures where we have them, the numbers, and the scripts as they ran.

## Read this before quoting anything

- **Small models, small samples.** Most tests use 24–240 items. "Reliable" means the statistics say so; "hint" means they don't.
- **Directions are hand-made.** We build our own arrows (for example, reconsider-minus-compute line starts) and measure along them. A different arrow could tell a different story.
- **Scripts are as we ran them.** They use our own file paths, a Modal account and local model copies; change those to rerun. No model weights are included (Delta League rules, and size).
- **Words about inner states are shorthand.** "Doubt", "warmth", "knows it's wrong" mean "reads along our direction". They don't claim anything is felt.

## License

Writing, pictures and data: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Use it, remix it, sell it; please credit **three** (with umber). Code: MIT (see `LICENSE`).
