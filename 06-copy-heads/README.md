# 06 · copy heads

## In plain words

Is there an attention head that stops little Qwen from repeating itself, like the "copy suppression" heads found in GPT-2?

- **No strong one.** Across 336 heads on 40 Wikipedia paragraphs, the most suppressing late head pushes repeats down by only 0.009 logits. *"Clearly negative" didn't hold.* The head we'd nicknamed **Brake Neck Bear** (layer 19, head 4, a weak Wait-suppressor from [01](../01-the-doubt-edit/)) ranks 3rd, so "top three" held, but it's tiny.
- **The late heads do the opposite, 5× stronger: copy-*boosting*.** Layer 23 heads 2 and 6 and layer 21 head 15 push already-seen words up. Little Qwen copies with attention and brakes with neurons.
- **Do the boosters drive the repetition grooves?** No: they push as hard in grooves as in ordinary copying (0.65–1.14×; predicted ≥ 1.5×). *Didn't hold.* A groove is ordinary copying that nothing stops, not extra copying.

Doubling Brake Neck Bear did cut the Waits (38 → 17 per run), but the right answers stayed at 0 and the doubt moved to "If", "Maybe", "No". Lesson: "Wait" is the symptom's word, not the doubt.

## Honest limits

Direct effects only; 40 paragraphs; only 2 of 6 looping runs had a clean single-line groove.

## Files

`copysup.py`, `groove_heads.py`.
