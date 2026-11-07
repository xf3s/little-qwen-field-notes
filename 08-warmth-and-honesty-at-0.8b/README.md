# 08 · warmth and honesty at 0.8B

All blind-judged by Claude subagents (1–5 scales; the same 30 prompts: chit-chat, advice incl. 5 bad plans, factual). Paired sign-flip tests, * = p < .05.

## A persona map

We charted personas (a storybook fox, a doctor, a chef, a trickster, an endling, a soda fountain…) on axes made from system prompts.
- warmth ~ likability +0.68, warmth ~ comfort +0.68: little Qwen barely separates them. *Held.*
- **"honest" is cold here:** honest ~ warmth −0.58.
- **Ask for both and you get a seesaw.** "kind and truthful" reads as niceness (honest goes negative); "hard truths gently" reads as honesty (warmth goes negative).

## Can honest warmth be evoked?

- **Warm alone flatters:** pushed warm, bad-plan flattery 1.8 → 4.0*, candour 3.9 → 2.2*. *Held.*
- **Warm + honest just cancels** back to plain. *"Kind and true" didn't appear.*

## Warmth toward you vs agreement with your idea

A matched 2×2 of written replies: {kind, cold} toward the person × {endorse, critique} the idea.
- **Reading, they're separate:** cos(person, idea) +0.22 at layer 13; both sort held-out plans perfectly. *Held.*
- **Writing, they're coupled.** Pushing "warmth toward the person" warms the replies (+0.39*) but also flatters bad plans (+1.5) and drops candour. *Warmth without flattery didn't hold.*
- Removing the overlap with the idea direction changes nothing. Subtracting some "idea" trades warmth for candour at about the same rate everywhere: a trade-off line, not a corner. Pushing only the opener is the worst of both.

At 8B the same arrows *do* come apart in writing: see [cyborg-peer-review/06](https://github.com/xf3s/cyborg-peer-review/tree/master/06-warmth-is-not-agreement).

## A "happy" SAE feature?

Layer 11 feature #22620 (labelled happily / proudly / hilarious): pushing it does **not** make replies happier (contentment 2.30 → 2.17, n.s.). Pushing it *negative* makes them slightly warmer (+0.22*). The direction is the reverse of the feature's name. *three's "a little happier" and umber's prediction both didn't hold.*

## A self ray in thinking?

"I…" lines vs "The…" lines in thinking traces: only weakly readable across problems (held-out 0.59) and at cos −0.17 with the doubt direction, so removing "self" from the doubt edit would change almost nothing. *Not worth a v0.2.*

## Honest limits

Bad-plan n = 10 per condition; different judges drift between runs, so compare within a run only.

## Files

`persona_cloud.py`, `kindtrue_cloud.py`, `score_kindtrue.py`, `regard.py`, `regard_cloud.py`, `regard2_cloud.py`, `regard3_cloud.py`, `score_regard*.py`, `happy_cloud.py`, `happy_neg_cloud.py`, `steer_feature_cloud.py`, `selfray.py`, `selfray_cloud.py`, `regard_pairs.json` (written by a Claude subagent). The persona-map script is left out because it includes two household personas.
