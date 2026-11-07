# 11 · where it gets lost

On the 37.2 Delta leader (entry `cd0b95bb`), with the platform's sampling.

## In plain words

Take runs that lost a problem the model sometimes gets right. Cut each one at 0%, 10%, … 90% of its thinking, let the model continue many times from each cut, and watch the chance of a right answer. Where it falls off a cliff is where the run got lost.

- **Two kinds of lost.** Long runs are doomed by 10–20% of the trace (a wrong road taken early, then walked to the cap). Short wrong runs hover at 20–50% right to the end: the ending is a coin flip, not a moment.
- **Predictions** (most lost runs have a cliff; cliffs are late "so…" conclusions): *both didn't hold.* The "late non sequitur" seen at 8 continuations per cut was noise at 16.
- **The clean cliffs are early slips, about 4–5 standard errors each:**
  - Right after correctly deriving 0.7-repeating = 7/9, it writes "This means 0.4-repeating = 4/10": p(right) 0.75 → 0.06.
  - Coins worth "75 cents", then "A nickel is worth **$0.05**": a true fact in the wrong unit. p(right) 1.00 → 0.25.

## The units cliff, up close

From the same point, sampling just the next line 64 times: **dollars 1/64**, cents 56/64. The losing line was a ~2% draw. Continuing after a cents line: right 1.00; after a dollars line: 0.38. Writing the earlier line consistently in dollars ("= $0.75$") keeps it right (0.90). Mixing frames is what hurts.

## Can it catch its own mistakes?

Restarting from just after each slip:
- The **original** model **notices** (30 of 32 continuations reconsider within a few lines: "Check: 4/10 = 0.400… Wait, is it a repeating decimal?") but almost never finishes in budget.
- The **leader** finishes but **doesn't notice**, or checks the wrong thing: "4/100 = 0.04. Correct."
- Inserting one line, "Wait, let me double-check that.", lifts the leader from 3 → 11 and 8 → 15 of 32. *Predicted ≥ +0.3; got +0.25 and +0.22, close.* The repair is there once it's pointed at the problem.

## Files

`resample_cloud.py`, `resample2_cloud.py`, `units_cloud.py`, `recover_cloud.py`.
