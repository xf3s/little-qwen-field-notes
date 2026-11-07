# 02 · two kinds of doubt

## In plain words

Not all doubt is the same. Using the unedited model's own long traces, we compared two kinds of moment:
- **stuck**: escaping a line it has just repeated three times ("a groove");
- **noodling**: reconsidering for no particular reason.

They're **different directions** through the middle of the model. Random handfuls of noodling examples agree with each other at cos 0.90+; stuck vs noodle sits near 0. Our doubt edit ([01](../01-the-doubt-edit/)) removed the noodling one. That matters: in the unedited model, **72 of 83 escapes from a groove begin with "Wait"**. The edited model has fewer grooves (24) but escapes only 15.

## A needle-lifter

We added one new neuron (widening the MLP, nothing overwritten) that fires on "stuck" and writes toward ending the thought.
- Calibrated, it fires in grooves (median 13.8) and essentially nowhere else.
- It **halves the grooves and cuts loops 25–45%**, but the **score doesn't rise** (47–49 vs 54 of 120). Lifted out of a groove, the model often leaves with a wrong answer. The loops were mostly runs that wouldn't have been right anyway.
- three's hunch held: gentle doses (0.5–2%) are the sweet spot ("it considers ending…"). Pushed hard toward ending, the grooves turn into questions ("* $159.75$ inches?").

## Also tried

- **Does the model know when its settled answer is right?** A direction for that came out *below* chance, consistently inverse, across and within problems. Not trusted, not used.
- **A neuron dictionary:** all 86,016 MLP neurons read through the vocabulary; 12,664 neuron-sides have a 5+ word family. Neuron 998 checks out.

## Files

`groove_cloud.py`, `implant.py`, `implant_calibrate_cloud.py`, `implant_check_cloud.py`, `confidence_cloud.py`, `dictionary.py`.
