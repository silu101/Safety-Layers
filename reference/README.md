# Reference: OpenReview submission code

`scaling_openreview_submission.py` is a verbatim copy of
`Code/Safety_layers_locating/scaling.py` **as it appeared in the paper's
OpenReview supplementary material** (the code submitted alongside the
ICLR 2025 paper), kept here unmodified for comparison against the current
`Code/Safety_layers_locating/scaling.py` in this repo.

**The two differ.** Diffed byte-for-byte 2026-08-24: every other file in
the repo (`Full_finetuning.py`, `SPPFT.py`, `att_scores.py`, all datasets)
is identical between the OpenReview archive and the current GitHub repo.
Only `scaling.py` differs, in `scaling()` and `scaling_phi3()`:

```diff
-            base_model.model.layers[i].self_attn.o_proj.weight.copy_(\
+            new_model.model.layers[i].self_attn.o_proj.weight.copy_(\
```

In the OpenReview (submission-time) version, the `o_proj` weight update
writes into `base_model` instead of `new_model`. Since `new_model` is a
`copy.deepcopy(base_model)` taken *before* this line runs, `new_model`'s
`o_proj` is never actually scaled -- it keeps its original value. Only
q/k/v (or the fused `qkv_proj` for Phi-3) and the MLP projections are
actually scaled in the model that gets returned and used for generation.
`base_model` (the caller's own input object) is mutated as an unused side
effect instead.

The current GitHub version has since been edited to target `new_model`
correctly. This means: the paper's actual published Table 1 numbers were
almost certainly produced WITHOUT `o_proj` being scaled at all -- a
materially different operation than what the current GitHub code (or a
literal reading of the paper's Section 3.4.1 formula) would produce.

See `silu101/safety-layers-repro`'s `docs/KNOWN_DISCREPANCIES.md` #17 for
the full write-up and reproduction implications.
