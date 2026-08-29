# Config-driven runner (added on top of the fork)

This file, `run_config.py`, and everything under `configs/` are additions
made on top of this fork -- they do not modify any file under `Code/` or
`Dataset/`. Everything under `Code/` runs completely unmodified.

## Why

The scripts in this repo (`save_all_pairs_cos_sim.py`, `scaling.py`,
`Full_finetuning.py`, `SPPFT.py`) are each a `fire.Fire`-based CLI, meant
to be run as `cd Code/<subdir> && python <script>.py --arg1=val1 ...`. For
large-scale or repeated experiments (many models, many hyperparameter
sweeps), a versioned config file is easier to track and re-run than a long
remembered command line. `run_config.py` reads a YAML file describing one
such invocation and runs it via `subprocess`, in the correct working
directory, exactly as if you'd typed the command yourself -- it does not
import or reimplement any of the original logic.

Every run automatically gets a `results/<run_name>/` folder containing:
- `run_metadata.json` -- the resolved config, the exact command run, this
  fork's git commit at run time, start/end timestamps, and the exit code.
- `stdout_stderr.log` -- the full captured output of the run.

## Usage

```bash
python run_config.py --config configs/gemma_finetune_full_normal.yaml

# Override a field without editing the file:
python run_config.py --config configs/gemma_finetune_full_normal.yaml --set args.learning_rate=5e-5
```

## Current configs

All four target `google/gemma-2b-it` (see the cross-referenced
`silu101/safety-layers-repro` repo for why gemma is this project's focus
model, and its `docs/KNOWN_DISCREPANCIES.md` for the reasoning behind
every non-default value below):

| Config | Script | Notes |
|---|---|---|
| `gemma_finetune_full_normal.yaml` | `Full_finetuning.py` | Normal (DN) data. LR=1e-4 per Appendix Table 6 (not the script's own 3e-4 default). |
| `gemma_finetune_full_implicit.yaml` | `Full_finetuning.py` | Uses `Backdoor_dataset.json` AS Implicit (DI) data -- its content matches DI's definition, not Backdoor's (see `KNOWN_DISCREPANCIES.md` #5). |
| `gemma_finetune_sppft_normal.yaml` | `SPPFT.py` | `begin_num=5, end_num=12` to correctly freeze gemma's confirmed safety layers [6,11] inclusive, given `SPPFT.py`'s strict-inequality freeze condition (`KNOWN_DISCREPANCIES.md` #13, #15). |
| `gemma_finetune_sppft_implicit.yaml` | `SPPFT.py` | Same DI-via-Backdoor_dataset.json note as above. |

True Backdoor (DB) data does not exist anywhere in this repo (confirmed by
inspecting `Dataset/Finetune/` directly) and is out of scope for now.

## Extending this to other scripts / future experiments

The same pattern applies to any of the other fire.Fire scripts in this
repo -- write a YAML with `script:` (path relative to repo root),
`run_name:`, and `args:` (a flat dict matching that script's CLI flags
exactly), then run it with `run_config.py`. This is the intended
extension point for later large-scale or out-of-distribution experiments
(new prompt sets, new datasets, new sweep parameters) without touching any
of the original code:

- `Code/Cos_sim_analysis/save_all_pairs_cos_sim.py` -- NOTE: this one calls
  `fire.Fire(main)` at module level with no `if __name__ == "__main__":`
  guard, so it must be invoked via `run_config.py` (subprocess) like the
  others -- it cannot be safely `import`-ed as a Python module.
- `Code/Safety_layers_locating/scaling.py`
- `Code/Attention_scores/att_scores.py` (remember the gemma-specific
  uncomment-lines-51-and-85 requirement noted in the main README when
  adding a config for this one -- `run_config.py` can't paper over an
  edit that has to happen in the source file itself).
