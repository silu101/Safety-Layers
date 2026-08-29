"""
Config-driven runner for this repo's original scripts -- added on top of
the unmodified fork, does not change any file under Code/ or Dataset/.

Every script here (save_all_pairs_cos_sim.py, scaling.py, Full_finetuning.py,
SPPFT.py) is a fire.Fire-based CLI, meant to be invoked as
`cd Code/<subdir> && python <script>.py --arg1=val1 --arg2=val2 ...`.
This wrapper reads a YAML config describing one such invocation and runs it
exactly that way via subprocess -- it does NOT import or reimplement any of
their logic, so the code that actually executes is 100% theirs, unmodified.

Why a config file instead of typing the CLI invocation by hand each time:
for large-scale / repeated experiments (many models, many hyperparameter
sweeps), a config file is easier to version, diff, and re-run than a long
remembered command line, and this wrapper automatically saves the resolved
config + git commit + timestamp + full stdout/stderr next to every run for
provenance.

Usage:
    python run_config.py --config configs/gemma_finetune_full_normal.yaml
    python run_config.py --config configs/gemma_finetune_full_normal.yaml --set args.learning_rate=5e-5
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent


def _git_commit(path: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def _apply_set_overrides(cfg: dict, sets: list[str]) -> dict:
    """--set args.learning_rate=5e-5 overrides cfg['args']['learning_rate'].
    --set cwd=Code/Fine_tune overrides a top-level key."""
    for kv in sets:
        key, _, value = kv.partition("=")
        parts = key.split(".")
        node = cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        # Try to parse the value as YAML (so numbers/bools come through
        # typed, not as strings), fall back to raw string.
        try:
            node[parts[-1]] = yaml.safe_load(value)
        except Exception:
            node[parts[-1]] = value
    return cfg


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config (see configs/).")
    parser.add_argument(
        "--set", action="append", default=[], dest="sets", metavar="key=value",
        help="Override a config field, e.g. --set args.learning_rate=5e-5",
    )
    args = parser.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg = _apply_set_overrides(cfg, args.sets)

    script = cfg["script"]           # e.g. "Code/Fine_tune/Full_finetuning.py"
    script_path = REPO_ROOT / script
    cwd = script_path.parent
    script_args = cfg.get("args", {})

    cli_args = [f"--{k}={v}" for k, v in script_args.items()]
    cmd = [sys.executable, script_path.name] + cli_args

    run_name = cfg.get("run_name") or f"{script_path.stem}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir = REPO_ROOT / "results" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "config_file": str(Path(args.config).resolve()),
        "resolved_config": cfg,
        "command": cmd,
        "cwd": str(cwd),
        "fork_git_commit": _git_commit(REPO_ROOT),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[run_config] + cd {cwd} && {' '.join(cmd)}")
    log_path = out_dir / "stdout_stderr.log"
    with open(log_path, "w") as log_f:
        result = subprocess.run(cmd, cwd=cwd, stdout=log_f, stderr=subprocess.STDOUT)

    metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["returncode"] = result.returncode
    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[run_config] Done (returncode={result.returncode}). Log: {log_path}")
    if result.returncode != 0:
        print(f"[run_config] NON-ZERO EXIT -- check {log_path} for the error.", file=sys.stderr)
    return out_dir


if __name__ == "__main__":
    main()
