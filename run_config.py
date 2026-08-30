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


def main(argv: list[str] | None = None) -> tuple[Path, int]:
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

    if cfg.get("use_deepspeed"):
        # DeepSpeed ZeRO-2, launched through `accelerate`, sharding
        # optimizer state across `num_processes` GPUs -- HF's Trainer
        # auto-detects DeepSpeed from the accelerate-launch environment
        # and wires it up transparently, so this needs ZERO changes to
        # Full_finetuning.py/SPPFT.py themselves (no `deepspeed=` argument
        # in their hardcoded TrainingArguments call). Fixes a CUDA OOM
        # that persisted even at micro_batch_size=1 (fixed, batch-size-
        # independent optimizer-state allocation, not activation memory).
        # A CPU-offload variant (ds_zero2_offload.json) was tried first
        # but failed to build its cpu_adam op (ninja/linker error) in this
        # container -- sharding across GPUs avoids that build entirely,
        # since the optimizer step still runs on GPU. Using CLI flags, not
        # a static accelerate config YAML, since a relative
        # deepspeed_config_file path in a committed YAML would resolve
        # differently locally vs. inside a container with a different
        # REPO_ROOT.
        ds_config = REPO_ROOT / "ds_zero2_shard.json"
        num_processes = str(cfg.get("num_processes", 1))
        cmd = [
            "accelerate", "launch",
            "--use_deepspeed",
            "--deepspeed_config_file", str(ds_config),
            "--zero_stage", "2",
            "--num_processes", num_processes,
            script_path.name,
        ] + cli_args
    else:
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

    print(f"[run_config] + cd {cwd} && {' '.join(cmd)}", flush=True)
    log_path = out_dir / "stdout_stderr.log"
    # Tee to both the log file AND this process's own stdout -- when run
    # inside a SageMaker training job, stdout is what CloudWatch Logs
    # captures, so the underlying script's real output (including error
    # tracebacks) is visible there immediately, without needing to
    # download the (potentially huge) S3 model artifact just to read a
    # small log file.
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, universal_newlines=True,
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
            log_f.write(line)
        proc.wait()
    returncode = proc.returncode

    metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["returncode"] = returncode
    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[run_config] Done (returncode={returncode}). Log: {log_path}")
    if returncode != 0:
        print(f"[run_config] NON-ZERO EXIT -- check {log_path} for the error.", file=sys.stderr)
    return out_dir, returncode


if __name__ == "__main__":
    _, _returncode = main()
    # Propagate the underlying script's exit code as run_config.py's own --
    # previously this always exited 0 even when the wrapped script failed,
    # which made the SageMaker entrypoint's per-config error handling
    # silently useless (it checked THIS process's returncode, not the
    # training script's).
    sys.exit(_returncode)
