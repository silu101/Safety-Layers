"""
SageMaker training-job entrypoint for Section 4 (Full FT vs SPPFT on
gemma-2b-it, Normal + Implicit data). Runs run_config.py for each of the
4 configs in order -- run_config.py itself invokes the ORIGINAL,
unmodified Full_finetuning.py/SPPFT.py via subprocess, so nothing here
reimplements their logic.

Mirrors silu101/safety-layers-repro's sagemaker/entrypoint_localization.py
for the environment fixes already worked out there (torch/transformers
pinning, HF login for gated gemma-2b-it, sentencepiece for its tokenizer).

Outputs (results/ metadata+logs, and output_models/ checkpoints) are
copied to /opt/ml/model/ so SageMaker syncs them back to S3.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/silu101/Safety-Layers"
REPO_DIR = Path("/opt/ml/code/Safety-Layers")
SM_MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))

CONFIGS = [
    "gemma_finetune_full_normal.yaml",
    "gemma_finetune_full_implicit.yaml",
    # sppft_normal and sppft_implicit already succeeded in the previous
    # job run (2026-08-30) -- not re-running them here to save time/cost.
    # Re-add if a full re-run is ever needed:
    # "gemma_finetune_sppft_normal.yaml",
    # "gemma_finetune_sppft_implicit.yaml",
]


def sh(cmd, **kw):
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kw)


def hf_login():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[entrypoint] No HF_TOKEN in environment -- skipping HF login.")
        return
    from huggingface_hub import login
    login(token=token)
    print("[entrypoint] Hugging Face login OK.")


def main():
    sh(["git", "clone", REPO_URL, str(REPO_DIR)])
    os.chdir(str(REPO_DIR))

    print("=" * 70)
    print("[entrypoint] DIAGNOSTIC: preinstalled torch before any pip install")
    subprocess.run([sys.executable, "-c",
                    "import torch; print('torch OK:', torch.__version__, torch.cuda.is_available())"])
    print("=" * 70)

    # Same fix as safety-layers-repro's SageMaker entrypoint: do NOT let
    # pip touch torch (breaks the DLC's preinstalled, GPU-matched build).
    # Pin transformers -- an unbounded install resolved to 5.16.1 there,
    # which broke torch detection entirely.
    sh([sys.executable, "-m", "pip", "install",
        "transformers==4.44.2", "accelerate==0.31.0", "datasets==2.20.0",
        "sentencepiece>=0.1.99", "fire>=0.5.0", "pyyaml>=6.0"])
    hf_login()

    for config in CONFIGS:
        print("=" * 70)
        print(f"[entrypoint] Running {config}")
        print("=" * 70)
        out_dir = subprocess.run(
            [sys.executable, "run_config.py", "--config", f"configs/{config}"],
            cwd=str(REPO_DIR),
        )
        print(f"[entrypoint] {config} -> returncode {out_dir.returncode}")
        if out_dir.returncode != 0:
            print(f"[entrypoint] WARNING: {config} failed (returncode {out_dir.returncode}), "
                  "continuing to the next config rather than aborting the whole job.")

    for name in ["results", "output_models"]:
        src = REPO_DIR / name
        dest = SM_MODEL_DIR / name
        if src.exists():
            shutil.copytree(src, dest, dirs_exist_ok=True)
            print(f"[entrypoint] Copied {src} -> {dest}")

    print("[entrypoint] Done.")


if __name__ == "__main__":
    main()
