"""
Launch Section 4 (Full FT vs SPPFT, gemma-2b-it, Normal + Implicit) as a
single SageMaker training job, running the configs in sagemaker_entrypoint.py's
CONFIGS list in sequence. Currently ml.g6e.12xlarge (4x L40S) for the Full
FT configs' DeepSpeed ZeRO-2 GPU-sharded optimizer state -- see that
file's CONFIGS list and configs/gemma_finetune_full_*.yaml.

Usage:
    python sagemaker_launch.py
"""
import sagemaker
from sagemaker.pytorch import PyTorch
from pathlib import Path

ROLE_ARN = "arn:aws:iam::344977996863:role/safety-layers-sagemaker-execution-role"
REGION = "us-east-1"

HF_TOKEN_FILE = Path.home() / ".hf_token_safety_layers"
hf_token = HF_TOKEN_FILE.read_text().strip() if HF_TOKEN_FILE.exists() else None
if not hf_token:
    raise SystemExit(
        f"No HF token found at {HF_TOKEN_FILE}. Create it first: "
        f'echo "hf_your_real_token" > {HF_TOKEN_FILE} && chmod 600 {HF_TOKEN_FILE}'
    )

session = sagemaker.Session(boto_session=__import__("boto3").Session(region_name=REGION))

estimator = PyTorch(
    entry_point="sagemaker_entrypoint.py",
    source_dir=".",
    role=ROLE_ARN,
    framework_version="2.3",
    py_version="py311",
    # 4x L40S (192GB total) -- Full FT needs DeepSpeed ZeRO-2 to shard
    # optimizer state across GPUs (see ds_zero2_shard.json,
    # configs/gemma_finetune_full_*.yaml's use_deepspeed+num_processes:4).
    # A single-GPU ml.g6e.xlarge OOM'd inside AdamW.step() even at
    # micro_batch_size=1, and CPU-offload failed to build its cpu_adam op.
    instance_type="ml.g6e.12xlarge",
    instance_count=1,
    sagemaker_session=session,
    base_job_name="safety-layers-section4-gemma",
    max_run=6 * 60 * 60,  # 6 hours ceiling -- 4 small fine-tuning runs
                          # (900-3900 train examples each, 3 epochs), a
                          # generous but bounded cost/safety guard.
    environment={
        "HF_TOKEN": hf_token,
        # Full FT hit a CUDA OOM inside AdamW's optimizer.step() (fixed
        # per-parameter optimizer-state memory, independent of batch
        # size), short by only 128MB. This is a pure memory-allocator
        # setting -- reduces fragmentation, changes nothing about
        # training semantics/hyperparameters/results. Try this before
        # touching any actual config value.
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    },
)

if __name__ == "__main__":
    estimator.fit(wait=True, logs=True)
    print("Job name:", estimator.latest_training_job.name)
    print("Model artifacts:", estimator.model_data)
