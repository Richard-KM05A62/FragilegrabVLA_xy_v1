"""Offline GPU/checkpoint validation; run as python -m tools.check_pi05_load."""

import argparse
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_STATS_SHA = "e63328620231674b9633171955503564e01353f311edfb84f5627938ab878b50"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("gpu", "model"), required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("OPENPI_DATA_HOME", str(ROOT / ".cache/openpi"))

    import jax
    import jax.numpy as jnp
    import numpy as np

    print("python:", sys.executable, sys.version, flush=True)
    for name in ("jax", "jaxlib", "flax", "orbax-checkpoint", "numpy", "datasets", "torch", "transformers"):
        print(name, md.version(name), flush=True)
    print("backend:", jax.default_backend(), "devices:", jax.devices(), flush=True)
    assert jax.default_backend() == "gpu", "GPU backend required"
    if args.stage == "gpu":
        x = jnp.ones((256, 256), dtype=jnp.bfloat16)
        y = jax.jit(lambda a: a @ a)(x)
        y.block_until_ready()
        assert bool(jnp.isfinite(y).all())
        print("GPU_SMOKE_OK", y.shape, y.dtype, flush=True)
        return

    import flax.nnx as nnx
    from flax import traverse_util
    from openpi.training import config as training_config
    from inference.piper_rtc_policy import create_rtc_policy

    checkpoint = args.checkpoint.resolve()
    stats_path = checkpoint / "assets/local/piper_x_recordings_v000/norm_stats.json"
    stats_sha = hashlib.sha256(stats_path.read_bytes()).hexdigest()
    assert stats_sha == EXPECTED_STATS_SHA, stats_sha
    stats = json.loads(stats_path.read_text())["norm_stats"]
    for name in ("state", "actions"):
        for key in ("mean", "std", "q01", "q99"):
            a = np.asarray(stats[name][key])
            assert a.shape == (7,) and np.isfinite(a).all(), (name, key, a.shape)
    cfg = training_config.get_config("pi05_piper_finetune")
    assert cfg.model.pi05 and cfg.model.action_horizon == 10 and cfg.model.action_dim == 32
    assert cfg.model.discrete_state_input is False
    assert cfg.model.paligemma_variant == "gemma_2b_lora"
    assert cfg.model.action_expert_variant == "gemma_300m_lora"
    print("checkpoint:", checkpoint, "stats_sha256:", stats_sha, flush=True)
    started = time.perf_counter()
    policy = create_rtc_policy("pi05_piper_finetune", checkpoint, num_steps=10, max_guidance=5.0)
    params = nnx.state(policy._base._model, nnx.Param).to_pure_dict()
    jax.block_until_ready(params)
    flat = traverse_util.flatten_dict(params)
    lora = [p for p in flat if "lora" in "/".join(map(str, p)).lower()]
    assert len(flat) == 71, f"Unexpected parameter leaves: {len(flat)}"
    assert len(lora) == 20, f"LoRA leaves missing/changed: {len(lora)}"
    for path, value in flat.items():
        assert isinstance(value, jax.Array), path
        assert all(d.platform == "gpu" for d in value.devices()), path
    metadata = policy.metadata
    expected = {
        "norm_stats_sha256": EXPECTED_STATS_SHA, "horizon": 10, "action_dim": 7,
        "model_action_dim": 32, "action_space": "absolute_joint_position",
        "joint_unit": "rad", "gripper": "closedness",
    }
    for key, value in expected.items():
        assert metadata[key] == value, (key, metadata[key])
    print("parameter_leaves:", len(flat), "lora_leaves:", len(lora), flush=True)
    print("metadata:", json.dumps(metadata, ensure_ascii=False, default=str), flush=True)
    print("MODEL_LOAD_OK elapsed_s=", round(time.perf_counter() - started, 3), flush=True)


if __name__ == "__main__":
    main()
