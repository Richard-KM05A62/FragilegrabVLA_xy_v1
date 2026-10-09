#!/usr/bin/env python3
"""Serve a read-only π0.5 base checkpoint with a verified PiPER LoRA overlay."""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

from flax import nnx
import jax

from openpi import transforms
from openpi.policies import policy as policy_lib
from openpi.serving import websocket_policy_server

from scripts.policy.piper_lora_snapshot import load_lora_snapshot, overlay_lora_params
from scripts.policy.piper_openpi_lora import ACTION_HORIZON, BATCH_SIZE, create_train_config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-snapshot", type=Path, required=True)
    parser.add_argument("--assets-base-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset-id", default="local/piper_x_embodiment_norm20_v0_1")
    parser.add_argument("--default-prompt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    train_config = create_train_config(
        base_checkpoint=args.base_checkpoint,
        assets_base_dir=args.assets_base_dir,
        checkpoint_base_dir=args.run_root,
        exp_name="inference-only",
        dataset_id=args.dataset_id,
    )
    arrays, manifest = load_lora_snapshot(args.adapter_snapshot)
    expected = {
        "dataset_id": args.dataset_id,
        "action_horizon": ACTION_HORIZON,
        "batch_size": BATCH_SIZE,
        "base_params_sha256": _sha256(args.base_checkpoint.resolve() / "params/_METADATA"),
    }
    mismatch = {key: (value, manifest.get(key)) for key, value in expected.items() if manifest.get(key) != value}
    if mismatch:
        raise RuntimeError(f"adapter identity mismatch: {mismatch}")

    abstract_model = nnx.eval_shape(train_config.model.create, jax.random.key(0))
    # Use the same NNX split/state representation as BaseModelConfig.load and
    # OpenPI's training initializer.  nnx.state(abstract_model) on this locked
    # Flax version leaves VariableState objects at the leaves; np.asarray then
    # reports scalar object shapes, which makes every real adapter array look
    # incompatible even though the checkpoint is valid.
    _, reference_state = nnx.split(abstract_model)
    reference_params = reference_state.to_pure_dict()
    base_with_initialized_lora = train_config.weight_loader.load(reference_params)
    params = overlay_lora_params(base_with_initialized_lora, args.adapter_snapshot)
    model = train_config.model.load(params)

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.norm_stats is None:
        raise FileNotFoundError("PiPER norm stats were not loaded")
    policy = policy_lib.Policy(
        model,
        transforms=[
            transforms.InjectDefaultPrompt(args.default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        metadata={
            **(train_config.policy_metadata or {}),
            "adapter_step": manifest["step"],
            "adapter_archive_sha256": manifest["archive_sha256"],
            "adapter_leaf_count": len(arrays),
        },
    )
    websocket_policy_server.WebsocketPolicyServer(
        policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
