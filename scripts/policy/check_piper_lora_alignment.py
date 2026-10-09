#!/usr/bin/env python3
"""Offline gate for PiPER LeRobot → OpenPI training alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flax import nnx
import jax
import numpy as np
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

from openpi import transforms
from openpi.training import data_loader

from scripts.policy.piper_openpi_lora import create_train_config


def _paths(value, prefix=()):
    if hasattr(value, "to_pure_dict"):
        value = value.to_pure_dict()
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _paths(child, (*prefix, str(key)))
    else:
        yield "/".join(prefix)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--assets-base-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset-id", default="local/piper_x_embodiment_norm20_v0_1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train_config = create_train_config(
        base_checkpoint=args.base_checkpoint,
        assets_base_dir=args.assets_base_dir,
        checkpoint_base_dir=args.run_root,
        exp_name="alignment-only",
        dataset_id=args.dataset_id,
    )
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.norm_stats is None:
        raise FileNotFoundError("norm stats must exist before the final alignment gate")
    metadata = lerobot_dataset.LeRobotDatasetMetadata(args.dataset_id)
    robot_type = getattr(metadata, "robot_type", None)
    if robot_type is None and isinstance(getattr(metadata, "info", None), dict):
        robot_type = metadata.info.get("robot_type")
    if not isinstance(robot_type, str) or not robot_type.lower().startswith("piper_x"):
        raise ValueError(f"dataset robot_type must identify PiPER-X, got {robot_type!r}")
    raw_dataset = data_loader.create_torch_dataset(data_config, 20, train_config.model)
    raw = raw_dataset[0]
    # Copy before DeltaActions, which intentionally updates the repacked action
    # array in place and may share storage with the raw sample.
    raw_state = np.array(raw["observation.state"], dtype=np.float32, copy=True)
    raw_actions = np.array(raw["action"], dtype=np.float32, copy=True)
    pre_norm_transform = transforms.compose(
        [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs]
    )
    transformed = pre_norm_transform(raw)

    state = np.asarray(transformed["state"])
    actions = np.asarray(transformed["actions"])
    if raw_state.shape != (7,) or raw_actions.shape != (20, 7):
        raise ValueError(f"raw contract mismatch: state={raw_state.shape}, actions={raw_actions.shape}")
    if state.shape != (7,) or actions.shape != (20, 7):
        raise ValueError(f"pre-norm contract mismatch: state={state.shape}, actions={actions.shape}")
    np.testing.assert_allclose(actions[..., :6], raw_actions[..., :6] - raw_state[:6], atol=1e-6, rtol=0)
    np.testing.assert_allclose(actions[..., 6], raw_actions[..., 6], atol=1e-6, rtol=0)

    full_dataset = data_loader.transform_dataset(raw_dataset, data_config)
    model_sample = full_dataset[0]
    if np.asarray(model_sample["state"]).shape != (32,):
        raise ValueError(f"model state must be padded to 32: {np.asarray(model_sample['state']).shape}")
    if np.asarray(model_sample["actions"]).shape != (20, 32):
        raise ValueError(f"model actions must be (20, 32): {np.asarray(model_sample['actions']).shape}")

    abstract_model = nnx.eval_shape(train_config.model.create, jax.random.key(0))
    trainable = nnx.state(abstract_model).filter(train_config.trainable_filter)
    trainable_paths = tuple(_paths(trainable))
    if not trainable_paths or not any("lora" in path.lower() for path in trainable_paths):
        raise ValueError("official trainable filter contains no LoRA parameter leaves")
    unexpected_llm = [
        path for path in trainable_paths if "llm" in path.lower() and "lora" not in path.lower()
    ]
    if unexpected_llm:
        raise ValueError(f"frozen LLM base parameters became trainable: {unexpected_llm[:10]}")

    report = {
        "dataset_id": args.dataset_id,
        "robot_type": robot_type,
        "raw_state_shape": list(raw_state.shape),
        "raw_action_shape": list(raw_actions.shape),
        "pre_norm_state_shape": list(state.shape),
        "pre_norm_action_shape": list(actions.shape),
        "model_state_shape": list(np.asarray(model_sample["state"]).shape),
        "model_action_shape": list(np.asarray(model_sample["actions"]).shape),
        "joint_delta_max_abs": float(np.max(np.abs(actions[..., :6]))),
        "gripper_min": float(np.min(actions[..., 6])),
        "gripper_max": float(np.max(actions[..., 6])),
        "trainable_leaf_count": len(trainable_paths),
        "trainable_filter_matches_official_lora_contract": True,
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
