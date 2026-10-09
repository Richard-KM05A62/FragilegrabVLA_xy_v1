#!/usr/bin/env python3
"""Compute PiPER stats after repack + robot transforms and before normalization."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import tqdm

from openpi.shared import normalize
from openpi.training import data_loader

from scripts.policy.piper_openpi_lora import create_train_config


class RemoveStrings:
    def __call__(self, value: dict) -> dict:
        return {key: item for key, item in value.items() if not np.issubdtype(np.asarray(item).dtype, np.str_)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--assets-base-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-base-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", default="local/piper_x_embodiment_norm20_v0_1")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    train_config = create_train_config(
        base_checkpoint=args.base_checkpoint,
        assets_base_dir=args.assets_base_dir,
        checkpoint_base_dir=args.checkpoint_base_dir,
        exp_name="norm-stats-only",
        dataset_id=args.dataset_id,
    )
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    dataset = data_loader.create_torch_dataset(data_config, train_config.model.action_horizon, train_config.model)
    dataset = data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
    )
    loader = data_loader.TorchDataLoader(
        dataset,
        local_batch_size=args.batch_size,
        num_workers=train_config.num_workers,
        shuffle=False,
        num_batches=len(dataset) // args.batch_size,
    )
    running = {key: normalize.RunningStats() for key in ("state", "actions")}
    first_batch_checked = False
    for batch in tqdm.tqdm(loader, total=len(dataset) // args.batch_size, desc="Computing PiPER stats"):
        if not first_batch_checked:
            state = np.asarray(batch["state"])
            actions = np.asarray(batch["actions"])
            if state.shape[-1] != 7:
                raise ValueError(f"transformed state must end in 7, got {state.shape}")
            if actions.shape[-2:] != (20, 7):
                raise ValueError(f"transformed actions must end in (20, 7), got {actions.shape}")
            if not np.isfinite(state).all() or not np.isfinite(actions).all():
                raise ValueError("transformed state/actions contain NaN or Inf")
            first_batch_checked = True
        for key in running:
            running[key].update(np.asarray(batch[key]))
    stats = {key: value.get_statistics() for key, value in running.items()}
    summary = {}
    for key, value in stats.items():
        fields = {
            name: np.asarray(getattr(value, name))
            for name in ("mean", "std", "q01", "q99")
            if getattr(value, name) is not None
        }
        for name, array in fields.items():
            if array.shape != (7,) or not np.isfinite(array).all():
                raise ValueError(f"invalid {key}.{name}: shape={array.shape}")
        span = fields["q99"] - fields["q01"]
        narrow = np.flatnonzero(span < 1e-3).tolist()
        if narrow:
            logging.warning("%s has q01/q99 span < 1e-3 at dimensions %s", key, narrow)
        summary[key] = {name: array.tolist() for name, array in fields.items()}
        summary[key]["q99_minus_q01"] = span.tolist()

    output = train_config.assets_dirs / args.dataset_id
    output.mkdir(parents=True, exist_ok=True)
    normalize.save(output, stats)
    stats_path = output / "norm_stats.json"
    provenance = {
        "schema_version": "fragilegrab.piper_norm_stats/0.1",
        "dataset_id": args.dataset_id,
        "contract_id": "piper_x_joint_position_dual_rgb_v0.3",
        "transform_order": ["repack", "PiperXInputs", "DeltaActions(q1..q6)"],
        "action_horizon": train_config.model.action_horizon,
        "effective_action_dim": 7,
        "norm_stats_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    (output / "stats_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
