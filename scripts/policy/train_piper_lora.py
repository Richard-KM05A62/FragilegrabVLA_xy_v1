#!/usr/bin/env python3
"""Train π0.5 PiPER-X LoRA without writing full-model checkpoints.

The optimization path is the locked OpenPI `scripts/train.py` implementation.
This entry point changes only orchestration and persistence: completed-step
numbering is fixed and every retained checkpoint contains only the official
trainable-filter state (LoRA plus small non-LLM projections/heads).
"""

from __future__ import annotations

import argparse
from flax.training import common_utils
import functools
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import platform

import etils.epath as epath
import jax
import jax.numpy as jnp
import numpy as np
import tqdm_loggable.auto as tqdm
import wandb

from openpi.training import data_loader as data_loader_lib
from openpi.training import sharding
from openpi.training import utils as training_utils

from scripts.policy.piper_lora_snapshot import SnapshotIdentity, save_lora_snapshot
from scripts.policy.piper_openpi_lora import (
    ACTION_HORIZON,
    BATCH_SIZE,
    SNAPSHOT_INTERVAL,
    TRAIN_STEPS,
    create_train_config,
    validate_train_config,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_official_train(openpi_root: Path):
    path = openpi_root.resolve() / "scripts/train.py"
    if not path.is_file():
        raise FileNotFoundError(f"official OpenPI train.py not found: {path}")
    spec = importlib.util.spec_from_file_location("fragilegrab_official_openpi_train", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official train module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _flatten_paths(value, prefix=()):
    if hasattr(value, "to_pure_dict"):
        value = value.to_pure_dict()
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _flatten_paths(child, (*prefix, str(key)))
    else:
        yield "/".join(prefix)


def _verify_only_lora_trainable(train_state, train_config) -> None:
    trainable = train_state.params.filter(train_config.trainable_filter)
    paths = tuple(_flatten_paths(trainable))
    if not paths:
        raise RuntimeError("trainable parameter set is empty")
    if not any("lora" in path.lower() for path in paths):
        raise RuntimeError("official trainable filter selected no LoRA leaves")
    # Official OpenPI LoRA also trains small non-LLM projections/heads.  What
    # must remain frozen are non-LoRA leaves inside either LLM.
    unexpected = [path for path in paths if "llm" in path.lower() and "lora" not in path.lower()]
    if unexpected:
        raise RuntimeError(f"frozen LLM base parameters became trainable: {unexpected[:10]}")
    logging.info("Verified %d official adapter-trainable leaves", len(paths))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--assets-base-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--dataset-id", default="local/piper_x_embodiment_norm20_v0_1")
    parser.add_argument("--steps", type=int, choices=(2, 500, 1000, 2000, TRAIN_STEPS), default=TRAIN_STEPS)
    parser.add_argument(
        "--save-steps",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Completed steps at which to save adapter-only snapshots. "
            "Defaults to every locked snapshot interval plus the final step."
        ),
    )
    parser.add_argument("--openpi-identity", required=True, help="Commit or recorded source hash identity")
    parser.add_argument("--minimum-free-gib", type=float, default=1.0)
    args = parser.parse_args()

    if args.save_steps is None:
        save_steps = set(range(SNAPSHOT_INTERVAL, args.steps + 1, SNAPSHOT_INTERVAL))
    else:
        save_steps = set(args.save_steps)
        invalid_save_steps = sorted(step for step in save_steps if step < 1 or step > args.steps)
        if invalid_save_steps:
            raise ValueError(f"save steps must be within [1, {args.steps}]: {invalid_save_steps}")
    save_steps.add(args.steps)

    official_train = _load_official_train(args.openpi_root)
    official_train.init_logging()
    logging.info("Running on %s", platform.node())

    train_config = create_train_config(
        base_checkpoint=args.base_checkpoint,
        assets_base_dir=args.assets_base_dir,
        checkpoint_base_dir=args.run_root,
        exp_name=args.exp_name,
        dataset_id=args.dataset_id,
        num_train_steps=args.steps,
    )
    if args.steps == TRAIN_STEPS:
        validate_train_config(train_config)
    if train_config.batch_size != BATCH_SIZE or train_config.model.action_horizon != ACTION_HORIZON:
        raise RuntimeError("batch size or action horizon changed from the locked contract")

    run_dir = Path(train_config.checkpoint_dir)
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists; choose a new exp-name: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "resolved_config.json").write_text(
        json.dumps(
            {
                "config_name": train_config.name,
                "dataset_id": args.dataset_id,
                "base_checkpoint": str(args.base_checkpoint.resolve()),
                "action_horizon": train_config.model.action_horizon,
                "model_action_dim": train_config.model.action_dim,
                "effective_action_dim": 7,
                "batch_size": train_config.batch_size,
                "num_train_steps": train_config.num_train_steps,
                "snapshot_interval": SNAPSHOT_INTERVAL,
                "save_steps": sorted(save_steps),
                "full_checkpoint_saving": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    norm_dir = train_config.assets_dirs / args.dataset_id
    norm_path = Path(norm_dir) / "norm_stats.json"
    if not norm_path.is_file():
        raise FileNotFoundError(f"transformed PiPER norm stats not found: {norm_path}")
    norm_sha = _sha256(norm_path)
    base_metadata = args.base_checkpoint.resolve() / "params/_METADATA"
    if not base_metadata.is_file():
        raise FileNotFoundError(f"base metadata not found: {base_metadata}")
    base_identity_sha = _sha256(base_metadata)

    if train_config.batch_size % jax.device_count() != 0:
        raise ValueError("batch size must be divisible by JAX device count")
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))
    rng = jax.random.key(train_config.seed)
    train_rng, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(train_config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    official_train.init_wandb(train_config, resuming=False, enabled=True)
    loader = data_loader_lib.create_data_loader(train_config, sharding=data_sharding, shuffle=True)
    iterator = iter(loader)
    batch = next(iterator)
    logging.info("Initialized data loader:\n%s", training_utils.array_tree_to_info(batch))
    def image_for_wandb(value):
        image = np.asarray(value)
        if np.issubdtype(image.dtype, np.floating) and image.size and float(image.min()) >= -1.01:
            image = (image + 1.0) / 2.0 * 255.0
        return np.clip(image, 0, 255).astype(np.uint8)

    images = [
        wandb.Image(np.concatenate([image_for_wandb(image[i]) for image in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images}, step=0)

    state, state_sharding = official_train.init_train_state(train_config, init_rng, mesh, resume=False)
    jax.block_until_ready(state)
    _verify_only_lora_trainable(state, train_config)
    ptrain_step = jax.jit(
        functools.partial(official_train.train_step, train_config),
        in_shardings=(replicated_sharding, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    infos = []
    progress = tqdm.tqdm(range(args.steps), total=args.steps, dynamic_ncols=True)
    try:
        for _ in progress:
            with sharding.set_mesh(mesh):
                state, info = ptrain_step(train_rng, state, batch)
            completed_step = int(state.step)
            infos.append(info)
            if completed_step % train_config.log_interval == 0 or completed_step == 1:
                stacked = common_utils.stack_forest(infos)
                reduced = jax.device_get(jax.tree.map(jnp.mean, stacked))
                reduced["learning_rate"] = np.asarray(
                    train_config.lr_schedule.create()(completed_step), dtype=np.float32
                )
                progress.write(", ".join(f"{key}={value:.4f}" for key, value in reduced.items()))
                wandb.log(reduced, step=completed_step)
                infos = []
            batch = next(iterator)

            should_save = completed_step in save_steps
            if should_save:
                output = run_dir / str(completed_step)
                identity = SnapshotIdentity(
                    step=completed_step,
                    openpi_commit=args.openpi_identity,
                    base_params_sha256=base_identity_sha,
                    dataset_id=args.dataset_id,
                    norm_stats_sha256=norm_sha,
                    action_horizon=ACTION_HORIZON,
                    batch_size=BATCH_SIZE,
                )
                manifest = save_lora_snapshot(
                    output,
                    state.params.filter(train_config.trainable_filter),
                    identity,
                    minimum_free_bytes_after_save=int(args.minimum_free_gib * 1024**3),
                )
                wandb.log(
                    {
                        "adapter/archive_bytes": manifest["archive_bytes"],
                        "adapter/leaf_count": manifest["parameter_leaf_count"],
                    },
                    step=completed_step,
                )
                logging.info("Saved adapter-only snapshot: %s", output)
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
