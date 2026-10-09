#!/usr/bin/env python3
"""PiPER-X data contract and OpenPI π0.5 LoRA configuration.

This module is intentionally project-owned: OpenPI has no upstream PiPER-X
adapter.  It uses only public interfaces present in the locked OpenPI archive.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np

from openpi import transforms
from openpi.models import model as model_lib
from openpi.models import pi0_config
from openpi.training import config
from openpi.training import optimizer
from openpi.training import weight_loaders


CONFIG_NAME = "pi05_piper_x_lora_h20"
CONTRACT_ID = "piper_x_joint_position_dual_rgb_v0.3"
DATASET_ID = "local/piper_x_embodiment_norm20_v0_1"
ACTION_HORIZON = 20
ACTION_DIM = 7
MODEL_ACTION_DIM = 32
BATCH_SIZE = 8
TRAIN_STEPS = 3_000
SNAPSHOT_INTERVAL = 500
OPENPI_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"
OPENPI_INTERFACE_BASE_COMMIT = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
NORM_STATS_ID = "piper_x_embodiment_norm20_v0_1"
ACTION_SEMANTICS = "absolute_joint_position_rad_plus_gripper_closure_0_1"


def _parse_image(value: Any, *, name: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"{name} must have three dimensions, got {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"{name} must be RGB, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all():
            raise ValueError(f"{name} contains NaN or Inf")
        # LeRobot 0.1 decodes video to float CHW in [0, 1].
        if image.size and float(np.max(image)) <= 1.0 + 1e-6:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _finite_last_dim(value: Any, *, name: str, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape[-1:] != (size,):
        raise ValueError(f"{name} must end in dimension {size}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


@dataclasses.dataclass(frozen=True)
class PiperXInputs(transforms.DataTransformFn):
    """Map PiPER dual-RGB records to OpenPI's canonical observation."""

    model_type: model_lib.ModelType

    def __call__(self, data: dict) -> dict:
        base = _parse_image(data["observation/image"], name="observation/image")
        wrist = _parse_image(data["observation/wrist_image"], name="observation/wrist_image")
        state = _finite_last_dim(data["observation/state"], name="observation/state", size=ACTION_DIM)
        output = {
            "state": state,
            "image": {
                "base_0_rgb": base,
                "left_wrist_0_rgb": wrist,
                "right_wrist_0_rgb": np.zeros_like(base),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_
                if self.model_type == model_lib.ModelType.PI0_FAST
                else np.False_,
            },
        }
        if "actions" in data:
            output["actions"] = _finite_last_dim(data["actions"], name="actions", size=ACTION_DIM)
        if "prompt" in data:
            output["prompt"] = data["prompt"]
        return output


@dataclasses.dataclass(frozen=True)
class PiperXOutputs(transforms.DataTransformFn):
    """Remove OpenPI's internal action padding after inverse transforms."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.shape[-1] < ACTION_DIM:
            raise ValueError(f"model action dimension is smaller than {ACTION_DIM}: {actions.shape}")
        return {"actions": actions[..., :ACTION_DIM]}


@dataclasses.dataclass(frozen=True)
class LeRobotPiperXDataConfig(config.DataConfigFactory):
    """LeRobot v2.1 PiPER absolute-joint action data."""

    @property
    def action_sequence_keys(self) -> tuple[str, ...]:
        return ("action",)

    def create(self, assets_dirs: Path, model_config: model_lib.BaseModelConfig) -> config.DataConfig:
        repack = transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.base",
                        "observation/wrist_image": "observation.images.wrist",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # q1..q6 are absolute joint targets and become delta relative to the
        # current q.  The seventh gripper value remains absolute.
        delta_mask = transforms.make_bool_mask(6, -1)
        data_transforms = transforms.Group(
            inputs=[PiperXInputs(model_type=model_config.model_type), transforms.DeltaActions(delta_mask)],
            outputs=[transforms.AbsoluteActions(delta_mask), PiperXOutputs()],
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=config.ModelTransformFactory()(model_config),
            action_sequence_keys=self.action_sequence_keys,
            prompt_from_task=True,
        )


def create_train_config(
    *,
    base_checkpoint: Path,
    assets_base_dir: Path,
    checkpoint_base_dir: Path,
    exp_name: str,
    dataset_id: str = DATASET_ID,
    num_train_steps: int = TRAIN_STEPS,
) -> config.TrainConfig:
    """Build the single supported PiPER LoRA experiment config."""
    base_checkpoint = base_checkpoint.resolve()
    params_dir = base_checkpoint / "params"
    if not params_dir.is_dir():
        raise FileNotFoundError(f"base checkpoint params not found: {params_dir}")
    model_config = pi0_config.Pi0Config(
        pi05=True,
        action_dim=MODEL_ACTION_DIM,
        action_horizon=ACTION_HORIZON,
        max_token_len=200,
        discrete_state_input=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    return config.TrainConfig(
        name=CONFIG_NAME,
        project_name="fragilegrab-piper-lora",
        exp_name=exp_name,
        model=model_config,
        data=LeRobotPiperXDataConfig(
            repo_id=dataset_id,
            base_config=config.DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(params_dir)),
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=100,
            peak_lr=2.5e-5,
            # A 2-step smoke must exercise the exact first two steps of the
            # formal 3k schedule; shrinking decay_steps below warmup_steps is
            # invalid in Optax and would also make the smoke unrepresentative.
            decay_steps=TRAIN_STEPS,
            decay_lr=2.5e-6,
        ),
        optimizer=optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        freeze_filter=model_config.get_freeze_filter(),
        assets_base_dir=str(assets_base_dir.resolve()),
        checkpoint_base_dir=str(checkpoint_base_dir.resolve()),
        batch_size=BATCH_SIZE,
        num_workers=2,
        num_train_steps=num_train_steps,
        log_interval=10,
        save_interval=SNAPSHOT_INTERVAL,
        keep_period=SNAPSHOT_INTERVAL,
        wandb_enabled=True,
        policy_metadata={
            "openpi_commit": OPENPI_COMMIT,
            "openpi_interface_base_commit": OPENPI_INTERFACE_BASE_COMMIT,
            "contract_id": CONTRACT_ID,
            "norm_stats_id": NORM_STATS_ID,
            "normalization_purpose": "piper_x_lora_training_and_inference",
            "execution_permitted": True,
            "effective_action_dim": ACTION_DIM,
            "model_action_dim": MODEL_ACTION_DIM,
            "action_horizon": ACTION_HORIZON,
            "action_semantics": ACTION_SEMANTICS,
            "image_roles": {
                "base_0_rgb": "present",
                "left_wrist_0_rgb": "present",
                "right_wrist_0_rgb": "masked",
            },
            "raw_action_semantics": "absolute_q1_q6_rad_plus_absolute_gripper_closedness",
            "model_action_semantics": "delta_q1_q6_rad_plus_absolute_gripper_closedness",
        },
    )


def validate_train_config(train_config: config.TrainConfig) -> None:
    """Fail closed if a CLI/config edit changes the locked experiment."""
    expected = {
        "name": CONFIG_NAME,
        "batch_size": BATCH_SIZE,
        "num_train_steps": TRAIN_STEPS,
        "action_dim": MODEL_ACTION_DIM,
        "action_horizon": ACTION_HORIZON,
    }
    actual = {
        "name": train_config.name,
        "batch_size": train_config.batch_size,
        "num_train_steps": train_config.num_train_steps,
        "action_dim": train_config.model.action_dim,
        "action_horizon": train_config.model.action_horizon,
    }
    mismatches = {key: (expected[key], actual[key]) for key in expected if expected[key] != actual[key]}
    if mismatches:
        raise ValueError(f"locked PiPER LoRA config mismatch: {mismatches}")
    if train_config.ema_decay is not None:
        raise ValueError("LoRA training must keep EMA disabled")
