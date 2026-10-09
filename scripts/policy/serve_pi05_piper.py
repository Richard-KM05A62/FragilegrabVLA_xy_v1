#!/usr/bin/env python3
"""在锁定 OpenPI 环境中用 PiPER 专用 transforms 启动 pi05_base 服务。"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
from pathlib import Path
import subprocess

import numpy as np

import openpi
from openpi import transforms
from openpi.models import pi0_config
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.shared import normalize
from openpi.training import config


OPENPI_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"
OPENPI_INTERFACE_BASE_COMMIT = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
CONTRACT_ID = "piper_x_joint_position_dual_rgb_v0.3"
NORM_STATS_ID_PREFIX = "piper_x_"
ACTION_SEMANTICS = "absolute_joint_position_rad_plus_gripper_closure_0_1"


@dataclasses.dataclass(frozen=True)
class PiperXInputs(transforms.DataTransformFn):
    """请求字段到 π0.5 标准三图像槽和七维 state。"""

    def __call__(self, data: dict) -> dict:
        image = np.asarray(data["observation/image"])
        wrist_image = np.asarray(data["observation/wrist_image"])
        state = np.asarray(data["observation/state"], dtype=np.float64)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("observation/image 必须是 HWC uint8 RGB")
        if wrist_image.dtype != np.uint8 or wrist_image.shape != image.shape:
            raise ValueError("observation/wrist_image 必须与全局图同 shape 的 HWC uint8 RGB")
        if state.shape != (7,) or not np.isfinite(state).all():
            raise ValueError("observation/state 必须是有限的七维向量")
        output = {
            "state": state,
            "image": {
                "base_0_rgb": image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        if "actions" in data:
            output["actions"] = np.asarray(data["actions"], dtype=np.float64)
        if "prompt" in data:
            output["prompt"] = data["prompt"]
        return output


@dataclasses.dataclass(frozen=True)
class PiperXOutputs(transforms.DataTransformFn):
    """去除模型内部 32 维 padding，保留已完成逆变换的七个有效维度。"""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :7])}


def _openpi_root() -> Path:
    module = Path(openpi.__file__).resolve()
    for parent in module.parents:
        if (parent / ".git").exists() and (parent / "src/openpi").exists():
            return parent
    raise RuntimeError("无法从 openpi module 定位 Git checkout")


def _verify_openpi_commit() -> str:
    root = _openpi_root()
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != OPENPI_COMMIT:
        raise RuntimeError(f"OpenPI commit 不匹配：需要 {OPENPI_COMMIT}，实际 {commit}")
    core_changes = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--", "src/openpi"], text=True
    ).strip()
    if core_changes:
        raise RuntimeError(f"OpenPI 核心源码存在未记录修改：\n{core_changes}")
    return commit


def create_config(
    norm_stats_id: str,
    *,
    normalization_purpose: str,
    execution_permitted: bool,
) -> config.TrainConfig:
    delta_mask = transforms.make_bool_mask(6, -1)

    def piper_transforms(_model_config):
        return transforms.Group(
            inputs=[PiperXInputs(), transforms.DeltaActions(delta_mask)],
            outputs=[transforms.AbsoluteActions(delta_mask), PiperXOutputs()],
        )

    return config.TrainConfig(
        name="pi05_piper_inference",
        model=pi0_config.Pi0Config(pi05=True),
        data=config.SimpleDataConfig(data_transforms=piper_transforms),
        policy_metadata={
            "openpi_commit": OPENPI_COMMIT,
            "openpi_interface_base_commit": OPENPI_INTERFACE_BASE_COMMIT,
            "contract_id": CONTRACT_ID,
            "norm_stats_id": norm_stats_id,
            "normalization_purpose": normalization_purpose,
            "execution_permitted": execution_permitted,
            "action_horizon": 50,
            "effective_action_dim": 7,
            "action_semantics": ACTION_SEMANTICS,
            "image_roles": {"base_0_rgb": "present", "left_wrist_0_rgb": "present", "right_wrist_0_rgb": "masked"},
            "model_action_dim": 32,
            "max_token_len": 200,
            "sample_steps": 10,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--norm-stats-dir", type=Path, required=True)
    parser.add_argument("--norm-stats-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--default-prompt")
    args = parser.parse_args()

    if not args.norm_stats_id.startswith(NORM_STATS_ID_PREFIX):
        raise RuntimeError("PiPER-X norm_stats_id 必须以 piper_x_ 开头")
    provenance_path = args.norm_stats_dir / "provenance.json"
    if not provenance_path.is_file():
        raise RuntimeError(f"PiPER-X normalization 缺少 provenance.json：{provenance_path}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected_provenance = {
        "norm_stats_id": args.norm_stats_id,
        "contract_id": CONTRACT_ID,
        "robot_model": "piper_x",
    }
    mismatches = {
        key: {"expected": expected, "actual": provenance.get(key)}
        for key, expected in expected_provenance.items()
        if provenance.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"PiPER-X normalization provenance 不匹配：{mismatches}")
    normalization_purpose = provenance.get("purpose")
    execution_permitted = provenance.get("execution_permitted")
    if not isinstance(normalization_purpose, str) or not normalization_purpose.strip():
        raise RuntimeError("PiPER-X normalization provenance 缺少非空 purpose")
    if not isinstance(execution_permitted, bool):
        raise RuntimeError("PiPER-X normalization provenance 缺少布尔 execution_permitted")

    stats_path = args.norm_stats_dir / "norm_stats.json"
    if not stats_path.is_file():
        raise RuntimeError(f"PiPER-X normalization 缺少 norm_stats.json：{stats_path}")
    actual_stats_sha256 = hashlib.sha256(stats_path.read_bytes()).hexdigest()
    if provenance.get("norm_stats_sha256") != actual_stats_sha256:
        raise RuntimeError("norm_stats.json 与 provenance 记录的 SHA-256 不匹配")

    _verify_openpi_commit()
    stats = normalize.load(args.norm_stats_dir)
    if set(stats) != {"state", "actions"}:
        raise RuntimeError(f"norm stats 必须且只能包含 state/actions，实际为 {set(stats)}")
    train_config = create_config(
        args.norm_stats_id,
        normalization_purpose=normalization_purpose,
        execution_permitted=execution_permitted,
    )
    policy = policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        norm_stats=stats,
        default_prompt=args.default_prompt,
    )
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=train_config.policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
