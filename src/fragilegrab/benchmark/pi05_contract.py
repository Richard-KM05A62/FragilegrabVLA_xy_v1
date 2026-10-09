"""PiPER MuJoCo 与锁定 OpenPI π0.5 之间的显式数据契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


OPENPI_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"
OPENPI_INTERFACE_BASE_COMMIT = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
CONTRACT_ID = "piper_x_joint_position_dual_rgb_v0.3"
NORM_STATS_ID_PREFIX = "piper_x_"
ACTION_SEMANTICS = "absolute_joint_position_rad_plus_gripper_closure_0_1"


class Pi05ContractError(ValueError):
    """观测、模型输出或服务元数据不满足 PiPER π0.5 契约。"""


@dataclass(frozen=True)
class PiperPi05Contract:
    """七维策略契约：六轴绝对关节角 + 归一化夹爪闭合度。"""

    norm_stats_id: str
    action_horizon: int = 50
    gripper_open_m: float = 0.05
    gripper_closed_m: float = 0.0
    finger_symmetry_tolerance_m: float = 0.003
    observed_gripper_bound_tolerance_m: float = 0.001

    def __post_init__(self) -> None:
        if not self.norm_stats_id:
            raise Pi05ContractError("norm_stats_id 不能为空")
        if not self.norm_stats_id.startswith(NORM_STATS_ID_PREFIX):
            raise Pi05ContractError(
                f"PiPER-X norm_stats_id 必须以 {NORM_STATS_ID_PREFIX!r} 开头"
            )
        if self.action_horizon not in {20, 50}:
            raise Pi05ContractError("当前只允许 LoRA 20-step 或 pi05_base 50-step action horizon")
        if not self.gripper_open_m > self.gripper_closed_m:
            raise Pi05ContractError("夹爪 open/closed 米制范围无效")

    @property
    def metadata_expectation(self) -> dict[str, Any]:
        return {
            "openpi_commit": OPENPI_COMMIT,
            "openpi_interface_base_commit": OPENPI_INTERFACE_BASE_COMMIT,
            "contract_id": CONTRACT_ID,
            "norm_stats_id": self.norm_stats_id,
            "action_horizon": self.action_horizon,
            "effective_action_dim": 7,
            "action_semantics": ACTION_SEMANTICS,
            "image_roles": {
                "base_0_rgb": "present",
                "left_wrist_0_rgb": "present",
                "right_wrist_0_rgb": "masked",
            },
        }

    def validate_server_metadata(
        self,
        metadata: dict[str, Any],
        *,
        require_execution: bool = True,
    ) -> None:
        if not isinstance(metadata, dict):
            raise Pi05ContractError("policy server metadata 必须是字典")
        mismatches = {
            key: {"expected": expected, "actual": metadata.get(key)}
            for key, expected in self.metadata_expectation.items()
            if metadata.get(key) != expected
        }
        if mismatches:
            raise Pi05ContractError(f"policy server 契约不匹配：{mismatches}")
        if not isinstance(metadata.get("normalization_purpose"), str) or not metadata[
            "normalization_purpose"
        ].strip():
            raise Pi05ContractError("policy server metadata 缺少 normalization_purpose")
        if not isinstance(metadata.get("execution_permitted"), bool):
            raise Pi05ContractError("policy server metadata 缺少布尔 execution_permitted")
        if require_execution and metadata["execution_permitted"] is not True:
            raise Pi05ContractError("当前 normalization 仅允许传输测试，拒绝用于动作执行")

    def gripper_m_to_closure(self, value_m: np.ndarray | float) -> np.ndarray:
        value = np.asarray(value_m, dtype=np.float64)
        lower = self.gripper_closed_m - self.observed_gripper_bound_tolerance_m
        upper = self.gripper_open_m + self.observed_gripper_bound_tolerance_m
        if np.any(value < lower) or np.any(value > upper):
            raise Pi05ContractError(f"夹爪观测位置越界：{value_m}")
        value = np.clip(value, self.gripper_closed_m, self.gripper_open_m)
        closure = (self.gripper_open_m - value) / (self.gripper_open_m - self.gripper_closed_m)
        return closure

    def gripper_closure_to_m(self, closure: np.ndarray | float) -> np.ndarray:
        value = np.asarray(closure, dtype=np.float64)
        if np.any(value < -1e-6) or np.any(value > 1.0 + 1e-6):
            raise Pi05ContractError("模型夹爪闭合度必须在 [0, 1]")
        value = np.clip(value, 0.0, 1.0)
        return self.gripper_open_m + value * (self.gripper_closed_m - self.gripper_open_m)

    def policy_state(self, joint_position: np.ndarray) -> np.ndarray:
        """把六轴加两指 MuJoCo qpos 转成七维 π0.5 state。"""
        joints = np.asarray(joint_position, dtype=np.float64)
        if joints.shape != (8,) or not np.isfinite(joints).all():
            raise Pi05ContractError("PiPER joint_position 必须是有限的八维向量")
        if abs(joints[6] + joints[7]) > self.finger_symmetry_tolerance_m:
            raise Pi05ContractError("PiPER 两指位置不满足 joint8=-joint7 对称约束")
        return np.r_[joints[:6], self.gripper_m_to_closure(joints[6])]

    def policy_action(self, actuator_target: np.ndarray) -> np.ndarray:
        """把 MuJoCo 七 actuator 绝对目标转成策略训练/统计契约。"""
        action = np.asarray(actuator_target, dtype=np.float64)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise Pi05ContractError("PiPER actuator target 必须是有限的七维向量")
        return np.r_[action[:6], self.gripper_m_to_closure(action[6])]

    def encode_observation(self, observation: dict[str, Any], prompt: str) -> dict[str, Any]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise Pi05ContractError("prompt 必须是非空字符串")
        try:
            image = np.asarray(observation["images"]["base_rgb"])
            wrist_image = np.asarray(observation["images"]["wrist_rgb"])
            joint_position = observation["joint_position"]
        except (KeyError, TypeError) as exc:
            raise Pi05ContractError("观测缺少 base_rgb、wrist_rgb 或 joint_position") from exc
        for name, value in (("base_rgb", image), ("wrist_rgb", wrist_image)):
            if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
                raise Pi05ContractError(f"{name} 必须是 HWC uint8 RGB")
        if wrist_image.shape != image.shape:
            raise Pi05ContractError("全局和腕部 RGB 尺寸必须一致")
        return {
            "observation/image": image,
            "observation/wrist_image": wrist_image,
            "observation/state": self.policy_state(joint_position),
            "prompt": prompt.strip(),
        }

    def decode_action_chunk(self, response: dict[str, Any], actuator_ctrlrange: np.ndarray) -> np.ndarray:
        decoded = self.decode_action_chunk_proposal(response)
        actions = np.asarray(response["actions"], dtype=np.float64)
        self.gripper_closure_to_m(actions[:, 6])
        limits = np.asarray(actuator_ctrlrange, dtype=np.float64)
        if limits.shape != (7, 2) or not np.isfinite(limits).all():
            raise Pi05ContractError("actuator_ctrlrange 必须是有限的 (7, 2)")
        if np.any(decoded < limits[:, 0] - 1e-9) or np.any(decoded > limits[:, 1] + 1e-9):
            raise Pi05ContractError("action chunk 越过 MuJoCo actuator ctrlrange；拒绝静默裁剪")
        return decoded

    def decode_action_chunk_proposal(self, response: dict[str, Any]) -> np.ndarray:
        """解码模型原始提议，允许诊断 runner 随后显式记录并投影越界值。

        该入口不代表动作可执行；调用方必须另行检查执行器范围和单步变化。
        """
        if not isinstance(response, dict) or "actions" not in response:
            raise Pi05ContractError("policy response 缺少 actions")
        actions = np.asarray(response["actions"], dtype=np.float64)
        if actions.shape != (self.action_horizon, 7) or not np.isfinite(actions).all():
            raise Pi05ContractError(
                f"actions 必须是有限的 ({self.action_horizon}, 7)，实际为 {actions.shape}"
            )
        decoded = actions.copy()
        decoded[:, 6] = self.gripper_open_m + actions[:, 6] * (
            self.gripper_closed_m - self.gripper_open_m
        )
        return decoded

    def validate_execution_step(
        self,
        previous: np.ndarray,
        target: np.ndarray,
        *,
        max_joint_delta_rad: float,
        max_gripper_delta_m: float,
    ) -> None:
        previous = np.asarray(previous, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if previous.shape != (7,) or target.shape != (7,):
            raise Pi05ContractError("执行目标必须是七维 actuator target")
        delta = np.abs(target - previous)
        if np.any(delta[:6] > max_joint_delta_rad + 1e-12):
            raise Pi05ContractError(f"关节单步变化过大：{delta[:6].tolist()}")
        if delta[6] > max_gripper_delta_m + 1e-12:
            raise Pi05ContractError(f"夹爪单步变化过大：{float(delta[6])}")
