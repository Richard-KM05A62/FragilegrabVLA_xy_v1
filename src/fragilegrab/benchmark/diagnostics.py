"""PiPER-X 控制映射与 π0.5 episode 的低成本诊断。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .environment import MujocoQualificationEnv
from .pi05_contract import OPENPI_INTERFACE_BASE_COMMIT, PiperPi05Contract


def _quantile_transform(
    value: np.ndarray, q01: np.ndarray, q99: np.ndarray, *, inverse: bool = False
) -> np.ndarray:
    """复现锁定 OpenPI transforms.Normalize/Unnormalize 的 quantile 分支。"""
    scale = q99 - q01 + 1e-6
    if inverse:
        return (value + 1.0) / 2.0 * scale + q01
    return (value - q01) / scale * 2.0 - 1.0


def audit_piper_transform_alignment(
    *,
    state: np.ndarray,
    norm_stats: dict[str, Any],
    norm_stats_id: str,
    normalization_purpose: str,
    joint_probe_rad: float,
    gripper_closures: list[float],
) -> dict[str, Any]:
    """逐维审计 Piper delta/absolute、quantile normalization、padding 和夹爪方向。"""
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError("state 必须是有限的七维向量")
    if not joint_probe_rad > 0:
        raise ValueError("joint_probe_rad 必须大于零")

    quantiles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key in ("state", "actions"):
        try:
            q01 = np.asarray(norm_stats[key]["q01"], dtype=np.float64)
            q99 = np.asarray(norm_stats[key]["q99"], dtype=np.float64)
        except (KeyError, TypeError) as exc:
            raise ValueError(f"norm stats 缺少 {key}.q01/q99") from exc
        if q01.shape != (7,) or q99.shape != (7,) or not np.isfinite([q01, q99]).all():
            raise ValueError(f"{key}.q01/q99 必须是有限的七维向量")
        if np.any(q99 <= q01):
            bad = np.flatnonzero(q99 <= q01).tolist()
            raise ValueError(f"{key} quantile range 非正，维度：{bad}")
        quantiles[key] = q01, q99

    state_q01, state_q99 = quantiles["state"]
    action_q01, action_q99 = quantiles["actions"]
    normalized_state = _quantile_transform(state, state_q01, state_q99)
    padded_normalized_state = np.pad(normalized_state, (0, 25))
    # OpenPI Unnormalize 只反变换统计覆盖的前七维，padding 原样保留。
    restored_padded_state = np.r_[
        _quantile_transform(padded_normalized_state[:7], state_q01, state_q99, inverse=True),
        padded_normalized_state[7:],
    ]

    rows: list[dict[str, Any]] = []
    tolerance = 1e-9
    for index in range(6):
        cases = []
        normalized_values = []
        for signed_probe in (-joint_probe_rad, 0.0, joint_probe_rad):
            absolute = state.copy()
            absolute[index] += signed_probe
            pre_normalization = absolute.copy()
            pre_normalization[:6] -= state[:6]
            normalized = _quantile_transform(pre_normalization, action_q01, action_q99)
            padded_normalized = np.pad(normalized, (0, 25))
            restored_delta = np.r_[
                _quantile_transform(
                    padded_normalized[:7], action_q01, action_q99, inverse=True
                ),
                padded_normalized[7:],
            ]
            restored_absolute = restored_delta.copy()
            restored_absolute[:6] += restored_padded_state[:6]
            effective = restored_absolute[:7]
            error = effective - absolute
            normalized_values.append(float(normalized[index]))
            cases.append(
                {
                    "signed_probe_rad": signed_probe,
                    "absolute_target": absolute.tolist(),
                    "delta_before_normalization": pre_normalization.tolist(),
                    "normalized_target_dimension": float(normalized[index]),
                    "restored_delta": restored_delta[:7].tolist(),
                    "restored_absolute_after_slice": effective.tolist(),
                    "max_abs_roundtrip_error": float(np.max(np.abs(error))),
                    "direction_preserved": bool(
                        signed_probe == 0.0
                        or (effective[index] - state[index]) * signed_probe > 0
                    ),
                }
            )
        monotonic = normalized_values[0] < normalized_values[1] < normalized_values[2]
        rows.append(
            {
                "action_index": index,
                "joint": f"joint{index + 1}",
                "delta_masked": True,
                "normalized_monotonic_negative_to_positive": monotonic,
                "cases": cases,
                "passed": bool(
                    monotonic
                    and all(case["direction_preserved"] for case in cases)
                    and max(case["max_abs_roundtrip_error"] for case in cases) <= tolerance
                ),
            }
        )

    contract = PiperPi05Contract(norm_stats_id=norm_stats_id)
    gripper_rows = []
    for closure in gripper_closures:
        raw_action = np.zeros(7, dtype=np.float64)
        raw_action[6] = closure
        normalized = _quantile_transform(raw_action, action_q01, action_q99)
        restored = _quantile_transform(normalized, action_q01, action_q99, inverse=True)
        gripper_rows.append(
            {
                "closure": closure,
                "normalized": float(normalized[6]),
                "restored_closure": float(restored[6]),
                "actuator_target_m": float(contract.gripper_closure_to_m(closure)),
                "roundtrip_error": float(abs(restored[6] - closure)),
            }
        )
    gripper_monotonic = all(
        left["closure"] < right["closure"]
        and left["normalized"] < right["normalized"]
        and left["actuator_target_m"] > right["actuator_target_m"]
        for left, right in zip(gripper_rows, gripper_rows[1:])
    )
    padding_pass = bool(
        np.array_equal(restored_padded_state[7:], padded_normalized_state[7:])
        and restored_padded_state.shape == (32,)
    )
    passed = all(row["passed"] for row in rows) and gripper_monotonic and padding_pass
    status = (
        "transport_only_math_pass"
        if passed and normalization_purpose == "transport_only"
        else "passed" if passed else "failed"
    )
    return {
        "schema_version": "fragilegrab.piper_transform_alignment/0.1",
        "status": status,
        "openpi_reference_commit": OPENPI_INTERFACE_BASE_COMMIT,
        "normalization": "quantile_q01_q99",
        "normalization_purpose": normalization_purpose,
        "norm_stats_id": norm_stats_id,
        "state": state.tolist(),
        "transform_order": [
            "absolute_action_to_delta_for_joint1..joint6",
            "quantile_normalize_state_and_action",
            "pad_to_32",
            "quantile_unnormalize_state_and_action_first_7_dims",
            "delta_to_absolute_using_restored_state_for_joint1..joint6",
            "slice_first_7_effective_dims",
        ],
        "joint_alignment": rows,
        "gripper_alignment": {
            "delta_masked": False,
            "closure_direction": "closure increases while actuator meters decrease",
            "samples": gripper_rows,
            "passed": gripper_monotonic,
        },
        "padding": {"model_dim": 32, "effective_dim": 7, "passed": padding_pass},
        "limitation": (
            "该结果只验证给定统计量下的变换顺序、逐轴正负方向、padding 和夹爪映射；"
            "不验证统计量来自真实 Piper 数据、策略训练 action convention 与 Piper 一致，"
            "也不授权执行 transport-only 输出。"
        ),
    }


def run_robot_adapter_sanity(docs: dict[str, Any], robot_assets: Path) -> dict[str, Any]:
    """逐维探测与策略执行相同的 MuJoCo position-actuator 路径。"""
    env = MujocoQualificationEnv(docs, robot_assets)
    cfg = docs["benchmark"]["values"]
    try:
        dt_s = float(env.model.opt.timestep)
        probe_steps = round(cfg["gripper_probe_s"] / dt_s)
        rows = []
        for index, actuator_name in enumerate(env.robot["actuators"][:6]):
            env.reset(cfg["seed"])
            initial_qpos = env.data.qpos[env.qadr].copy()
            initial_ctrl = env.data.ctrl.copy()
            limits = env.model.actuator_ctrlrange[index]
            magnitude = float(cfg["joint_probe_delta_rad"])
            signed_delta = magnitude if initial_ctrl[index] + magnitude <= limits[1] else -magnitude
            target = initial_ctrl.copy()
            target[index] += signed_delta
            target = env.action_adapter.decode(env, target)
            for _ in range(probe_steps):
                env.step_physics(target)
            final_qpos = env.data.qpos[env.qadr].copy()
            observed_delta = float(final_qpos[index] - initial_qpos[index])
            rows.append(
                {
                    "action_index": index,
                    "actuator": actuator_name,
                    "joint": env.robot["arm_joints"][index],
                    "unit": "rad",
                    "command_semantics": "absolute_joint_position",
                    "ctrlrange": limits.tolist(),
                    "command_delta": signed_delta,
                    "observed_joint_delta": observed_delta,
                    "response_sign_matches": bool(observed_delta * signed_delta > 0),
                    "tracking_error": float(abs(final_qpos[index] - target[index])),
                    "max_other_joint_motion": float(
                        np.max(np.abs(np.delete(final_qpos[:6] - initial_qpos[:6], index)))
                    ),
                }
            )

        env.reset(cfg["seed"])
        gripper_limits = env.model.actuator_ctrlrange[6]
        open_target = env.data.ctrl.copy()
        open_target[6] = gripper_limits[1]
        for _ in range(probe_steps):
            env.step_physics(open_target)
        opened = env.data.qpos[env.qadr[-2:]].copy()
        close_target = env.data.ctrl.copy()
        close_target[6] = gripper_limits[0]
        for _ in range(probe_steps):
            env.step_physics(close_target)
        closed = env.data.qpos[env.qadr[-2:]].copy()
        rows.append(
            {
                "action_index": 6,
                "actuator": env.robot["actuators"][6],
                "joint": env.robot["finger_joints"],
                "unit": "m",
                "command_semantics": "absolute_left_finger_position_with_mimic_right_finger",
                "ctrlrange": gripper_limits.tolist(),
                "open_command": float(gripper_limits[1]),
                "close_command": float(gripper_limits[0]),
                "opened_finger_qpos": opened.tolist(),
                "closed_finger_qpos": closed.tolist(),
                "closure_direction": "decreasing action dimension closes gripper",
                "finger_symmetry_pass": bool(
                    abs(opened.sum()) <= cfg["gripper_tolerance_m"]
                    and abs(closed.sum()) <= cfg["gripper_tolerance_m"]
                ),
            }
        )
        arm_pass = all(
            row["response_sign_matches"]
            and row["tracking_error"] <= cfg["tracking_tolerance_rad"]
            for row in rows[:6]
        )
        return {
            "schema_version": "fragilegrab.robot_adapter_sanity/0.1",
            "status": "passed" if arm_pass and rows[6]["finger_symmetry_pass"] else "failed",
            "robot": "piper_x",
            "physics_dt_s": dt_s,
            "physics_rate_hz": 1.0 / dt_s,
            "control_steps_per_action": cfg["control_steps"],
            "control_rate_hz": 1.0 / (dt_s * cfg["control_steps"]),
            "adapter": "ScriptedActionAdapter",
            "mapping": rows,
            "limitation": "验证 MuJoCo actuator 映射和响应；不证明 π0.5 action/norm stats 正确或真机标定一致。",
        }
    finally:
        env.close()


def analyze_pi05_timeline(path: Path) -> dict[str, Any]:
    """分析闭环 episode 已记录的 chunk、执行前缀和边界连续性。"""
    path = Path(path)
    episode = path if path.is_dir() else path.parent
    timeline = episode / "timeline.jsonl" if path.is_dir() else path
    audit_path = episode / "model_audit.json"
    if not timeline.is_file() or not audit_path.is_file():
        raise FileNotFoundError("需要 timeline.jsonl 和同目录 model_audit.json")
    limits = np.asarray(json.loads(audit_path.read_text(encoding="utf-8"))["actuator_ctrlrange"])
    latest_state = None
    queries: list[dict[str, Any]] = []
    executions: dict[str, list[dict[str, Any]]] = {}
    with timeline.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("record_type") == "observation":
                latest_state = row.get("joint_position")
            elif row.get("record_type") == "policy_query":
                predicted = np.asarray(row["predicted_action_chunk"], dtype=np.float64)
                decoded = np.asarray(row["decoded_actuator_chunk"], dtype=np.float64)
                if predicted.shape != (50, 7) or decoded.shape != (50, 7):
                    raise ValueError(f"query {row.get('query_id')} chunk shape 不是 (50,7)")
                queries.append(
                    {
                        "query_id": row["query_id"],
                        "simulation_time_s": row["simulation_time_s"],
                        "client_round_trip_ms": row.get("client_round_trip_ms"),
                        "joint_state_at_request": latest_state,
                        "predicted_action_chunk": predicted,
                        "decoded_actuator_chunk": decoded,
                    }
                )
            elif row.get("record_type") == "execution":
                executions.setdefault(row["query_id"], []).append(row)
    if not queries:
        raise ValueError("timeline 中没有 policy_query")

    summaries = []
    previous_executed = None
    for query in queries:
        predicted = query.pop("predicted_action_chunk")
        decoded = query.pop("decoded_actuator_chunk")
        within_chunk = np.diff(decoded, axis=0)
        executed_rows = executions.get(query["query_id"], [])
        executed = np.asarray([row["executed_action"] for row in executed_rows], dtype=np.float64)
        boundary_jump = None if previous_executed is None else decoded[0] - previous_executed
        if len(executed):
            previous_executed = executed[-1]
        summaries.append(
            {
                **query,
                "output_transform_semantics": "seven effective dimensions after Piper inverse normalization and robot-specific absolute-action transform; slicing removes model padding",
                "post_output_transform_action_min": predicted.min(axis=0).tolist(),
                "post_output_transform_action_max": predicted.max(axis=0).tolist(),
                "decoded_min": decoded.min(axis=0).tolist(),
                "decoded_max": decoded.max(axis=0).tolist(),
                "max_abs_within_chunk_step": np.max(np.abs(within_chunk), axis=0).tolist(),
                "range_violation_count": int(
                    np.count_nonzero((decoded < limits[:, 0]) | (decoded > limits[:, 1]))
                ),
                "gripper_closure_min": float(predicted[:, 6].min()),
                "gripper_closure_max": float(predicted[:, 6].max()),
                "executed_count": int(len(executed_rows)),
                "executed_actions": executed.tolist(),
                "first_proposal_minus_previous_executed": (
                    None if boundary_jump is None else boundary_jump.tolist()
                ),
            }
        )
    return {
        "schema_version": "fragilegrab.pi05_trace_inspection/0.1",
        "source_timeline": timeline.name,
        "query_count": len(summaries),
        "normalized_raw_output_available": False,
        "normalized_raw_output_note": "锁定 OpenPI WebSocket response 只暴露 robot-specific output transform 后的七个有效维度；末尾 [:7] 是 padding 去除，不是当前风险项。协议不暴露 transform 前内部 normalized tensor。",
        "queries": summaries,
    }
