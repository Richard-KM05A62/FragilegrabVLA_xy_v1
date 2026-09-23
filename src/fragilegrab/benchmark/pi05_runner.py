"""π0.5 → PiPER MuJoCo 的同步 receding-horizon 最小闭环。"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import platform
import subprocess
import time
from typing import Any

import cv2
import mujoco
import numpy as np
import yaml

from .environment import MujocoQualificationEnv, load_configs, sha256
from .evaluator import GraspTendencyEvaluator, SafetyEvaluator
from .pi05_contract import CONTRACT_ID, OPENPI_COMMIT, OPENPI_INTERFACE_BASE_COMMIT, PiperPi05Contract
from .runner import model_audit, native, save_json
from .video import export_webm


def load_rollout_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != "fragilegrab.pi05_rollout/0.1":
        raise ValueError("π0.5 rollout config schema_version 不符")
    required_equal = {
        "openpi_commit": OPENPI_COMMIT,
        "openpi_interface_base_commit": OPENPI_INTERFACE_BASE_COMMIT,
        "contract_id": CONTRACT_ID,
        "model": "pi05_base",
        "model_action_dim": 32,
        "action_horizon": 50,
        "max_token_len": 200,
        "sample_steps": 10,
    }
    for key, expected in required_equal.items():
        if config.get(key) != expected:
            raise ValueError(f"{key} 必须是 {expected!r}")
    for key in ("norm_stats_id", "prompt"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"{key} 必须是非空字符串")
    PiperPi05Contract(config["norm_stats_id"])
    for key in ("execution_horizon", "max_policy_queries", "control_steps_per_action"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f"{key} 必须是正整数")
    if config["execution_horizon"] > config["action_horizon"]:
        raise ValueError("execution_horizon 不能大于 action_horizon")
    for key in ("max_joint_delta_rad", "max_gripper_delta_m"):
        if not np.isfinite(config.get(key)) or config[key] <= 0:
            raise ValueError(f"{key} 必须是正有限数")
    mode = config.get("execution_mode", "strict_reject")
    if mode not in {"strict_reject", "diagnostic_projection"}:
        raise ValueError("execution_mode 必须是 strict_reject 或 diagnostic_projection")
    config["execution_mode"] = mode
    if mode == "diagnostic_projection":
        ceiling = config.get("diagnostic_stop_contact_force_N")
        if (
            isinstance(ceiling, bool)
            or not isinstance(ceiling, (int, float))
            or not np.isfinite(ceiling)
            or ceiling <= 0
        ):
            raise ValueError("diagnostic_projection 需要正有限 diagnostic_stop_contact_force_N")
        if type(config.get("stop_on_unintended_collision")) is not bool:
            raise ValueError("diagnostic_projection 需要 bool stop_on_unintended_collision")
    if config.get("task") != "F1":
        raise ValueError("当前 π0.5 v0.1 执行器只验证 F1；qualification 阶段判定尚未接入 policy rollout")
    if type(config.get("stop_on_task_success")) is not bool:
        raise ValueError("stop_on_task_success 必须是 bool")
    return config


def project_diagnostic_target(
    previous: np.ndarray,
    proposed: np.ndarray,
    actuator_ctrlrange: np.ndarray,
    *,
    max_joint_delta_rad: float,
    max_gripper_delta_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """把不可执行提议显式投影到 MuJoCo 诊断边界，并返回逐项证据。"""
    previous = np.asarray(previous, dtype=np.float64)
    proposed = np.asarray(proposed, dtype=np.float64)
    limits = np.asarray(actuator_ctrlrange, dtype=np.float64)
    if previous.shape != (7,) or proposed.shape != (7,) or limits.shape != (7, 2):
        raise ValueError("诊断投影要求 previous/proposed=(7,) 且 ctrlrange=(7,2)")
    if not np.isfinite(previous).all() or not np.isfinite(proposed).all() or not np.isfinite(limits).all():
        raise ValueError("诊断投影输入必须全部有限")

    range_bounded = np.clip(proposed, limits[:, 0], limits[:, 1])
    step_limit = np.array([max_joint_delta_rad] * 6 + [max_gripper_delta_m], dtype=np.float64)
    projected = previous + np.clip(range_bounded - previous, -step_limit, step_limit)
    projected = np.clip(projected, limits[:, 0], limits[:, 1])
    range_changed = np.abs(range_bounded - proposed) > 1e-12
    rate_changed = np.abs(projected - range_bounded) > 1e-12
    return projected, {
        "mode": "diagnostic_projection",
        "range_changed_dims": np.flatnonzero(range_changed).tolist(),
        "rate_changed_dims": np.flatnonzero(rate_changed).tolist(),
        "changed": bool(np.any(range_changed | rate_changed)),
        "max_abs_correction": float(np.max(np.abs(projected - proposed))),
    }


def _display_observation(observation: dict[str, Any], status: str) -> bool:
    panels = []
    for key, label in (("base_rgb", "base_0_rgb"), ("wrist_rgb", "left_wrist_0_rgb")):
        bgr = cv2.cvtColor(observation["images"][key], cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        panels.append(bgr)
    view = np.concatenate(panels, axis=1)
    cv2.putText(view, status, (8, view.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("FragileGrab π0.5 MuJoCo diagnostic | Q/Esc stop", view)
    return cv2.waitKey(1) & 0xFF in (ord("q"), 27)


def _git(repo: Path, *args: str) -> str | None:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def run_pi05_episode(
    benchmark_config_dir: Path,
    rollout_config_path: Path,
    robot_assets: Path,
    output: Path,
    policy: Any,
    *,
    seed: int | None = None,
    display: bool = False,
    object_config: Path | None = None,
) -> dict[str, Any]:
    docs = load_configs(benchmark_config_dir, object_config=object_config)
    rollout = load_rollout_config(rollout_config_path)
    benchmark = docs["benchmark"]["values"]
    if seed is None:
        seed = benchmark["seed"]
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed 必须为 uint32 整数")

    contract = PiperPi05Contract(rollout["norm_stats_id"])
    contract.validate_server_metadata(policy.metadata)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / ".incomplete").touch()
    (output / "streams/base_rgb").mkdir(parents=True)
    (output / "streams/wrist_rgb").mkdir(parents=True)
    env = None
    video = None
    wrist_video = None
    try:
        env = MujocoQualificationEnv(docs, robot_assets)
        if rollout["control_steps_per_action"] != benchmark["control_steps"]:
            raise ValueError("rollout 与 benchmark control_steps 不一致")
        observation = env.reset(seed)
        diagnostic_doc = docs.get("grasp_diagnostic")
        if not diagnostic_doc:
            raise ValueError("缺少 grasp_diagnostic.yaml")
        grasp_diagnostic = GraspTendencyEvaluator(
            diagnostic_doc["values"],
            {
                "object_position_m": env.data.xpos[env.target_id].copy(),
                "grasp_position_m": env.grasp_position().copy(),
            },
            env.target_id,
            env.finger_ids,
        )
        audit = model_audit(env)
        audit["render_nonconstant"] = bool(observation["images"]["base_rgb"].std() > 0)
        audit["wrist_render_nonconstant"] = bool(observation["images"]["wrist_rgb"].std() > 0)
        save_json(output / "model_audit.json", audit)
        save_json(output / "benchmark_config.json", docs)
        (output / "pi05_rollout_config.yaml").write_text(
            Path(rollout_config_path).read_text(encoding="utf-8"), encoding="utf-8"
        )
        (output / "scene.xml").write_text(env.scene_xml, encoding="utf-8")

        dt_s = env.model.opt.timestep
        h, w = benchmark["resolution_hw"]
        fps = 1.0 / (dt_s * rollout["control_steps_per_action"])
        video = cv2.VideoWriter(
            str(output / "base_rgb.mp4"),
            cv2.VideoWriter_fourcc(*benchmark["video_fourcc"]),
            fps,
            (w, h),
        )
        if not video.isOpened():
            raise RuntimeError("视频编码器未打开")
        wrist_video = cv2.VideoWriter(
            str(output / "wrist_rgb.mp4"),
            cv2.VideoWriter_fourcc(*benchmark["video_fourcc"]),
            fps,
            (w, h),
        )
        if not wrist_video.isOpened():
            raise RuntimeError("腕部视频编码器未打开")

        allowed = {
            tuple(sorted((env.model.body("base_link").id, env.model.body("table").id))),
            tuple(sorted(env.finger_ids)),
        }
        evaluator = SafetyEvaluator(
            benchmark,
            env.initial_object,
            env.target_id,
            env.finger_ids,
            env.robot_bodies,
            allowed,
            rollout["task"],
        )
        trajectory_hash = hashlib.sha256()
        rgb_hashes = {"base_rgb": hashlib.sha256(), "wrist_rgb": hashlib.sha256()}
        frames = 0
        physics_steps = 0
        actions_executed = 0
        query_count = 0
        previous = env.data.ctrl.copy()
        projected_action_count = 0
        max_projection_correction = 0.0
        termination_reason = None

        timeline_path = output / "timeline.jsonl"
        with timeline_path.open("w", encoding="utf-8") as log:
            def record(row: dict[str, Any]) -> None:
                log.write(json.dumps(native(row), ensure_ascii=False, allow_nan=False) + "\n")

            def record_observation(obs: dict[str, Any], query_id: str | None = None) -> None:
                nonlocal frames
                image_refs = {}
                for name, writer in (("base_rgb", video), ("wrist_rgb", wrist_video)):
                    rgb = obs["images"][name]
                    image_ref = f"streams/{name}/{frames:06d}.npy"
                    np.save(output / image_ref, rgb)
                    rgb_hashes[name].update(rgb.tobytes())
                    writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    image_refs[name] = image_ref
                state_row = {key: value for key, value in obs.items() if key != "images"}
                record({
                    "record_type": "observation",
                    "episode_physics_step": physics_steps,
                    "query_id": query_id,
                    "image_refs": image_refs,
                    **state_row,
                })
                frames += 1

            record_observation(observation)
            if display and _display_observation(observation, "reset"):
                termination_reason = "user_stop"
            for query_index in range(rollout["max_policy_queries"]):
                if termination_reason is not None:
                    break
                query_id = f"query-{query_index:06d}"
                request = contract.encode_observation(observation, rollout["prompt"])
                request_started_ns = time.monotonic_ns()
                response = policy.infer(request)
                response_received_ns = time.monotonic_ns()
                if rollout["execution_mode"] == "strict_reject":
                    decoded = contract.decode_action_chunk(response, env.model.actuator_ctrlrange)
                else:
                    decoded = contract.decode_action_chunk_proposal(response)
                record({
                    "record_type": "policy_query",
                    "query_id": query_id,
                    "simulation_time_s": float(env.data.time),
                    "request_started_monotonic_ns": request_started_ns,
                    "response_received_monotonic_ns": response_received_ns,
                    "client_round_trip_ms": (response_received_ns - request_started_ns) / 1e6,
                    "predicted_action_chunk": np.asarray(response["actions"]),
                    "decoded_actuator_chunk": decoded,
                    "chunk_horizon": contract.action_horizon,
                    "planned_execution_count": rollout["execution_horizon"],
                    "execution_mode": rollout["execution_mode"],
                    "action_semantics_ref": CONTRACT_ID,
                    "server_timing": response.get("server_timing"),
                    "policy_timing": response.get("policy_timing"),
                })
                query_count += 1
                executed_in_query = 0

                # Validate the complete prefix before submitting any part of it.
                # This keeps strict-reject queries atomic: an invalid later action
                # cannot leave a partially executed prefix in the simulator.
                if rollout["execution_mode"] == "strict_reject":
                    validation_previous = previous.copy()
                    for proposed in decoded[: rollout["execution_horizon"]]:
                        contract.validate_execution_step(
                            validation_previous,
                            proposed,
                            max_joint_delta_rad=rollout["max_joint_delta_rad"],
                            max_gripper_delta_m=rollout["max_gripper_delta_m"],
                        )
                        validation_previous = proposed

                for chunk_index, proposed in enumerate(decoded[: rollout["execution_horizon"]]):
                    if rollout["execution_mode"] == "strict_reject":
                        target = proposed.copy()
                        projection = {"mode": "strict_reject", "changed": False,
                                      "range_changed_dims": [], "rate_changed_dims": [],
                                      "max_abs_correction": 0.0}
                    else:
                        target, projection = project_diagnostic_target(
                            previous,
                            proposed,
                            env.model.actuator_ctrlrange,
                            max_joint_delta_rad=rollout["max_joint_delta_rad"],
                            max_gripper_delta_m=rollout["max_gripper_delta_m"],
                        )
                    if projection["changed"]:
                        projected_action_count += 1
                    max_projection_correction = max(
                        max_projection_correction, projection["max_abs_correction"]
                    )
                    target = env.action_adapter.decode(env, target)
                    record({
                        "record_type": "execution",
                        "query_id": query_id,
                        "chunk_index": chunk_index,
                        "simulation_time_s": float(env.data.time),
                        "policy_proposed_action": proposed,
                        "requested_action": target,
                        "executed_action": target,
                        "projection": projection,
                        "dispatch_status": "submitted_to_mujoco_position_actuators",
                    })
                    actions_executed += 1
                    executed_in_query += 1
                    for _ in range(rollout["control_steps_per_action"]):
                        state = env.step_physics(target)
                        evaluator.update(state, "policy", dt_s)
                        grasp_diagnostic.update(state, dt_s)
                        trajectory_hash.update(np.asarray(state["qpos"]).tobytes())
                        trajectory_hash.update(np.asarray(state["qvel"]).tobytes())
                        record({
                            "record_type": "contact",
                            "episode_physics_step": physics_steps,
                            "query_id": query_id,
                            "chunk_index": chunk_index,
                            **state,
                        })
                        physics_steps += 1
                        snapshot = evaluator.result()
                        if (rollout["execution_mode"] == "diagnostic_projection"
                                and snapshot["peak_contact_force_N"]
                                > rollout["diagnostic_stop_contact_force_N"]):
                            termination_reason = "diagnostic_contact_force_ceiling"
                            break
                        if (rollout["execution_mode"] == "diagnostic_projection"
                                and rollout["stop_on_unintended_collision"]
                                and snapshot["unintended_collision_events"] > 0):
                            termination_reason = "diagnostic_unintended_collision"
                            break
                    previous = target.copy()
                    mujoco.mj_forward(env.model, env.data)
                    observation = env.observe()
                    record_observation(observation, query_id)
                    if display and _display_observation(
                        observation,
                        f"query={query_index} step={chunk_index} projected={projection['changed']}",
                    ):
                        termination_reason = "user_stop"
                    if termination_reason is not None:
                        record({
                            "record_type": "diagnostic_stop",
                            "query_id": query_id,
                            "chunk_index": chunk_index,
                            "reason": termination_reason,
                        })
                        break

                if executed_in_query < contract.action_horizon:
                    record({
                        "record_type": "chunk_cancelled",
                        "query_id": query_id,
                        "first_cancelled_index": executed_in_query,
                        "cancelled_count": contract.action_horizon - executed_in_query,
                        "reason": termination_reason or "receding_horizon_replan",
                    })
                if termination_reason is not None:
                    break
                if rollout["stop_on_task_success"] and evaluator.result()["task_success"]:
                    termination_reason = "task_success"
                    break

        video.release()
        video = None
        wrist_video.release()
        wrist_video = None
        preview = export_webm(output, frames, (h, w), fps, benchmark["preview_fourcc"])
        result = evaluator.result()
        result["grasp_tendency"] = grasp_diagnostic.result()
        if termination_reason is None:
            termination_reason = "max_policy_queries"
        result["policy_rollout"] = {
            "query_count": query_count,
            "actions_executed": actions_executed,
            "physics_steps": physics_steps,
            "execution_horizon": rollout["execution_horizon"],
            "execution_mode": rollout["execution_mode"],
            "projected_action_count": projected_action_count,
            "projection_fraction": (
                projected_action_count / actions_executed if actions_executed else 0.0
            ),
            "max_projection_correction": max_projection_correction,
            "termination": termination_reason,
        }
        save_json(output / "metrics.json", result)

        repo = Path(__file__).resolve().parents[3]
        manifest = {
            "episode_schema_version": "fragilegrab.episode/0.1",
            "profile": "mujoco_pi05_closed_loop/0.1",
            "episode_id": output.name,
            "source": {"kind": "simulation", "simulator": "mujoco"},
            "robot": {"name": "piper_x", "source": "AgileX official pinned agx_arm_sim PiPER-X+D435 URDF/USD"},
            "task": {"id": rollout["task"], "instruction": rollout["prompt"]},
            "seed": seed,
            "policy": {
                "name": "pi05_base",
                "openpi_commit": OPENPI_COMMIT,
                "contract_id": CONTRACT_ID,
                "norm_stats_id": rollout["norm_stats_id"],
                "server_metadata": policy.metadata,
            },
            "execution": {
                "action_horizon": 50,
                "execution_horizon": rollout["execution_horizon"],
                "control_steps_per_action": rollout["control_steps_per_action"],
                "control_rate_hz": fps,
                "replanning": "synchronous_receding_horizon",
                "mode": rollout["execution_mode"],
            },
            "streams": {
                "base_rgb": {"shape": [h, w, 3], "dtype": "uint8", "fps": fps},
                "wrist_rgb": {"shape": [h, w, 3], "dtype": "uint8", "fps": fps},
                "robot_state": {"names": env.joint_names, "units": ["rad"] * 6 + ["m"] * 2},
            },
            "provenance": {
                **env.provenance,
                "code_commit": _git(repo, "rev-parse", "HEAD"),
                "dirty": _git(repo, "status", "--porcelain"),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
            },
            "result": result,
            "termination": {"status": "complete", "reason": result["policy_rollout"]["termination"]},
            "artifacts": {
                "timeline": "timeline.jsonl",
                "video": "base_rgb.mp4",
                "wrist_video": "wrist_rgb.mp4",
                "preview_video": "base_rgb.webm",
                "metrics": "metrics.json",
                "audit": "model_audit.json",
            },
            "integrity": {
                "physics_steps": physics_steps,
                "frames": frames,
                "queries": query_count,
                "actions_executed": actions_executed,
                "trajectory_sha256": trajectory_hash.hexdigest(),
                "raw_base_rgb_sha256": rgb_hashes["base_rgb"].hexdigest(),
                "raw_wrist_rgb_sha256": rgb_hashes["wrist_rgb"].hexdigest(),
                "timeline_sha256": sha256(timeline_path),
                "preview_video_sha256": preview["sha256"],
            },
        }
        save_json(output / "episode_manifest.json", manifest)
        (output / ".incomplete").unlink()
        return manifest
    except BaseException as exc:
        save_json(output / "failure.json", {"status": "failed", "error": str(exc)})
        raise
    finally:
        if display:
            cv2.destroyAllWindows()
        if video is not None:
            video.release()
        if wrist_video is not None:
            wrist_video.release()
        if env is not None:
            env.close()
