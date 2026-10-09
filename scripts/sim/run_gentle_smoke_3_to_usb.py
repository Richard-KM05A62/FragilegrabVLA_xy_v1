#!/usr/bin/env python3
"""Generate a minimally validated provisional scripted PiPER-X LeRobot dataset."""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import cv2
import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fragilegrab.benchmark.environment import MujocoQualificationEnv, load_configs
from fragilegrab.benchmark.pi05_contract import PiperPi05Contract
from fragilegrab.benchmark.runner import save_json
from fragilegrab.benchmark.scripted_quality import (
    SCRIPTED_TASK_TYPES,
    summarize_episode_metrics,
    validate_episode_structure,
    validate_scripted_dataset_config,
)


STATE_NAMES = [*[f"joint_{i}_rad" for i in range(1, 7)], "gripper_closedness"]
ACTION_NAMES = [*[f"joint_{i}_target_rad" for i in range(1, 7)], "gripper_target_closedness"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_usb(uuid: str | None, explicit: Path | None) -> Path:
    if explicit is not None:
        result = explicit.expanduser().resolve()
        if not result.is_dir():
            raise RuntimeError(f"USB mount does not exist: {result}")
        return result
    data = json.loads(subprocess.run(
        ["lsblk", "--json", "--output", "UUID,MOUNTPOINTS"],
        check=True, capture_output=True, text=True,
    ).stdout)
    matches: list[Path] = []

    def visit(device: dict[str, Any]) -> None:
        if device.get("uuid") == uuid:
            matches.extend(Path(x) for x in device.get("mountpoints") or [] if x)
        for child in device.get("children") or []:
            visit(child)

    for device in data.get("blockdevices", []):
        visit(device)
    unique = sorted({path.resolve() for path in matches})
    if len(unique) != 1:
        raise RuntimeError(f"USB UUID {uuid!r} resolved to {unique}")
    return unique[0]


def verify_writable(path: Path) -> None:
    probe = path / f".fragilegrab-write-test-{os.getpid()}"
    try:
        probe.write_text("ok\n", encoding="utf-8")
    finally:
        probe.unlink(missing_ok=True)


def rotation_about_world_z(base: np.ndarray, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return rotation @ np.asarray(base, dtype=np.float64)


def set_free_joint_pose(env: MujocoQualificationEnv, body_name: str, xy: list[float], yaw: float) -> None:
    joint = env.model.joint(body_name.replace("_main", "_joint0"))
    address = int(env.model.jnt_qposadr[joint.id])
    env.data.qpos[address] = float(xy[0])
    env.data.qpos[address + 1] = float(xy[1])
    env.data.qpos[address + 3 : address + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    env.data.qvel[int(env.model.jnt_dofadr[joint.id]) : int(env.model.jnt_dofadr[joint.id]) + 6] = 0.0


def create_lerobot_dataset(root: Path, cfg: dict[str, Any]):
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    h, w = cfg["resolution_hw"]
    image_feature = {"dtype": "video", "shape": (3, h, w), "names": ["channels", "height", "width"]}
    return LeRobotDataset.create(
        repo_id=cfg["dataset_repo_id"],
        fps=int(cfg["control_hz"]),
        root=root,
        robot_type="piper_x_mujoco_scripted",
        features={
            "observation.images.base": image_feature,
            "observation.images.wrist": image_feature,
            "observation.state": {"dtype": "float32", "shape": (7,), "names": [STATE_NAMES]},
            "action": {"dtype": "float32", "shape": (7,), "names": [ACTION_NAMES]},
        },
        use_videos=True,
    )


class Recorder:
    def __init__(
        self,
        env: MujocoQualificationEnv,
        output: Path,
        cfg: dict[str, Any],
        instruction: str,
        goal_position_m: np.ndarray | None,
        dataset=None,
    ):
        self.env, self.output, self.cfg = env, output, cfg
        self.instruction = instruction
        self.goal_position_m = goal_position_m
        self.dataset = dataset
        self.contract = PiperPi05Contract("piper_x_gentle_demo_smoke_v0_1")
        h, w = cfg["resolution_hw"]
        fps = cfg["control_hz"]
        fourcc = cv2.VideoWriter_fourcc(*cfg["video_fourcc"])
        suffix = cfg["video_extension"]
        self.video_paths = {
            "front": output / f"front.{suffix}",
            "wrist": output / f"wrist.{suffix}",
        }
        self.writers = {
            role: cv2.VideoWriter(str(path), fourcc, fps, (w, h))
            for role, path in self.video_paths.items()
        }
        if not all(writer.isOpened() for writer in self.writers.values()):
            self.close()
            raise RuntimeError(
                f"{cfg['video_fourcc']}/{cfg['video_extension']} review video writer failed to open"
            )
        self.samples_path = output / "samples.jsonl.gz"
        self.samples = gzip.open(self.samples_path, "wt", encoding="utf-8")
        self.frame = 0
        self.max_lift = 0.0
        self.peak_target_force = 0.0
        self.last_state: dict[str, Any] | None = None
        self.initial_object = env.data.xpos[env.target_id].copy()
        self.previous_action: np.ndarray | None = None
        self.previous_images: dict[str, np.ndarray] = {}
        self.max_joint_action_step_rad = 0.0
        self.max_gripper_action_step_closure = 0.0
        self.finite_state_action = True
        self.shape_valid = True
        self.action_within_hard_limits = True
        self.timestamps_s: list[float] = []
        self.stream_lengths = {name: 0 for name in ("front", "wrist", "state", "action", "timestamp")}
        self.tracking_error_samples: list[list[float]] = []
        self.bilateral_contact_physics_steps = 0
        self.lifted_control_ticks = 0
        self.image_accumulators = {
            role: {"std_sum": 0.0, "diff_sum": 0.0, "diff_count": 0}
            for role in ("front", "wrist")
        }

    def close(self) -> None:
        if hasattr(self, "samples") and self.samples is not None:
            self.samples.close()
            self.samples = None
        for writer in getattr(self, "writers", {}).values():
            writer.release()

    def control_tick(self, action: np.ndarray, phase: str) -> dict[str, Any]:
        observation = self.env.observe()
        images = observation.pop("images")
        policy_state = self.contract.policy_state(observation["joint_position"])
        policy_action = self.contract.policy_action(action)
        expected_shape = (*self.cfg["resolution_hw"], 3)
        for name in ("base_rgb", "wrist_rgb"):
            image = images[name]
            valid_image = image.dtype == np.uint8 and image.shape == expected_shape
            self.shape_valid = self.shape_valid and valid_image
            if not valid_image:
                raise RuntimeError(
                    f"{name} shape/dtype mismatch: expected uint8 {expected_shape}, "
                    f"got {image.dtype} {image.shape}"
                )
        finite = bool(np.isfinite(policy_state).all() and np.isfinite(policy_action).all())
        self.finite_state_action = self.finite_state_action and finite
        if not finite:
            raise RuntimeError("non-finite state/action in scripted demonstration")
        limits = self.env.model.actuator_ctrlrange
        in_range = bool(np.all(action >= limits[:, 0] - 1e-9) and np.all(action <= limits[:, 1] + 1e-9))
        self.action_within_hard_limits = self.action_within_hard_limits and in_range
        if not in_range:
            raise RuntimeError(f"scripted action outside actuator ctrlrange: {action.tolist()}")
        if self.previous_action is not None:
            delta = np.abs(policy_action - self.previous_action)
            self.max_joint_action_step_rad = max(self.max_joint_action_step_rad, float(delta[:6].max()))
            self.max_gripper_action_step_closure = max(
                self.max_gripper_action_step_closure, float(delta[6])
            )
        self.previous_action = policy_action.copy()
        timestamp_s = float(observation["simulation_time_s"])
        self.timestamps_s.append(timestamp_s)
        self.stream_lengths["timestamp"] += 1
        for role, image_key in (("front", "base_rgb"), ("wrist", "wrist_rgb")):
            image = images[image_key]
            self.image_accumulators[role]["std_sum"] += float(image.std())
            previous = self.previous_images.get(role)
            if previous is not None:
                self.image_accumulators[role]["diff_sum"] += float(
                    np.mean(np.abs(image.astype(np.int16) - previous.astype(np.int16)))
                )
                self.image_accumulators[role]["diff_count"] += 1
            self.previous_images[role] = image.copy()
            self.stream_lengths[role] += 1
        self.writers["front"].write(cv2.cvtColor(images["base_rgb"], cv2.COLOR_RGB2BGR))
        self.writers["wrist"].write(cv2.cvtColor(images["wrist_rgb"], cv2.COLOR_RGB2BGR))
        if self.dataset is not None:
            self.dataset.add_frame(
                {
                    "observation.images.base": images["base_rgb"],
                    "observation.images.wrist": images["wrist_rgb"],
                    "observation.state": policy_state.astype(np.float32),
                    "action": policy_action.astype(np.float32),
                },
                task=self.instruction,
            )
        self.stream_lengths["state"] += 1
        self.stream_lengths["action"] += 1
        row = {
            "frame_index": self.frame,
            "simulation_time_s": observation["simulation_time_s"],
            "phase": phase,
            "observation_state": policy_state.tolist(),
            "action": policy_action.tolist(),
            "raw_actuator_target": action.tolist(),
            "object_position_m": self.env.data.xpos[self.env.target_id].tolist(),
            "goal_position_m": (
                None if self.goal_position_m is None else self.goal_position_m.tolist()
            ),
            "front_video_frame": self.frame,
            "wrist_video_frame": self.frame,
        }
        self.samples.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        state = None
        for _ in range(self.env.cfg["control_steps"]):
            state = self.env.step_physics(action)
            self.max_lift = max(self.max_lift, float(state["object_position_m"][2] - self.initial_object[2]))
            contacted_fingers: set[int] = set()
            for contact in state["contacts"]:
                if self.env.target_id in {contact["body1"], contact["body2"]}:
                    self.peak_target_force = max(self.peak_target_force, contact["normal_force_N"])
                    other = contact["body2"] if contact["body1"] == self.env.target_id else contact["body1"]
                    if other in self.env.finger_ids:
                        contacted_fingers.add(other)
            if len(contacted_fingers) == 2 and phase in {"grasp_hold", "lift", "transport"}:
                self.bilateral_contact_physics_steps += 1
        actual_state = self.contract.policy_state(self.env.data.qpos[self.env.qadr])
        self.tracking_error_samples.append(np.abs(actual_state - policy_action).tolist())
        if float(self.env.data.xpos[self.env.target_id][2] - self.initial_object[2]) >= float(
            self.cfg["minimum_lift_m"]
        ):
            self.lifted_control_ticks += 1
        self.frame += 1
        self.last_state = state
        return state

    def diagnostics(self) -> dict[str, Any]:
        image_stats = {}
        for role, values in self.image_accumulators.items():
            image_stats[role] = {
                "mean_std": values["std_sum"] / max(1, self.frame),
                "mean_temporal_absdiff": values["diff_sum"] / max(1, values["diff_count"]),
            }
        return {
            "frame_count": self.frame,
            "finite_state_action": self.finite_state_action,
            "shape_valid": self.shape_valid,
            "action_within_hard_limits": self.action_within_hard_limits,
            "timestamps_s": self.timestamps_s,
            "stream_lengths": self.stream_lengths,
            "max_joint_action_step_rad": self.max_joint_action_step_rad,
            "max_gripper_action_step_closure": self.max_gripper_action_step_closure,
            "tracking_error_samples": self.tracking_error_samples,
            "bilateral_contact_physics_steps": self.bilateral_contact_physics_steps,
            "lifted_control_ticks": self.lifted_control_ticks,
            "image_stats": image_stats,
        }

    def smooth_move(self, start: np.ndarray, target: np.ndarray, seconds: float, phase: str) -> np.ndarray:
        ticks = max(1, round(seconds * self.cfg["control_hz"]))
        for index in range(ticks):
            x = (index + 1) / ticks
            blend = x * x * (3.0 - 2.0 * x)
            action = self.env.action_adapter.decode(self.env, start + blend * (target - start))
            self.control_tick(action, phase)
        return target.copy()

    def hold(self, action: np.ndarray, seconds: float, phase: str) -> None:
        for _ in range(max(1, round(seconds * self.cfg["control_hz"]))):
            self.control_tick(action, phase)


def inspect_video(path: Path, expected_hw: list[int]) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    frames = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if list(frame.shape[:2]) != list(expected_hw):
            capture.release()
            raise RuntimeError(f"decoded video resolution changed: {path} frame={frames}")
        frames += 1
    metadata = {
        "path": path.name,
        "frames": frames,
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "sha256": sha256(path),
    }
    capture.release()
    return metadata


def run_one(output: Path, pilot: dict[str, Any], variation: dict[str, Any], dataset=None) -> dict[str, Any]:
    scene_config = ROOT / "configs/benchmark" / pilot.get(
        "scene_object_config", "object.libero_spatial_cookies_to_plate_pilot.yaml"
    )
    docs = load_configs(
        ROOT / "configs/benchmark",
        object_config=scene_config,
    )
    docs["benchmark"]["values"]["resolution_hw"] = pilot["resolution_hw"]
    output.mkdir(parents=True, exist_ok=False)
    (output / ".incomplete").touch()
    env = MujocoQualificationEnv(docs, ROOT / "assets/vendor/agx_arm_sim")
    recorder = None
    try:
        env.reset(int(variation["seed"]))
        set_free_joint_pose(env, pilot["source_object_body"], variation["source_xy_m"], variation["object_yaw_rad"])
        goal_kind = variation["goal_kind"]
        if goal_kind == "body":
            set_free_joint_pose(env, pilot["goal_body"], variation["goal_xy_m"], 0.0)
        reset_delta = float(variation["robot_joint1_reset_delta_rad"])
        env.data.qpos[env.qadr[0]] += reset_delta
        env.data.ctrl[0] = env.data.qpos[env.qadr[0]]
        mujoco.mj_forward(env.model, env.data)
        for _ in range(round(env.cfg["settle_s"] / env.model.opt.timestep)):
            mujoco.mj_step(env.model, env.data)
        mujoco.mj_forward(env.model, env.data)
        env.initial_object = env.data.xpos[env.target_id].copy()

        target_up = env.data.xmat[env.target_id].reshape(3, 3)[:, 2]
        upright_cosine = float(np.dot(target_up, [0.0, 0.0, 1.0]))
        source = env.initial_object.copy()
        source_xy_drift = float(np.linalg.norm(source[:2] - variation["source_xy_m"]))
        goal_id = env.model.body(pilot["goal_body"]).id
        if goal_kind == "body":
            goal = env.data.xpos[goal_id].copy()
            goal_xy_drift = float(np.linalg.norm(goal[:2] - variation["goal_xy_m"]))
            placement_reference = np.array([
                goal[0], goal[1], goal[2] + pilot["placement_reference_above_goal_m"],
            ])
            expected_final_z_offset = float(pilot["goal_z_offset_m"])
        elif goal_kind == "tabletop":
            goal = np.array([*variation["goal_xy_m"], source[2]], dtype=np.float64)
            goal_xy_drift = None
            placement_reference = np.array([
                goal[0], goal[1], source[2] + pilot["grasp_reference_height_m"],
            ])
            expected_final_z_offset = 0.0
        else:
            goal = None
            goal_xy_drift = None
            placement_reference = None
            expected_final_z_offset = None
        base_rotation = np.asarray(env.cfg["wrist_rotation_world"], dtype=np.float64)
        desired_rotation = rotation_about_world_z(base_rotation, float(variation["wrist_yaw_rad"]))
        current_q = env.data.qpos[env.qadr[:6]].copy()
        grasp_point = source + [0.0, 0.0, pilot["grasp_reference_height_m"]]
        pregrasp = source + [0.0, 0.0, pilot["pregrasp_clearance_m"]]
        lifted = source + [0.0, 0.0, pilot["lift_clearance_m"]]
        def solve_ik(stage: str, target: np.ndarray, initial: np.ndarray) -> np.ndarray:
            try:
                return env.inverse_kinematics(target, initial, desired_rotation)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"IK stage={stage} target_m={target.tolist()} is unreachable"
                ) from exc

        q_pre = solve_ik("pregrasp", pregrasp, current_q)
        q_grasp = solve_ik("grasp", grasp_point, q_pre)
        q_lift = solve_ik("lift", lifted, q_grasp)
        q_goal = None
        q_place = None
        if goal is not None:
            above_goal = np.array([goal[0], goal[1], lifted[2]])
            q_goal = solve_ik("transport_above_goal", above_goal, q_lift)
            q_place = solve_ik("place", placement_reference, q_goal)

        recorder = Recorder(
            env,
            output,
            pilot,
            instruction=variation["instruction"],
            goal_position_m=goal,
            dataset=dataset,
        )
        action = np.r_[current_q, env.robot["gripper_actuator_open_m"]]
        recorder.hold(action, 0.5, "reset_hold")
        action = recorder.smooth_move(action, np.r_[q_pre, action[-1]], pilot["move_duration_s"], "approach")
        action = recorder.smooth_move(action, np.r_[q_grasp, action[-1]], pilot["move_duration_s"], "reach")

        # Demonstration control is deliberately open loop with respect to contact
        # and force. Contacts may be recorded as passive diagnostics, but they do
        # not alter actions, phase durations, or termination.
        action = recorder.smooth_move(
            action,
            np.r_[q_grasp, pilot["gripper_closed_target_m"]],
            pilot["close_duration_s"],
            "geometric_close",
        )
        recorder.hold(action, pilot["close_hold_s"], "grasp_hold")
        action = recorder.smooth_move(action, np.r_[q_lift, action[-1]], pilot["move_duration_s"], "lift")
        if variation["task_type"] == "pick":
            recorder.hold(action, pilot["lifted_hold_s"], "lifted_hold")
        else:
            action = recorder.smooth_move(
                action, np.r_[q_goal, action[-1]], pilot["move_duration_s"], "transport"
            )
            action = recorder.smooth_move(
                action,
                np.r_[q_place, action[-1]],
                pilot["move_duration_s"],
                "geometric_place",
            )
            open_target = np.r_[q_place, env.robot["gripper_actuator_open_m"]]
            action = recorder.smooth_move(
                action, open_target, pilot["release_duration_s"], "release"
            )
            action = recorder.smooth_move(
                action,
                np.r_[q_goal, env.robot["gripper_actuator_open_m"]],
                pilot["move_duration_s"],
                "retract",
            )
            recorder.hold(action, pilot["post_release_hold_s"], "post_release")
        final_object = env.data.xpos[env.target_id].copy()
        final_xy_error = (
            None if goal is None else float(np.linalg.norm(final_object[:2] - goal[:2]))
        )
        final_z_offset = None if goal is None else float(final_object[2] - goal[2])
        final_lift = float(final_object[2] - source[2])
        target_joint = env.model.joint(pilot["source_object_body"].replace("_main", "_joint0"))
        target_velocity = env.data.qvel[
            int(env.model.jnt_dofadr[target_joint.id]) : int(env.model.jnt_dofadr[target_joint.id]) + 3
        ]
        final_object_speed = float(np.linalg.norm(target_velocity))
        final_upright_cosine = float(np.dot(
            env.data.xmat[env.target_id].reshape(3, 3)[:, 2], [0.0, 0.0, 1.0],
        ))
        if variation["task_type"] == "pick":
            success = bool(
                recorder.max_lift >= pilot["minimum_lift_m"]
                and final_lift >= pilot["minimum_lift_m"]
            )
        else:
            success = bool(
                recorder.max_lift >= pilot["minimum_lift_m"]
                and final_xy_error <= pilot["goal_xy_tolerance_m"]
                and abs(final_z_offset - expected_final_z_offset) <= pilot["goal_z_tolerance_m"]
                and final_object_speed <= pilot["final_object_speed_tolerance_m_s"]
                and final_upright_cosine >= pilot["final_upright_cosine_min"]
            )
        result = {
            "task_success": success,
            "task_type": variation["task_type"],
            "variation_id": variation["id"],
            "control_mode": "fixed_geometry_no_contact_or_force_feedback",
            "commanded_gripper_target_m": pilot["gripper_closed_target_m"],
            "max_lift_m": recorder.max_lift,
            "final_lift_m": final_lift,
            "passive_diagnostic_peak_target_contact_force_N": recorder.peak_target_force,
            "source_settle_xy_drift_m": source_xy_drift,
            "goal_settle_xy_drift_m": goal_xy_drift,
            "final_goal_xy_error_m": final_xy_error,
            "final_goal_z_offset_m": final_z_offset,
            "final_object_speed_m_s": final_object_speed,
            "final_upright_cosine": final_upright_cosine,
            "upright_cosine_at_reset": upright_cosine,
        }
        diagnostics = recorder.diagnostics()
        recorder.close()
        recorder = None
        videos = {
            role: inspect_video(output / f"{role}.{pilot['video_extension']}", pilot["resolution_hw"])
            for role in ("front", "wrist")
        }
        structural = validate_episode_structure(pilot, diagnostics, videos)
        metrics = summarize_episode_metrics(result, diagnostics)
        (output / "scene.xml").write_text(env.scene_xml, encoding="utf-8")
        raw_manifest = {
            "schema_version": pilot["schema_version"],
            "episode_id": output.name,
            "source": {"kind": "simulation", "simulator": "mujoco"},
            "task": {
                "id": f"{pilot['task_set_id']}:{variation['task_type']}",
                "type": variation["task_type"],
                "instruction": variation["instruction"],
            },
            "variation": variation,
            "structural_validation": structural,
            "metrics": metrics,
            "videos": videos,
            "artifacts": {"samples": "samples.jsonl.gz", "scene": "scene.xml"},
        }
        # Raw evidence is finalized before the LeRobot commit.  A later dataset
        # serialization failure must never erase the original videos or sidecar.
        save_json(output / "raw_episode_manifest.json", raw_manifest)
        lerobot_episode_index = None
        if dataset is not None:
            if structural["passed"]:
                lerobot_episode_index = int(dataset.meta.total_episodes)
                dataset.save_episode()
            else:
                dataset.clear_episode_buffer()
        manifest = {
            "schema_version": pilot["schema_version"],
            "episode_id": output.name,
            "purpose": "provisional_scripted_dataset_candidate",
            "source": {"kind": "simulation", "simulator": "mujoco"},
            "task": {
                "id": f"{pilot['task_set_id']}:{variation['task_type']}",
                "type": variation["task_type"],
                "instruction": variation["instruction"],
            },
            "variation": variation,
            "observation": {"images": ["front", "wrist"], "resolution_hw": pilot["resolution_hw"],
                            "state": "q1..q6 rad + gripper closure [0,1]"},
            "action": {"semantic": "absolute q1..q6 rad + gripper closure [0,1]",
                       "raw_actuator_target_also_saved": True},
            "timing": {"fps": pilot["control_hz"], "physics_dt_s": env.model.opt.timestep,
                       "contract": "obs_t -> action_t -> physics steps -> obs_t+1"},
            "result": result,
            "structural_validation": structural,
            "metrics": metrics,
            "quality_flag": metrics["quality_flag"] if structural["passed"] else "invalid_structure",
            "rejection_reason": (
                None
                if structural["passed"]
                else "structural_validation_failed:" + ",".join(structural["failed_checks"])
            ),
            "manual_dual_video_review": "pending" if structural["passed"] else "not_applicable",
            "lerobot_episode_index": lerobot_episode_index,
            "videos": videos,
            "artifacts": {
                "samples": "samples.jsonl.gz",
                "scene": "scene.xml",
                "raw_manifest": "raw_episode_manifest.json",
            },
        }
        save_json(output / "episode_manifest.json", manifest)
        (output / ".incomplete").unlink()
        return manifest
    except BaseException as exc:
        if dataset is not None:
            dataset.clear_episode_buffer()
        save_json(output / "failure.json", {"error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        if recorder is not None:
            recorder.close()
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Explicit dataset plan YAML. No default is provided to avoid accidental stale runs.",
    )
    parser.add_argument("--usb-uuid", default="D850-5AAE")
    parser.add_argument("--usb-mount", type=Path)
    parser.add_argument("--output", type=Path, help="Exact run directory; must not already exist")
    parser.add_argument("--run-name")
    parser.add_argument(
        "--limit",
        type=int,
        help="Only run the first N configured variations (use 3 for manual preflight).",
    )
    parser.add_argument(
        "--variation",
        action="append",
        dest="variation_ids",
        help="Run only this variation id; repeat the option to select multiple ids.",
    )
    parser.add_argument(
        "--task-type",
        action="append",
        choices=SCRIPTED_TASK_TYPES,
        dest="task_types",
        help="Run one task category; repeat to select multiple categories.",
    )
    parser.add_argument(
        "--list-variations",
        action="store_true",
        help="Print the configured ids, task types, prompts and seeds without generating data.",
    )
    args = parser.parse_args()
    pilot_path = args.config.expanduser()
    if not pilot_path.is_absolute():
        pilot_path = ROOT / pilot_path
    pilot_path = pilot_path.resolve()
    if not pilot_path.is_file():
        parser.error(f"config does not exist: {pilot_path}")
    pilot = yaml.safe_load(pilot_path.read_text(encoding="utf-8"))["values"]
    validate_scripted_dataset_config(pilot)
    if pilot.get("execution_enabled") is False:
        raise RuntimeError(
            f"dataset config is disabled and must not be generated: {pilot_path}"
        )
    variations = pilot["variations"]
    if args.list_variations:
        print(json.dumps([
            {key: variation[key] for key in ("id", "task_type", "instruction", "seed")}
            for variation in variations
        ], ensure_ascii=False, indent=2))
        return
    if args.variation_ids and args.task_types:
        parser.error("--variation and --task-type cannot be used together")
    if args.variation_ids:
        known_ids = {variation["id"] for variation in variations}
        unknown_ids = set(args.variation_ids) - known_ids
        if unknown_ids:
            parser.error(f"unknown variation ids: {sorted(unknown_ids)}")
        selected_ids = set(args.variation_ids)
        variations = [variation for variation in variations if variation["id"] in selected_ids]
    if args.task_types:
        selected_types = set(args.task_types)
        variations = [variation for variation in variations if variation["task_type"] in selected_types]
    if args.limit is not None:
        if not 1 <= args.limit <= len(variations):
            parser.error(f"--limit must be between 1 and {len(variations)}")
        variations = variations[:args.limit]
    if args.output is not None and (args.usb_mount is not None or args.run_name is not None):
        parser.error("--output cannot be combined with --usb-mount or --run-name")
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        storage_root = output.parent
    else:
        mount = find_usb(args.usb_uuid, args.usb_mount)
        timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        output = mount / "FragileGrabVLA/provisional_data" / (
            args.run_name or f"piper_x_scripted_provisional_{len(variations)}ep_{timestamp}"
        )
        storage_root = mount
    verify_writable(storage_root)
    if shutil.disk_usage(storage_root).free < 5 * 1024**3:
        raise RuntimeError("dataset storage has less than 5 GiB free")
    output.mkdir(parents=True, exist_ok=False)
    (output / "PROVISIONAL_DATASET_REQUIRES_REVIEW.txt").write_text(
        "Raw structurally valid episodes are saved without metric-based rejection. Review both videos and "
        "quality_flag/rejection_reason before selecting a later train split.\n",
        encoding="utf-8",
    )
    stored_config_name = pilot_path.name
    shutil.copy2(pilot_path, output / stored_config_name)
    dataset = create_lerobot_dataset(output / "lerobot", pilot)
    summary = {
        "schema_version": "fragilegrab.scripted_provisional_summary/0.4",
        "status": "running",
        "requested_episode_count": len(variations),
        "requested_task_type_counts": {
            task_type: sum(v["task_type"] == task_type for v in variations)
            for task_type in SCRIPTED_TASK_TYPES
        },
        "training_admission": "not_evaluated_raw_dataset",
        "dataset_repo_id": pilot["dataset_repo_id"],
        "norm_stats_id": pilot["norm_stats_id"],
        "lerobot_root": "lerobot",
        "config": {"storage_ref": stored_config_name, "sha256": sha256(pilot_path)},
        "episodes": [],
    }
    save_json(output / "smoke_summary.json", summary)
    for variation in variations:
        episode = output / variation["id"]
        print(f"running {variation['id']} -> {episode}", flush=True)
        try:
            manifest = run_one(episode, pilot, variation, dataset=dataset)
            item = {
                "id": variation["id"],
                "status": "complete",
                "structural_valid": manifest["structural_validation"]["passed"],
                "failed_structural_checks": manifest["structural_validation"]["failed_checks"],
                "quality_flag": manifest["quality_flag"],
                "rejection_reason": manifest["rejection_reason"],
                "warnings": manifest["metrics"]["warnings"],
                "lerobot_episode_index": manifest["lerobot_episode_index"],
                **manifest["result"],
            }
        except Exception as exc:
            item = {"id": variation["id"], "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        summary["episodes"].append(item)
        save_json(output / "smoke_summary.json", summary)
    raw_complete = sum(item.get("status") == "complete" for item in summary["episodes"])
    structurally_valid = sum(item.get("structural_valid") is True for item in summary["episodes"])
    task_successes = sum(item.get("task_success") is True for item in summary["episodes"])
    all_requested_serialized = structurally_valid == len(variations)
    summary.update({
        "status": "complete" if all_requested_serialized else "complete_with_structural_failures",
        "raw_episode_count": raw_complete,
        "structurally_valid_episode_count": structurally_valid,
        "lerobot_episode_count": int(dataset.meta.total_episodes),
        "serialized_task_type_counts": {
            task_type: sum(
                item.get("structural_valid") is True and item.get("task_type") == task_type
                for item in summary["episodes"]
            )
            for task_type in SCRIPTED_TASK_TYPES
        },
        "task_success_count_metric_only": task_successes,
        "all_requested_serialized": all_requested_serialized,
        "training_admission": "not_evaluated_pending_visual_review",
    })
    save_json(output / "smoke_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not all_requested_serialized:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
