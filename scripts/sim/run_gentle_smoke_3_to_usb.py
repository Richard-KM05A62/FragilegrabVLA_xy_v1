#!/usr/bin/env python3
"""Record three diverse, geometry-controlled, dual-view smoke episodes to USB."""

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


class Recorder:
    def __init__(self, env: MujocoQualificationEnv, output: Path, cfg: dict[str, Any]):
        self.env, self.output, self.cfg = env, output, cfg
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
            raise RuntimeError("MJPG video writer failed to open")
        self.samples_path = output / "samples.jsonl.gz"
        self.samples = gzip.open(self.samples_path, "wt", encoding="utf-8")
        self.frame = 0
        self.max_lift = 0.0
        self.peak_target_force = 0.0
        self.last_state: dict[str, Any] | None = None
        self.initial_object = env.data.xpos[env.target_id].copy()

    def close(self) -> None:
        if hasattr(self, "samples") and self.samples is not None:
            self.samples.close()
            self.samples = None
        for writer in getattr(self, "writers", {}).values():
            writer.release()

    def control_tick(self, action: np.ndarray, phase: str) -> dict[str, Any]:
        observation = self.env.observe()
        images = observation.pop("images")
        self.writers["front"].write(cv2.cvtColor(images["base_rgb"], cv2.COLOR_RGB2BGR))
        self.writers["wrist"].write(cv2.cvtColor(images["wrist_rgb"], cv2.COLOR_RGB2BGR))
        row = {
            "frame_index": self.frame,
            "simulation_time_s": observation["simulation_time_s"],
            "phase": phase,
            "observation_state": self.contract.policy_state(observation["joint_position"]).tolist(),
            "action": self.contract.policy_action(action).tolist(),
            "raw_actuator_target": action.tolist(),
            "object_position_m": self.env.data.xpos[self.env.target_id].tolist(),
            "goal_position_m": self.env.data.xpos[self.env.model.body(self.env.obj["goal_body"]).id].tolist(),
            "front_video_frame": self.frame,
            "wrist_video_frame": self.frame,
        }
        self.samples.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        state = None
        for _ in range(self.env.cfg["control_steps"]):
            state = self.env.step_physics(action)
            self.max_lift = max(self.max_lift, float(state["object_position_m"][2] - self.initial_object[2]))
            for contact in state["contacts"]:
                if self.env.target_id in {contact["body1"], contact["body2"]}:
                    self.peak_target_force = max(self.peak_target_force, contact["normal_force_N"])
        self.frame += 1
        self.last_state = state
        return state

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


def run_one(output: Path, pilot: dict[str, Any], variation: dict[str, Any]) -> dict[str, Any]:
    docs = load_configs(
        ROOT / "configs/benchmark",
        object_config=ROOT / "configs/benchmark/object.libero_spatial_cookies_to_plate_pilot.yaml",
    )
    docs["benchmark"]["values"]["resolution_hw"] = pilot["resolution_hw"]
    output.mkdir(parents=True, exist_ok=False)
    (output / ".incomplete").touch()
    env = MujocoQualificationEnv(docs, ROOT / "assets/vendor/agx_arm_sim")
    recorder = None
    try:
        env.reset(int(variation["seed"]))
        set_free_joint_pose(env, pilot["source_object_body"], variation["source_xy_m"], variation["object_yaw_rad"])
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
        if upright_cosine < pilot["upright_cosine_min"]:
            raise RuntimeError(f"source object is tilted: upright cosine={upright_cosine}")
        goal_id = env.model.body(pilot["goal_body"]).id
        source = env.initial_object.copy()
        goal = env.data.xpos[goal_id].copy()
        source_xy_drift = float(np.linalg.norm(source[:2] - variation["source_xy_m"]))
        goal_xy_drift = float(np.linalg.norm(goal[:2] - variation["goal_xy_m"]))
        drift_limit = float(pilot["settled_pose_xy_tolerance_m"])
        if source_xy_drift > drift_limit or goal_xy_drift > drift_limit:
            raise RuntimeError(
                "scene placement moved during settling: "
                f"source drift={source_xy_drift:.6f} m, goal drift={goal_xy_drift:.6f} m"
            )
        base_rotation = np.asarray(env.cfg["wrist_rotation_world"], dtype=np.float64)
        desired_rotation = rotation_about_world_z(base_rotation, float(variation["wrist_yaw_rad"]))
        current_q = env.data.qpos[env.qadr[:6]].copy()
        grasp_point = source + [0.0, 0.0, pilot["grasp_reference_height_m"]]
        pregrasp = source + [0.0, 0.0, pilot["pregrasp_clearance_m"]]
        lifted = source + [0.0, 0.0, pilot["lift_clearance_m"]]
        above_goal = np.array([goal[0], goal[1], lifted[2]])
        place_reference = np.array([
            goal[0], goal[1], goal[2] + pilot["placement_reference_above_goal_m"],
        ])
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
        q_goal = solve_ik("transport_above_goal", above_goal, q_lift)
        q_place = solve_ik("place", place_reference, q_goal)

        recorder = Recorder(env, output, pilot)
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
        action = recorder.smooth_move(action, np.r_[q_goal, action[-1]], pilot["move_duration_s"], "transport")

        action = recorder.smooth_move(
            action,
            np.r_[q_place, action[-1]],
            pilot["move_duration_s"],
            "geometric_place",
        )

        open_target = np.r_[q_place, env.robot["gripper_actuator_open_m"]]
        action = recorder.smooth_move(action, open_target, pilot["release_duration_s"], "release")
        action = recorder.smooth_move(
            action,
            np.r_[q_goal, env.robot["gripper_actuator_open_m"]],
            pilot["move_duration_s"],
            "retract",
        )
        recorder.hold(action, pilot["post_release_hold_s"], "post_release")
        final_object = env.data.xpos[env.target_id].copy()
        final_xy_error = float(np.linalg.norm(final_object[:2] - goal[:2]))
        final_z_offset = float(final_object[2] - env.data.xpos[goal_id][2])
        target_joint = env.model.joint(pilot["source_object_body"].replace("_main", "_joint0"))
        target_velocity = env.data.qvel[
            int(env.model.jnt_dofadr[target_joint.id]) : int(env.model.jnt_dofadr[target_joint.id]) + 3
        ]
        final_object_speed = float(np.linalg.norm(target_velocity))
        final_upright_cosine = float(np.dot(
            env.data.xmat[env.target_id].reshape(3, 3)[:, 2], [0.0, 0.0, 1.0],
        ))
        success = bool(
            recorder.max_lift >= pilot["minimum_lift_m"]
            and final_xy_error <= pilot["goal_xy_tolerance_m"]
            and abs(final_z_offset - pilot["goal_z_offset_m"]) <= pilot["goal_z_tolerance_m"]
            and final_object_speed <= pilot["final_object_speed_tolerance_m_s"]
            and final_upright_cosine >= pilot["final_upright_cosine_min"]
        )
        result = {
            "task_success": success,
            "variation_id": variation["id"],
            "control_mode": "fixed_geometry_no_contact_or_force_feedback",
            "commanded_gripper_target_m": pilot["gripper_closed_target_m"],
            "max_lift_m": recorder.max_lift,
            "passive_diagnostic_peak_target_contact_force_N": recorder.peak_target_force,
            "source_settle_xy_drift_m": source_xy_drift,
            "goal_settle_xy_drift_m": goal_xy_drift,
            "final_goal_xy_error_m": final_xy_error,
            "final_goal_z_offset_m": final_z_offset,
            "final_object_speed_m_s": final_object_speed,
            "final_upright_cosine": final_upright_cosine,
            "upright_cosine_at_reset": upright_cosine,
        }
        recorder.close()
        recorder = None
        videos = {}
        for role, path in (("front", output / "front.avi"), ("wrist", output / "wrist.avi")):
            capture = cv2.VideoCapture(str(path))
            videos[role] = {
                "path": path.name,
                "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fps": float(capture.get(cv2.CAP_PROP_FPS)),
                "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "sha256": sha256(path),
            }
            capture.release()
        manifest = {
            "schema_version": pilot["schema_version"],
            "episode_id": output.name,
            "purpose": "expert_smoke_not_yet_training_data",
            "source": {"kind": "simulation", "simulator": "mujoco"},
            "task": {"id": pilot["derived_task_id"], "instruction": pilot["instruction"]},
            "variation": variation,
            "observation": {"images": ["front", "wrist"], "resolution_hw": pilot["resolution_hw"],
                            "state": "q1..q6 rad + gripper closure [0,1]"},
            "action": {"semantic": "absolute q1..q6 rad + gripper closure [0,1]",
                       "raw_actuator_target_also_saved": True},
            "timing": {"fps": pilot["control_hz"], "physics_dt_s": env.model.opt.timestep,
                       "contract": "obs_t -> action_t -> physics steps -> obs_t+1"},
            "result": result,
            "videos": videos,
            "artifacts": {"samples": "samples.jsonl.gz", "scene": "scene.xml"},
        }
        (output / "scene.xml").write_text(env.scene_xml, encoding="utf-8")
        save_json(output / "episode_manifest.json", manifest)
        (output / ".incomplete").unlink()
        return manifest
    except BaseException as exc:
        save_json(output / "failure.json", {"error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        if recorder is not None:
            recorder.close()
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usb-uuid", default="D850-5AAE")
    parser.add_argument("--usb-mount", type=Path)
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
    args = parser.parse_args()
    mount = find_usb(args.usb_uuid, args.usb_mount)
    verify_writable(mount)
    if shutil.disk_usage(mount).free < 2 * 1024**3:
        raise RuntimeError("USB has less than 2 GiB free")
    pilot_path = ROOT / "configs/benchmark/gentle_demo_pilot.yaml"
    pilot = yaml.safe_load(pilot_path.read_text(encoding="utf-8"))["values"]
    variations = pilot["variations"]
    if args.limit is not None and args.variation_ids:
        parser.error("--limit and --variation cannot be used together")
    if args.variation_ids:
        known_ids = {variation["id"] for variation in variations}
        unknown_ids = set(args.variation_ids) - known_ids
        if unknown_ids:
            parser.error(f"unknown variation ids: {sorted(unknown_ids)}")
        selected_ids = set(args.variation_ids)
        variations = [variation for variation in variations if variation["id"] in selected_ids]
    if args.limit is not None:
        if not 1 <= args.limit <= len(variations):
            parser.error(f"--limit must be between 1 and {len(variations)}")
        variations = variations[:args.limit]
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    output = mount / "FragileGrabVLA/pilot_data" / (
        args.run_name or f"geometric_cookies_task01_{len(variations)}ep_{timestamp}"
    )
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema_version": "fragilegrab.geometric_demo_smoke_summary/0.2",
        "status": "running",
        "requested_episode_count": len(variations),
        "training_admission": "rejected_until_all_requested_pass_and_manual_video_review",
        "config": str(pilot_path),
        "episodes": [],
    }
    save_json(output / "smoke_summary.json", summary)
    for variation in variations:
        episode = output / variation["id"]
        print(f"running {variation['id']} -> {episode}", flush=True)
        try:
            manifest = run_one(episode, pilot, variation)
            item = {"id": variation["id"], "status": "complete", **manifest["result"]}
        except Exception as exc:
            item = {"id": variation["id"], "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        summary["episodes"].append(item)
        save_json(output / "smoke_summary.json", summary)
    passed = sum(item.get("task_success") is True for item in summary["episodes"])
    all_requested_passed = passed == len(variations)
    summary.update({
        "status": "complete",
        "passed": passed,
        "all_requested_passed": all_requested_passed,
        "training_admission": "pending_manual_dual_video_review" if all_requested_passed else "rejected",
    })
    save_json(output / "smoke_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not all_requested_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
