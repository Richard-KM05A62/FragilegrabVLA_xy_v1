#!/usr/bin/env python3
"""Use a read-only real PiPER-X as a local MuJoCo position-teleoperation leader."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "data_collect"))

from aligned_capture import PiperCanAdapter
from fragilegrab.benchmark.environment import MujocoQualificationEnv, load_configs
from fragilegrab.benchmark.teleop import (
    LatestLeaderReader,
    PositionMapper,
    TeleopEpisodeRecorder,
    latency_summary_ms,
    load_teleop_config,
    sim_observation_state,
)


def _label(rgb: np.ndarray, text: str) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 27), (0, 0, 0), -1)
    cv2.putText(bgr, text, (7, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return bgr


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _latencies(read_to_apply: list[int], read_to_render: list[int], apply_to_render: list[int]) -> dict:
    return {
        "clock": "host time.monotonic_ns()",
        "read_to_action_apply": latency_summary_ms(read_to_apply),
        "read_to_render_finished": latency_summary_ms(read_to_render),
        "action_apply_to_render_finished": latency_summary_ms(apply_to_render),
    }


def _print_latency(label: str, summary: dict) -> None:
    print(label)
    for name, row in summary.items():
        if not isinstance(row, dict) or row.get("samples", 0) == 0:
            continue
        print(f"  {name}: mean={row['mean_ms']:.2f} ms, p95={row['p95_ms']:.2f} ms, n={row['samples']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-assets", type=Path, required=True, help="Pinned agx_arm_sim checkout")
    parser.add_argument("--teleop-config", type=Path, default=ROOT / "configs/teleop/piper_mujoco.yaml")
    parser.add_argument("--benchmark-config-dir", type=Path, default=ROOT / "configs/benchmark")
    parser.add_argument(
        "--object-config",
        type=Path,
        default=ROOT / "configs/benchmark/object.libero_spatial_bowl_to_plate.yaml",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data/teleop_mujoco")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--task", help="Override the task prompt from the object config")
    args = parser.parse_args()

    config_path = args.teleop_config.resolve()
    config = load_teleop_config(config_path)
    loops = config["loops"]
    visual_cfg = config["visual"]
    reader_cfg = config["reader"]
    recording_cfg = config["recording"]
    for name in ("read_hz", "control_hz", "render_hz"):
        if not np.isfinite(float(loops[name])) or float(loops[name]) <= 0:
            raise ValueError(f"loops.{name} must be positive")
    if float(loops["control_hz"]) > float(loops["read_hz"]):
        print("Warning: control_hz exceeds read_hz; repeated leader samples are expected.")

    docs = load_configs(args.benchmark_config_dir, object_config=args.object_config)
    docs["benchmark"]["values"]["resolution_hw"] = list(visual_cfg["record_resolution_hw"])
    task = args.task or docs["object"]["values"].get("instruction")
    if not task:
        raise ValueError("task prompt is required")
    env = MujocoQualificationEnv(docs, args.robot_assets)
    mapper = PositionMapper(config["mapping"], env.model.actuator_ctrlrange)
    adapter = PiperCanAdapter(
        gripper_open_width_m=float(reader_cfg["gripper_open_width_m"]),
        channel=str(reader_cfg["can_channel"]),
        interface=str(reader_cfg["can_interface"]),
    )
    reader = LatestLeaderReader(adapter, float(loops["read_hz"]))
    recorder = TeleopEpisodeRecorder(
        args.output,
        str(recording_cfg["repo_id"]),
        int(round(float(loops["render_hz"]))),
        int(recording_cfg.get("image_writer_threads", 0)),
    )
    seed = args.seed
    read_to_apply: list[int] = []
    read_to_render: list[int] = []
    apply_to_render: list[int] = []
    last_alignment: dict | None = None
    window = "FragileGrab teleop | S start | E save | Y success | F failure | D discard | R reset | Q quit"

    def align_sim_to_current_leader() -> tuple[dict, np.ndarray]:
        nonlocal last_alignment
        leader_at_start = reader.latest(timeout_s=float(reader_cfg["first_sample_timeout_s"]))
        mapper.reset_filter()
        reset_action, mapping_audit = mapper.map(leader_at_start.state)
        env.data.qpos[env.qadr[:6]] = reset_action[:6]
        env.data.qpos[env.qadr[6]] = reset_action[6]
        env.data.qpos[env.qadr[7]] = -reset_action[6]
        env.data.ctrl[:] = reset_action
        mujoco.mj_forward(env.model, env.data)
        mapper.reset_filter(reset_action)
        last_alignment = {
            "real_read_timestamp_ns": leader_at_start.read_timestamp_ns,
            "real_start_state_q1_q6_and_closedness": leader_at_start.state.astype(float).tolist(),
            "sim_initial_action_q1_q6_and_gripper_m": reset_action.astype(float).tolist(),
            "mapping": mapping_audit,
        }
        return last_alignment, env.observe()

    def reset_scene() -> dict:
        nonlocal seed
        observation = env.reset(seed)
        startup_mode = config["mapping"].get("startup_alignment", {}).get("mode", "configured_mapping")
        if startup_mode == "real_absolute":
            _, observation = align_sim_to_current_leader()
        elif startup_mode == "configured_mapping":
            mapper.reset_filter(env.data.ctrl)
        else:
            raise ValueError(f"unsupported mapping.startup_alignment.mode: {startup_mode}")
        print(f"Scene reset (seed={seed}). Move the real Piper; only MuJoCo receives commands.")
        return observation

    def start_episode() -> None:
        nonlocal observation, read_to_apply, read_to_render, apply_to_render
        if recorder.active:
            print("Episode is already recording.")
            return
        if config["mapping"].get("startup_alignment", {}).get("mode") == "real_absolute":
            alignment, observation = align_sim_to_current_leader()
        else:
            alignment = None
        read_to_apply, read_to_render, apply_to_render = [], [], []
        metadata = {
            "episode_schema_version": "fragilegrab.teleop_mujoco/0.1",
            "source": {"kind": "simulation", "simulator": "mujoco", "leader": "real_piper_read_only"},
            "task": {"prompt": task},
            "scene_seed": seed,
            "robot_reset_pose": {
                "joint_position_q1_q6_and_two_fingers": env.data.qpos[env.qadr].astype(float).tolist(),
                "actuator_target_q1_q6_and_gripper": env.data.ctrl.astype(float).tolist(),
            },
            "object_initial_pose": {
                "position_m": env.data.xpos[env.target_id].astype(float).tolist(),
                "quaternion_wxyz": env.data.xquat[env.target_id].astype(float).tolist(),
            },
            "leader_start_alignment": alignment,
            "action": {
                "semantics": "mapping/filtering/clamp output actually applied to MuJoCo: q1..q6 rad + gripper actuator m",
                "source": "real_piper_position_teleoperation",
            },
            "observation_state": "sim q1..q6 rad + normalized gripper closedness",
            "images": {
                "record_resolution_hw": list(visual_cfg["record_resolution_hw"]),
                "preview_resolution_hw": list(visual_cfg["preview_resolution_hw"]),
                "preview_source": "area-downsample of the same recorded MuJoCo render; no second render",
            },
            "config": {"source_path": str(config_path), "storage_ref": "teleop_config.yaml", "sha256": _sha256(config_path)},
            "reader_start": reader.start_metadata,
        }
        path = recorder.start(metadata)
        (path / "teleop_config.yaml").write_bytes(config_path.read_bytes())
        print(f"Recording started: {path}")

    def finish_episode(outcome: str, reason: str) -> None:
        if not recorder.active:
            print("No active episode.")
            return
        summary = _latencies(read_to_apply, read_to_render, apply_to_render)
        try:
            path = recorder.finish(outcome, reason, summary)
        except RuntimeError as exc:
            print(f"Cannot save episode: {exc}")
            return
        print(f"Saved {outcome} episode: {path}")
        _print_latency("Episode latency:", summary)

    reader.start()
    observation = reset_scene()
    leader = reader.latest(timeout_s=float(reader_cfg["first_sample_timeout_s"]))
    action, mapping_info = mapper.map(leader.state)
    last_apply_ns = time.monotonic_ns()
    last_read_ns = leader.read_timestamp_ns
    max_age_ns = round(float(reader_cfg["max_sample_age_ms"]) * 1e6)
    control_period = 1.0 / float(loops["control_hz"])
    render_period = 1.0 / float(loops["render_hz"])
    physics_dt = float(env.model.opt.timestep)
    physics_credit_s = 0.0
    next_control = time.perf_counter()
    next_render = next_control

    try:
        while True:
            now = time.perf_counter()
            if now >= next_control:
                leader = reader.latest(timeout_s=0.0)
                action, mapping_info = mapper.map(leader.state)
                apply_ns = time.monotonic_ns()
                if apply_ns - leader.read_timestamp_ns > max_age_ns:
                    raise RuntimeError(
                        f"Piper sample is stale by {(apply_ns - leader.read_timestamp_ns) / 1e6:.1f} ms; stopping"
                    )
                physics_credit_s += control_period
                physics_steps = max(1, int(physics_credit_s / physics_dt))
                physics_credit_s -= physics_steps * physics_dt
                for _ in range(physics_steps):
                    env.step_physics(action)
                last_apply_ns = apply_ns
                last_read_ns = leader.read_timestamp_ns
                if recorder.active:
                    read_to_apply.append(apply_ns - leader.read_timestamp_ns)
                next_control += control_period
                while next_control <= now:
                    next_control += control_period

            key = -1
            if now >= next_render:
                observation = env.observe()
                render_ns = time.monotonic_ns()
                if recorder.active:
                    read_to_render.append(render_ns - last_read_ns)
                    apply_to_render.append(render_ns - last_apply_ns)
                    recorder.append(
                        front=observation["images"]["base_rgb"],
                        wrist=observation["images"]["wrist_rgb"],
                        state=sim_observation_state(env),
                        action=action,
                        task=task,
                        timing={
                            "real_joint_read_timestamp_ns": last_read_ns,
                            "sim_action_apply_timestamp_ns": last_apply_ns,
                            "render_finished_timestamp_ns": render_ns,
                            "simulation_time_s": float(env.data.time),
                        },
                        leader_state=leader.state,
                        mapping=mapping_info,
                    )
                status = "REC" if recorder.active else "IDLE"
                preview_h, preview_w = visual_cfg["preview_resolution_hw"]
                base_preview = cv2.resize(
                    observation["images"]["base_rgb"],
                    (preview_w, preview_h),
                    interpolation=cv2.INTER_AREA,
                )
                wrist_preview = cv2.resize(
                    observation["images"]["wrist_rgb"],
                    (preview_w, preview_h),
                    interpolation=cv2.INTER_AREA,
                )
                base = _label(base_preview, f"front | {status} | seed {seed}")
                wrist = _label(wrist_preview, "wrist")
                cv2.imshow(window, np.concatenate([base, wrist], axis=1))
                key = cv2.waitKey(1) & 0xFF
                next_render += render_period
                while next_render <= now:
                    next_render += render_period

            if key == ord("s"):
                start_episode()
            elif key == ord("e"):
                finish_episode("unlabeled", "operator_end")
            elif key == ord("y"):
                finish_episode("success", "operator_marked_success")
            elif key == ord("f"):
                finish_episode("failure", "operator_marked_failure")
            elif key == ord("d"):
                recorder.discard()
                print("Active episode discarded.")
            elif key == ord("r"):
                if recorder.active:
                    print("Reset refused while recording; save or discard the episode first.")
                else:
                    observation = reset_scene()
            elif key in (ord("q"), 27):
                if recorder.active:
                    recorder.discard()
                    print("Quit with an active episode: incomplete episode discarded.")
                break

            sleep_s = min(next_control, next_render) - time.perf_counter()
            if sleep_s > 0:
                time.sleep(min(sleep_s, 0.002))
    finally:
        if recorder.active:
            recorder.discard()
        reader.close()
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()
