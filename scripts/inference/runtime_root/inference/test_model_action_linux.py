"""One-shot Linux pi0.5 action test with bounded Piper motion.

This is intentionally separate from the continuous RTC launcher. It records the
complete model output before optionally sending one low-speed joint target. The
gripper is never commanded.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np

from .rtc_core import DataBuffer, RuntimeConfig, bootstrap_reference, checked_actions
from .rtc_sources_linux import SourcesLinux
from .rtc_transport import WebsocketRTCBackend


def _wait_live_observation(data: DataBuffer, sources: SourcesLinux, timeout_s: float):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        failure = sources.failure()
        if failure is not None:
            raise RuntimeError(f"{failure[0]} source failed: {failure[1]}") from failure[1]
        now_ns = time.monotonic_ns()
        observation = data.latest(now_ns)
        if observation is not None and data.is_stationary(now_ns):
            return observation
        time.sleep(0.02)
    raise TimeoutError("live inputs did not produce a stationary observation")


def _wait_joint_feedback(robot, timeout_s: float = 5.0) -> np.ndarray:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        value = robot.get_joint_angles()
        if value is not None and value.msg is not None and len(value.msg) == 6:
            return np.asarray(value.msg, dtype=np.float64)
        time.sleep(0.05)
    raise TimeoutError("Piper joint feedback timeout")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--base-serial")
    parser.add_argument("--orbbec-sdk-root", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, default=root / "models/assets/local/piper_x_recordings_v000/norm_stats.json")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--execution-steps", type=int, default=4)
    parser.add_argument("--target-index", type=int, default=1)
    parser.add_argument("--max-joint-delta", type=float, default=0.05)
    parser.add_argument("--speed-percent", type=int, default=5)
    parser.add_argument(
        "--invert-joints",
        action="store_true",
        help="Negate all six model joint targets before applying safety limits; gripper is unchanged",
    )
    parser.add_argument("--output", type=Path, default=root / "logs/linux_model_action_once.json")
    parser.add_argument("--execute", action="store_true", help="Send one bounded joint target; gripper remains unchanged")
    args = parser.parse_args()
    try:
        config = RuntimeConfig(horizon=args.horizon, execution_steps=args.execution_steps)
    except ValueError as exc:
        parser.error(str(exc))
    if not 0 <= args.target_index < config.horizon:
        parser.error(f"--target-index must be in [0,{config.horizon - 1}]")
    if not np.isfinite(args.max_joint_delta) or not 0 < args.max_joint_delta <= 0.08:
        parser.error("--max-joint-delta must be in (0,0.08]")
    if not 1 <= args.speed_percent <= 10:
        parser.error("--speed-percent must be in [1,10]")

    stop = threading.Event()
    data = DataBuffer(config)
    sources = SourcesLinux(data, stop)
    backend = WebsocketRTCBackend(args.host, args.port, config, args.norm_stats, response_timeout=180)
    # A first JAX compile can block the single-process local server for more
    # than websockets' default keepalive window. Reconnect this one-shot Linux
    # client without protocol pings; the explicit response timeout remains.
    initial_socket = backend._ws
    if initial_socket is not None:
        initial_socket.close()
    backend._ws = None
    default_connect = backend._connect_fn
    backend._connect_fn = lambda uri, **kwargs: default_connect(uri, ping_interval=None, **kwargs)
    backend._connect()
    result: dict = {
        "motion_requested": bool(args.execute),
        "motion_sent": False,
        "gripper_commanded": False,
        "target_index": args.target_index,
        "horizon": config.horizon,
        "execution_steps": config.execution_steps,
        "max_joint_delta_rad": args.max_joint_delta,
        "speed_percent": args.speed_percent,
    }
    try:
        sources.start_live(args.base_serial, None, args.orbbec_sdk_root)
        observation = _wait_live_observation(data, sources, 10.0)
        reference, weights = bootstrap_reference(observation, config)
        payload = observation.policy_input(args.task)
        payload["rtc/reference_actions"] = reference
        payload["rtc/reference_weights"] = weights
        payload["rtc/request_id"] = 1
        response = backend.infer(payload)
        actions = np.asarray(checked_actions(response["actions"], config.horizon))
        selected_model = actions[args.target_index, :6].astype(np.float64)
        selected = -selected_model if args.invert_joints else selected_model.copy()
        result.update({
            "observation_state": observation.state.tolist(),
            "actions": actions.tolist(),
            "selected_model_joints_rad": selected_model.tolist(),
            "invert_joints": bool(args.invert_joints),
            "selected_command_joints_before_limits_rad": selected.tolist(),
            "selected_gripper_closedness": float(actions[args.target_index, 6]),
            "policy_timing": response.get("policy_timing", {}),
            "server_metadata": backend.metadata,
        })
    finally:
        backend.close()
        stop.set()
        unfinished = sources.join()
        result["unfinished_source_threads"] = unfinished

    if args.execute:
        from pyAgxArm import AgxArmFactory, create_agx_arm_config

        robot = AgxArmFactory.create_arm(create_agx_arm_config(
            robot="piper_x", comm="can", channel="can0", interface="socketcan"
        ))
        robot.connect()
        try:
            current = _wait_joint_feedback(robot)
            limits = []
            for index in range(1, 7):
                response = robot.get_joint_angle_vel_limits(index)
                if response is None or response.msg is None:
                    raise RuntimeError(f"missing joint {index} limits")
                limits.append([float(response.msg.min_angle_limit), float(response.msg.max_angle_limit)])
            lower = np.asarray([item[0] for item in limits])
            upper = np.asarray([item[1] for item in limits])
            bounded = np.clip(selected, lower, upper)
            delta = bounded - current
            result.update({
                "motion_start_joints_rad": current.tolist(),
                "firmware_joint_limits_rad": limits,
                "bounded_target_joints_rad": bounded.tolist(),
                "target_clamp_rad": (bounded - selected).tolist(),
                "command_delta_rad": delta.tolist(),
            })
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            if not np.isfinite(bounded).all() or float(np.max(np.abs(delta))) > args.max_joint_delta:
                raise RuntimeError(
                    f"bounded model target exceeds {args.max_joint_delta:.3f} rad step: "
                    f"max={float(np.max(np.abs(delta))):.6f}"
                )
            if not all(robot.get_joint_enable_status(index) for index in range(1, 7)):
                raise RuntimeError("all joints must already be enabled")
            status = robot.get_arm_status()
            if status is None or status.msg is None or "NORMAL" not in repr(status.msg.arm_status):
                raise RuntimeError(f"Piper status is not normal: {status}")
            robot.set_speed_percent(args.speed_percent)
            robot.move_j(bounded.tolist())
            result["motion_sent"] = True
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            deadline = time.monotonic() + 6.0
            while True:
                reached = _wait_joint_feedback(robot, 0.5)
                error = float(np.max(np.abs(reached - bounded)))
                if error <= 0.01:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"bounded target not reached; max error={error:.6f}")
                time.sleep(0.05)
            result["motion_end_joints_rad"] = reached.tolist()
            result["motion_end_max_error_rad"] = error
        except BaseException as exc:
            result["motion_error"] = f"{type(exc).__name__}: {exc}"
            try:
                result["motion_end_joints_rad"] = _wait_joint_feedback(robot, 1.0).tolist()
            except Exception:
                pass
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            raise
        finally:
            robot.disconnect()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
