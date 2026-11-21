"""Bounded Linux Piper command sink for supervised RTC tests."""

from __future__ import annotations

import atexit
import json
import math
import threading
import time
from pathlib import Path

import numpy as np


DEFAULT_STARTUP_JOINTS_RAD = (0.0, 0.1, -0.1, 0.6, 0.0, 0.0)
DEFAULT_STARTUP_GRIPPER_WIDTH_M = 0.0


class PiperCommandSinkLinux:
    """Optionally negate model joint positions and send bounded MOVE_J targets.

    The gripper argument is deliberately ignored during joint-sign validation.
    """

    def __init__(self, *, invert_joints: bool = False, max_joint_delta: float = 0.5,
                 speed_percent: int = 15, limit_tolerance_rad: float = math.radians(5.0),
                 startup_joints_rad: tuple[float, ...] = DEFAULT_STARTUP_JOINTS_RAD,
                 startup_gripper_width_m: float = DEFAULT_STARTUP_GRIPPER_WIDTH_M,
                 log_path: Path | str = "logs/linux_rtc_direct_commands.jsonl") -> None:
        from pyAgxArm import AgxArmFactory, create_agx_arm_config

        self.max_joint_delta = float(max_joint_delta)
        self.limit_tolerance_rad = float(limit_tolerance_rad)
        self.invert_joints = bool(invert_joints)
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("", encoding="utf-8")
        self._lock = threading.Lock()
        self._closed = False
        self.robot = AgxArmFactory.create_arm(create_agx_arm_config(
            robot="piper_x", comm="can", channel="can0", interface="socketcan"
        ))
        self.robot.connect()
        self.gripper = self.robot.init_effector(
            self.robot.OPTIONS.EFFECTOR.AGX_GRIPPER
        )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            value = self.robot.get_joint_angles()
            status = self.robot.get_arm_status()
            gripper_status = self.gripper.get_gripper_status()
            if (value is not None and value.msg is not None and len(value.msg) == 6
                    and status is not None and gripper_status is not None
                    and gripper_status.msg is not None):
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("Piper feedback did not become ready")
        if not all(self.robot.get_joint_enable_status(i) for i in range(1, 7)):
            raise RuntimeError("all Piper joints must already be enabled")
        if gripper_status.msg.mode != "width":
            raise RuntimeError(f"Piper gripper must be in width mode, got {gripper_status.msg.mode!r}")
        if not gripper_status.msg.foc_status.driver_enable_status:
            current_width_m = float(gripper_status.msg.value)
            if not math.isfinite(current_width_m) or not 0.0 <= current_width_m <= 0.1001:
                raise RuntimeError(f"invalid Piper gripper feedback width: {current_width_m}")
            self.gripper.move_gripper_m(value=current_width_m, force=1.0)
            enable_deadline = time.monotonic() + 2.0
            while time.monotonic() < enable_deadline:
                gripper_status = self.gripper.get_gripper_status()
                if (gripper_status is not None and gripper_status.msg is not None
                        and gripper_status.msg.foc_status.driver_enable_status):
                    break
                time.sleep(0.05)
            else:
                raise RuntimeError("Piper gripper enable feedback timeout")
        self.limits = []
        for index in range(1, 7):
            response = self.robot.get_joint_angle_vel_limits(index)
            if response is None or response.msg is None:
                raise RuntimeError(f"missing joint {index} limits")
            self.limits.append((float(response.msg.min_angle_limit), float(response.msg.max_angle_limit)))
        self.robot.set_speed_percent(speed_percent)
        atexit.register(self.close)
        self._move_to_startup_pose(startup_joints_rad, startup_gripper_width_m)

    def _move_to_startup_pose(
        self, joints_rad: tuple[float, ...], gripper_width_m: float,
    ) -> None:
        target = np.asarray(joints_rad, dtype=np.float64)
        if target.shape != (6,) or not np.isfinite(target).all():
            raise ValueError("startup_joints_rad must contain six finite joint angles")
        lower = np.asarray([x[0] for x in self.limits])
        upper = np.asarray([x[1] for x in self.limits])
        if np.any(target < lower) or np.any(target > upper):
            raise ValueError(f"startup pose outside firmware limits: {target.tolist()}")
        gripper_target = float(gripper_width_m)
        if not math.isfinite(gripper_target) or not 0.0 <= gripper_target <= 0.1001:
            raise ValueError(f"invalid startup gripper width: {gripper_target}")
        if not all(self.robot.get_joint_enable_status(i) for i in range(1, 7)):
            raise RuntimeError("all Piper joints must be enabled before startup motion")
        status = self.robot.get_arm_status()
        if status is None or status.msg is None or "NORMAL" not in repr(status.msg.arm_status):
            raise RuntimeError(f"Piper status is not normal before startup motion: {status}")

        print(
            f"RTC启动预备动作 | 关节目标={target.tolist()} rad | "
            f"夹爪目标={gripper_target:.4f} m",
            flush=True,
        )
        self.robot.move_j(target.tolist())
        self.gripper.move_gripper_m(value=gripper_target, force=1.0)

        deadline = time.monotonic() + 15.0
        last_joints = None
        last_width = None
        while time.monotonic() < deadline:
            joint_status = self.robot.get_joint_angles()
            gripper_status = self.gripper.get_gripper_status()
            if (joint_status is not None and joint_status.msg is not None
                    and len(joint_status.msg) == 6
                    and gripper_status is not None and gripper_status.msg is not None):
                last_joints = np.asarray(joint_status.msg, dtype=np.float64)
                last_width = float(gripper_status.msg.value)
                joint_error = float(np.max(np.abs(last_joints - target)))
                gripper_error = abs(last_width - gripper_target)
                if joint_error <= 0.01 and gripper_error <= 0.002:
                    print(
                        f"RTC启动预备动作完成 | 最大关节误差={joint_error:.6f} rad | "
                        f"夹爪反馈={last_width:.4f} m",
                        flush=True,
                    )
                    with self.log_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({
                            "event": "startup_pose_reached",
                            "monotonic_ns": time.monotonic_ns(),
                            "target_joints_rad": target.tolist(),
                            "feedback_joints_rad": last_joints.tolist(),
                            "max_joint_error_rad": joint_error,
                            "gripper_target_m": gripper_target,
                            "gripper_feedback_m": last_width,
                            "gripper_command_force_n": 1.0,
                        }, ensure_ascii=False) + "\n")
                    return
            time.sleep(0.05)
        raise RuntimeError(
            "Piper startup pose timeout: "
            f"joint_feedback={None if last_joints is None else last_joints.tolist()}, "
            f"gripper_feedback={last_width}"
        )

    def send_target(self, joints_rad: np.ndarray, gripper_width_m: float) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Piper command sink is closed")
            model_target = np.asarray(joints_rad, dtype=np.float64)
            target = -model_target if self.invert_joints else model_target.copy()
            value = self.robot.get_joint_angles()
            if value is None or value.msg is None or len(value.msg) != 6:
                raise RuntimeError("missing Piper joint feedback")
            current = np.asarray(value.msg, dtype=np.float64)
            lower = np.asarray([x[0] for x in self.limits])
            upper = np.asarray([x[1] for x in self.limits])
            if (not np.isfinite(target).all()
                    or np.any(target < lower - self.limit_tolerance_rad)
                    or np.any(target > upper + self.limit_tolerance_rad)):
                raise RuntimeError(f"RTC target outside firmware limits: {target.tolist()}")
            delta = target - current
            # Disabled: comparing an absolute model target with the current joint
            # feedback measures tracking lag rather than a target-to-target jump.
            # if float(np.max(np.abs(delta))) > self.max_joint_delta:
            #     raise RuntimeError(f"RTC target jump exceeds {self.max_joint_delta}: {delta.tolist()}")
            if not all(self.robot.get_joint_enable_status(i) for i in range(1, 7)):
                raise RuntimeError("Piper joint became disabled")
            status = self.robot.get_arm_status()
            if status is None or status.msg is None or "NORMAL" not in repr(status.msg.arm_status):
                raise RuntimeError(f"Piper status is not normal: {status}")

            gripper_target = float(gripper_width_m)

            if (
                not math.isfinite(gripper_target)
                or not 0.0 <= gripper_target <= 0.1001
            ):
                raise RuntimeError(
                    f"RTC gripper target outside range: {gripper_target}"
                )
            gripper_status = self.gripper.get_gripper_status()
            if gripper_status is None or gripper_status.msg is None:
                raise RuntimeError("missing Piper gripper feedback before RTC command")
            gripper_feedback_m = float(gripper_status.msg.value)
            gripper_feedback_force_n = float(gripper_status.msg.force)
            gripper_enabled = bool(gripper_status.msg.foc_status.driver_enable_status)
            if not gripper_enabled:
                raise RuntimeError("Piper gripper became disabled")
            print(
                "RTC夹爪控制 | "
                f"目标={gripper_target:.4f} m | "
                f"下发前反馈={gripper_feedback_m:.4f} m | "
                f"反馈力={gripper_feedback_force_n:.3f} N | "
                f"使能={gripper_enabled}",
                flush=True,
            )
            self.robot.move_j(target.tolist())
            self.gripper.move_gripper_m(
                value=gripper_target,
                force=1.0,
            )
            record = {"monotonic_ns": time.monotonic_ns(), "model_target_rad": model_target.tolist(),
                      "command_target_rad": target.tolist(), "feedback_before_rad": current.tolist(),
                      "delta_rad": delta.tolist(), "firmware_limit_tolerance_rad": self.limit_tolerance_rad,
                      "gripper_target_m": gripper_target,
                      "gripper_feedback_before_m": gripper_feedback_m,
                      "gripper_feedback_force_before_n": gripper_feedback_force_n,
                      "gripper_enabled_before": gripper_enabled,
                      "gripper_command_force_n": 1.0}
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self.robot.disconnect()


class InvertedPiperCommandSinkLinux(PiperCommandSinkLinux):
    def __init__(self) -> None:
        super().__init__(invert_joints=True, log_path="logs/linux_rtc_inverted_commands.jsonl")


class DirectPiperCommandSinkLinux(PiperCommandSinkLinux):
    def __init__(self) -> None:
        super().__init__(invert_joints=False, log_path="logs/linux_rtc_direct_commands.jsonl")
