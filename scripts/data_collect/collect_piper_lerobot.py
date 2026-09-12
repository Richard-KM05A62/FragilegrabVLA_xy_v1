"""Read-only Piper X demonstration recorder that writes a LeRobot dataset.

The leader-to-follower command path bypasses this computer.  Consequently the
computer cannot observe the true command sent by the leader.  This recorder
uses ``next_observation_proxy`` actions: frame k stores the follower state at
frame k + 1 as its absolute action target.  This is explicit in metadata and
can later be replaced by a real command-stream adapter without changing the
dataset schema.

This file deliberately never calls enable(), move_*, reset(), calibrate_*(),
or any gripper control method.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from pyAgxArm import AgxArmFactory, create_agx_arm_config


JOINT_NAMES = [f"joint_{index}_rad" for index in range(1, 7)]
STATE_NAMES = [*JOINT_NAMES, "gripper_closedness"]
DEFAULT_GRIPPER_OPEN_WIDTH_M = 0.1001  # Measured read-only on this arm; re-calibrate before deployment.


@dataclass(frozen=True)
class CaptureConfig:
    repo_id: str
    root: Path
    task: str
    fps: int
    episode_seconds: float
    episodes: int
    camera_index: int
    width: int
    height: int
    use_videos: bool
    gripper_open_width_m: float


def parse_args() -> CaptureConfig:
    parser = argparse.ArgumentParser(description="Read-only Piper X -> LeRobot recorder")
    parser.add_argument("--repo-id", default="local/piper_x_teleop")
    parser.add_argument("--root", type=Path, required=True, help="New, empty LeRobot dataset directory")
    parser.add_argument("--task", required=True, help="Natural-language task label for every recorded frame")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--episode-seconds", type=float, default=30.0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--images", action="store_true", help="Store image frames instead of encoded videos")
    parser.add_argument("--gripper-open-width-m", type=float, default=DEFAULT_GRIPPER_OPEN_WIDTH_M)
    args = parser.parse_args()
    if args.fps <= 0 or args.episode_seconds <= 0 or args.episodes <= 0:
        parser.error("fps, episode-seconds, and episodes must be positive")
    if args.gripper_open_width_m <= 0:
        parser.error("gripper-open-width-m must be positive")
    return CaptureConfig(
        repo_id=args.repo_id,
        root=args.root.resolve(),
        task=args.task,
        fps=args.fps,
        episode_seconds=args.episode_seconds,
        episodes=args.episodes,
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        use_videos=not args.images,
        gripper_open_width_m=args.gripper_open_width_m,
    )


def make_features(config: CaptureConfig) -> dict[str, dict[str, Any]]:
    image_dtype = "video" if config.use_videos else "image"
    return {
        "observation.images.base": {
            "dtype": image_dtype,
            "shape": (3, config.height, config.width),
            "names": ["channels", "height", "width"],
        },
        "observation.state": {"dtype": "float32", "shape": (7,), "names": [STATE_NAMES]},
        "action": {"dtype": "float32", "shape": (7,), "names": [STATE_NAMES]},
    }


def gripper_closedness(width_m: float, open_width_m: float) -> float:
    return float(1.0 - np.clip(width_m / open_width_m, 0.0, 1.0))


class PiperReadonlyReader:
    """CAN reader only. Connecting opens CAN; it does not enable motors."""

    def __init__(self, open_width_m: float) -> None:
        self.open_width_m = open_width_m
        cfg = create_agx_arm_config(
            robot="piper_x", comm="can", channel="0", interface="agx_cando"
        )
        self.robot = AgxArmFactory.create_arm(cfg)
        self.effector: Any | None = None

    def connect(self) -> dict[str, Any]:
        self.robot.connect()
        if not self.robot.is_connected() or not self.robot.is_ok():
            raise RuntimeError("CAN connection was established but Piper is not healthy")
        self.effector = self.robot.init_effector(self.robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        return {
            "firmware": self.robot.get_firmware(),
            "joint_enabled_at_start": [self.robot.get_joint_enable_status(i) for i in range(1, 7)],
            "arm_status_at_start": repr(self.robot.get_arm_status().msg),
        }

    def read_state(self) -> tuple[np.ndarray, dict[str, Any]]:
        if self.effector is None:
            raise RuntimeError("Reader is not connected")
        joints = self.robot.get_joint_angles()
        gripper = self.effector.get_gripper_status()
        if joints is None or joints.msg is None or len(joints.msg) != 6:
            raise RuntimeError("Missing six-axis joint feedback")
        if gripper is None or gripper.msg is None:
            raise RuntimeError("Missing gripper feedback; do not record a dataset without it")
        width_m = float(gripper.msg.value)
        state = np.asarray([*joints.msg, gripper_closedness(width_m, self.open_width_m)], dtype=np.float32)
        raw = {
            "wall_time_s": time.time(),
            "joint_feedback_timestamp_s": joints.timestamp,
            "joint_feedback_hz": joints.hz,
            "gripper_feedback_timestamp_s": gripper.timestamp,
            "gripper_feedback_hz": gripper.hz,
            "gripper_width_m": width_m,
            "gripper_force_n": float(gripper.msg.force),
            "gripper_mode": gripper.msg.mode,
        }
        return state, raw

    def close(self) -> None:
        self.robot.disconnect()


def open_camera(config: CaptureConfig) -> cv2.VideoCapture:
    camera = cv2.VideoCapture(config.camera_index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    camera.set(cv2.CAP_PROP_FPS, config.fps)
    if not camera.isOpened():
        raise RuntimeError(f"Cannot open camera index {config.camera_index}")
    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (actual_width, actual_height) != (config.width, config.height):
        raise RuntimeError(
            f"Camera returned {actual_width}x{actual_height}; requested {config.width}x{config.height}. "
            "Use the camera's native resolution consistently instead of resizing here."
        )
    return camera


def capture_sample(camera: cv2.VideoCapture, reader: PiperReadonlyReader) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    ok, bgr = camera.read()
    if not ok or bgr is None:
        raise RuntimeError("Camera frame read failed")
    # OpenCV yields BGR. LeRobot/OpenPI inputs are RGB uint8.
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    state, raw = reader.read_state()
    raw["image_capture_wall_time_s"] = time.time()
    return rgb, state, raw


def append_frame(
    dataset: LeRobotDataset,
    rgb: np.ndarray,
    state: np.ndarray,
    action: np.ndarray,
    task: str,
    frame_index: int,
    fps: int,
) -> None:
    dataset.add_frame(
        {
            "observation.images.base": rgb,
            "observation.state": state,
            "action": action,
            "task": task,
            # Omit timestamp intentionally: this LeRobot revision generates the canonical
            # frame_index / fps timeline internally, which passes its strict sync check.
        }
    )


def write_manifest(root: Path, config: CaptureConfig, device_info: dict[str, Any]) -> None:
    manifest = {
        "schema_version": 1,
        "capture": asdict(config) | {"root": str(config.root)},
        "device": device_info,
        "observation_state": "[joint_1..joint_6 rad, gripper_closedness]",
        "action": "absolute next follower feedback at t+1 (next_observation_proxy)",
        "action_source": "next_observation_proxy",
        "gripper_closedness": "1 - clip(width_m / gripper_open_width_m, 0, 1)",
        "image": "camera-native RGB uint8; no crop or resize in the collector",
        "openpi_training_transform": "joint actions become target-current deltas; gripper action remains absolute",
    }
    (root / "piper_capture_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def record_episode(
    dataset: LeRobotDataset,
    raw_log: Any,
    camera: cv2.VideoCapture,
    reader: PiperReadonlyReader,
    config: CaptureConfig,
    episode_index: int,
) -> int:
    period = 1.0 / config.fps
    deadline = time.monotonic() + config.episode_seconds
    next_tick = time.monotonic()
    pending: tuple[np.ndarray, np.ndarray, dict[str, Any]] | None = None
    frame_index = 0
    while time.monotonic() < deadline:
        remaining = next_tick - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        next_tick += period
        rgb, state, raw = capture_sample(camera, reader)
        raw |= {"episode_index": episode_index, "frame_index": frame_index}
        if pending is not None:
            prev_rgb, prev_state, prev_raw = pending
            append_frame(dataset, prev_rgb, prev_state, state, config.task, frame_index - 1, config.fps)
            raw_log.write(json.dumps(prev_raw) + "\n")
        pending = (rgb, state, raw)
        frame_index += 1
    if pending is not None:
        # No future sample exists for the final step. Keep the final target stationary.
        rgb, state, raw = pending
        append_frame(dataset, rgb, state, state.copy(), config.task, frame_index - 1, config.fps)
        raw_log.write(json.dumps(raw | {"final_action_is_current_state": True}) + "\n")
    if frame_index < 2:
        raise RuntimeError("Episode is too short; no valid transition was captured")
    dataset.save_episode()
    return frame_index


def main() -> None:
    config = parse_args()
    if config.root.exists():
        raise FileExistsError(f"Dataset root already exists: {config.root}. Choose a new root for this run.")
    config.root.parent.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset.create(
        repo_id=config.repo_id,
        fps=config.fps,
        root=config.root,
        robot_type="piper_x",
        features=make_features(config),
        use_videos=config.use_videos,
    )
    reader = PiperReadonlyReader(config.gripper_open_width_m)
    camera: cv2.VideoCapture | None = None
    try:
        device_info = reader.connect()
        write_manifest(config.root, config, device_info)
        camera = open_camera(config)
        raw_dir = config.root / "piper_raw"
        raw_dir.mkdir()
        for episode_index in range(config.episodes):
            log_path = raw_dir / f"episode_{episode_index:06d}.jsonl"
            print(f"Recording episode {episode_index + 1}/{config.episodes}; no commands are sent to Piper.")
            with log_path.open("w", encoding="utf-8") as raw_log:
                frames = record_episode(dataset, raw_log, camera, reader, config, episode_index)
            print(f"Saved episode {episode_index}: {frames} frames")
    finally:
        if camera is not None:
            camera.release()
        reader.close()


if __name__ == "__main__":
    main()
