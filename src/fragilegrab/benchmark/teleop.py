"""Local read-only Piper leader to MuJoCo position-teleoperation utilities."""

from __future__ import annotations

import json
import inspect
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TextIO

import numpy as np
import yaml


STATE_NAMES = [*[f"joint_{i}_rad" for i in range(1, 7)], "gripper_closedness"]
ACTION_NAMES = [*[f"joint_{i}_target_rad" for i in range(1, 7)], "gripper_target_m"]


@dataclass(frozen=True)
class LeaderSample:
    state: np.ndarray
    read_timestamp_ns: int
    raw: dict[str, Any]


class LatestLeaderReader:
    """Poll a read-only adapter independently and expose the newest complete sample."""

    def __init__(self, adapter: Any, poll_hz: float) -> None:
        if not np.isfinite(poll_hz) or poll_hz <= 0:
            raise ValueError("reader poll_hz must be positive")
        self.adapter = adapter
        self.period_s = 1.0 / poll_hz
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._sample: LeaderSample | None = None
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None
        self.start_metadata: dict[str, Any] | None = None

    def start(self) -> None:
        self.start_metadata = self.adapter.start()
        self._thread = threading.Thread(target=self._run, name="PiperTeleopReader", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            deadline = time.perf_counter()
            while not self._stop.is_set():
                try:
                    value = self.adapter.read_state()
                except RuntimeError as exc:
                    # pyAgxArm feedback caches may be empty for a short period
                    # immediately after connect.  Before the first complete
                    # sample, let latest(timeout_s=...) own the startup timeout.
                    if self._sample is None and str(exc).startswith("Missing "):
                        self._stop.wait(self.period_s)
                        continue
                    raise
                sample = LeaderSample(
                    state=np.asarray(value.value.state, dtype=np.float64).copy(),
                    read_timestamp_ns=int(value.timestamp_ns),
                    raw=dict(value.value.raw),
                )
                if sample.state.shape != (7,) or not np.isfinite(sample.state).all():
                    raise RuntimeError("Piper reader returned an invalid seven-dimensional state")
                with self._condition:
                    self._sample = sample
                    self._condition.notify_all()
                deadline += self.period_s
                self._stop.wait(max(0.0, deadline - time.perf_counter()))
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()
            self._stop.set()

    def latest(self, timeout_s: float | None = None) -> LeaderSample:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        with self._condition:
            while self._sample is None and self._error is None:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("Timed out waiting for the first Piper joint sample")
                self._condition.wait(remaining)
            if self._error is not None:
                raise RuntimeError("Piper reader failed") from self._error
            assert self._sample is not None
            return self._sample

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.adapter.close()


class PositionMapper:
    """Configured real q/closure to MuJoCo absolute actuator target mapping."""

    def __init__(self, config: dict[str, Any], model_limits: np.ndarray | None = None) -> None:
        joint = config["joints"]
        self.sign = self._six(joint["sign"], "joints.sign")
        self.offset = self._six(joint["offset_rad"], "joints.offset_rad")
        self.scale = self._six(joint["scale"], "joints.scale")
        limits = np.asarray(joint["limits_rad"], dtype=np.float64)
        if limits.shape != (6, 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
            raise ValueError("joints.limits_rad must be a finite 6x2 [min,max] array")
        if not np.all(np.isin(self.sign, (-1.0, 1.0))) or np.any(self.scale <= 0):
            raise ValueError("joint sign must be +/-1 and scale must be positive")
        self.joint_limits = limits
        gripper = config["gripper"]
        self.gripper_input_min = float(gripper["input_closedness_limits"][0])
        self.gripper_input_max = float(gripper["input_closedness_limits"][1])
        self.gripper_open_m = float(gripper["sim_open_m"])
        self.gripper_closed_m = float(gripper["sim_closed_m"])
        self.gripper_limits = np.asarray(gripper["limits_m"], dtype=np.float64)
        if (self.gripper_limits.shape != (2,) or self.gripper_input_min >= self.gripper_input_max
                or self.gripper_limits[0] >= self.gripper_limits[1]):
            raise ValueError("invalid gripper mapping limits")
        smoothing = config.get("smoothing", {})
        self.alpha = float(smoothing.get("alpha", 1.0)) if smoothing.get("enabled", False) else 1.0
        if not 0 < self.alpha <= 1:
            raise ValueError("smoothing.alpha must be in (0,1]")
        self._filtered: np.ndarray | None = None
        if model_limits is not None:
            model_limits = np.asarray(model_limits, dtype=np.float64)
            configured = np.vstack((self.joint_limits, self.gripper_limits))
            if model_limits.shape != (7, 2) or np.any(configured[:, 0] < model_limits[:, 0] - 1e-9) \
                    or np.any(configured[:, 1] > model_limits[:, 1] + 1e-9):
                raise ValueError("teleop configured limits exceed MuJoCo actuator ctrlrange")

    @staticmethod
    def _six(value: Any, name: str) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64)
        if result.shape != (6,) or not np.isfinite(result).all():
            raise ValueError(f"{name} must contain six finite values")
        return result

    def reset_filter(self, initial_action: np.ndarray | None = None) -> None:
        if initial_action is None:
            self._filtered = None
            return
        initial_action = np.asarray(initial_action, dtype=np.float64)
        if initial_action.shape != (7,) or not np.isfinite(initial_action).all():
            raise ValueError("initial filter action must be finite float[7]")
        self._filtered = initial_action.copy()

    def map(self, leader_state: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        leader_state = np.asarray(leader_state, dtype=np.float64)
        if leader_state.shape != (7,) or not np.isfinite(leader_state).all():
            raise ValueError("leader state must be finite float[7]")
        raw_joints = self.sign * self.scale * leader_state[:6] + self.offset
        joints = np.clip(raw_joints, self.joint_limits[:, 0], self.joint_limits[:, 1])
        closure = np.clip(
            (leader_state[6] - self.gripper_input_min)
            / (self.gripper_input_max - self.gripper_input_min),
            0.0,
            1.0,
        )
        raw_gripper = self.gripper_open_m + closure * (self.gripper_closed_m - self.gripper_open_m)
        target = np.r_[joints, np.clip(raw_gripper, *self.gripper_limits)]
        if self._filtered is None:
            filtered = target
        else:
            filtered = self.alpha * target + (1.0 - self.alpha) * self._filtered
        executed = np.r_[
            np.clip(filtered[:6], self.joint_limits[:, 0], self.joint_limits[:, 1]),
            np.clip(filtered[6], *self.gripper_limits),
        ]
        self._filtered = executed.copy()
        return executed, {
            "unclamped_target": np.r_[raw_joints, raw_gripper].tolist(),
            "clamped_before_filter": target.tolist(),
            "smoothing_alpha": self.alpha,
            "executed_action": executed.tolist(),
        }


def load_teleop_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("version") != "piper_mujoco_teleop/0.1":
        raise ValueError("teleop config version must be piper_mujoco_teleop/0.1")
    for section in ("reader", "loops", "mapping", "recording"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"missing teleop config section: {section}")
    return config


def latency_summary_ms(values_ns: list[int]) -> dict[str, float | int | None]:
    if not values_ns:
        return {"samples": 0, "mean_ms": None, "p95_ms": None}
    values = np.asarray(values_ns, dtype=np.float64) / 1e6
    return {"samples": len(values_ns), "mean_ms": float(values.mean()), "p95_ms": float(np.percentile(values, 95))}


class TeleopEpisodeRecorder:
    """One episode per LeRobot dataset, plus causal timing and reset sidecars."""

    def __init__(self, output_root: Path, repo_id: str, fps: int) -> None:
        self.output_root = output_root.resolve()
        self.repo_id = repo_id
        self.fps = fps
        self.dataset: Any | None = None
        self.episode_dir: Path | None = None
        self.log: TextIO | None = None
        self.frame_count = 0
        self.metadata: dict[str, Any] | None = None

    @property
    def active(self) -> bool:
        return self.metadata is not None

    def start(self, metadata: dict[str, Any]) -> Path:
        if self.active:
            raise RuntimeError("an episode is already recording")
        stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000_000:09d}"
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.episode_dir = self.output_root / f"episode_{stamp}.incomplete"
        self.episode_dir.mkdir()
        (self.episode_dir / ".fragilegrab_teleop_episode").write_text("0.1\n", encoding="utf-8")
        self.log = (self.episode_dir / "teleop_timeline.jsonl").open("w", encoding="utf-8")
        self.metadata = metadata
        self.frame_count = 0
        self.dataset = None
        return self.episode_dir

    def _create_dataset(self, front: np.ndarray, wrist: np.ndarray) -> None:
        try:
            try:
                from lerobot.datasets.lerobot_dataset import LeRobotDataset
            except ImportError:
                from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError("teleop recording requires lerobot in the MuJoCo Python environment") from exc

        def image_feature(image: np.ndarray) -> dict[str, Any]:
            h, w, c = image.shape
            if image.dtype != np.uint8 or c != 3:
                raise ValueError("teleop images must be RGB uint8 HWC")
            return {"dtype": "video", "shape": (3, h, w), "names": ["channels", "height", "width"]}

        assert self.episode_dir is not None
        self.dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=self.fps,
            root=self.episode_dir / "lerobot",
            robot_type="piper_x_mujoco_teleop",
            features={
                "observation.images.base": image_feature(front),
                "observation.images.wrist": image_feature(wrist),
                "observation.state": {"dtype": "float32", "shape": (7,), "names": [STATE_NAMES]},
                "action": {"dtype": "float32", "shape": (7,), "names": [ACTION_NAMES]},
            },
            use_videos=True,
        )

    def append(
        self,
        *,
        front: np.ndarray,
        wrist: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        task: str,
        timing: dict[str, int],
        leader_state: np.ndarray,
        mapping: dict[str, Any],
    ) -> None:
        if not self.active:
            return
        if self.dataset is None:
            self._create_dataset(front, wrist)
        frame = {
            "observation.images.base": front,
            "observation.images.wrist": wrist,
            "observation.state": np.asarray(state, dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
        }
        if "task" in inspect.signature(self.dataset.add_frame).parameters:
            self.dataset.add_frame(frame, task=task)
        else:
            self.dataset.add_frame({**frame, "task": task})
        assert self.log is not None
        self.log.write(json.dumps({
            "frame_index": self.frame_count,
            **timing,
            "leader_state": np.asarray(leader_state).tolist(),
            "sim_observation_state": np.asarray(state).tolist(),
            "action": np.asarray(action).tolist(),
            "mapping": mapping,
        }, ensure_ascii=False, allow_nan=False) + "\n")
        self.frame_count += 1

    def finish(self, outcome: str, termination_reason: str, latency: dict[str, Any]) -> Path:
        if outcome not in {"success", "failure", "unlabeled"}:
            raise ValueError("invalid episode outcome")
        if not self.active or self.episode_dir is None:
            raise RuntimeError("no active episode")
        if self.frame_count == 0 or self.dataset is None:
            raise RuntimeError("cannot save an episode with no frames")
        self.dataset.save_episode()
        assert self.log is not None and self.metadata is not None
        self.log.close()
        self.log = None
        manifest = {
            **self.metadata,
            "frames": self.frame_count,
            "result": {"success": outcome == "success", "label": outcome},
            "termination_reason": termination_reason,
            "latency": latency,
            "storage": {"lerobot_dataset": "lerobot", "timeline": "teleop_timeline.jsonl"},
        }
        (self.episode_dir / "episode_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        final = self.episode_dir.with_suffix("")
        self.episode_dir.rename(final)
        self._clear()
        return final

    def discard(self) -> None:
        if not self.active or self.episode_dir is None:
            return
        if self.log is not None:
            self.log.close()
        marker = self.episode_dir / ".fragilegrab_teleop_episode"
        if self.episode_dir.parent != self.output_root or not marker.is_file() or not self.episode_dir.name.endswith(".incomplete"):
            raise RuntimeError("refusing to discard an unrecognized directory")
        shutil.rmtree(self.episode_dir)
        self._clear()

    def _clear(self) -> None:
        self.dataset = None
        self.episode_dir = None
        self.log = None
        self.metadata = None
        self.frame_count = 0


def sim_observation_state(env: Any) -> np.ndarray:
    q = env.data.qpos[env.qadr]
    closure = np.clip(
        (env.robot["gripper_actuator_open_m"] - q[6])
        / (env.robot["gripper_actuator_open_m"] - env.robot["gripper_actuator_closed_m"]),
        0.0,
        1.0,
    )
    return np.asarray([*q[:6], closure], dtype=np.float32)


def paced_deadline(previous: float, hz: float, now: Callable[[], float] = time.perf_counter) -> float:
    """Advance a periodic deadline without coupling it to other loop rates."""
    deadline = previous + 1.0 / hz
    current = now()
    while deadline <= current:
        deadline += 1.0 / hz
    return deadline
