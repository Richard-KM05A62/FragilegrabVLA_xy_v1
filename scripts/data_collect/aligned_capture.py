"""Time-aligned, read-only capture core for Piper X demonstrations.

This module intentionally contains no robot control calls.  The two camera
adapters are placeholders: only their ``read_rgb`` implementations need to be
filled in once the camera SDKs are installed.  Every source is timestamped in
the same computer ``time.monotonic_ns`` clock domain at data arrival.
"""

from __future__ import annotations

import ctypes
import json
import os
import queue
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Iterator, TextIO, TypeVar

import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from pyAgxArm import AgxArmFactory, create_agx_arm_config


FPS = 30
PERIOD_NS = round(1_000_000_000 / FPS)
MAX_SKEW_NS = 30_000_000       # A selected source sample must be within ±30 ms.
STARTUP_DELAY_NS = 100_000_000  # T0 = common_start + 100 ms.
BUFFER_WINDOW_NS = 2_000_000_000
DEFAULT_GRIPPER_OPEN_WIDTH_M = 0.1001
STATE_NAMES = [*[f"joint_{index}_rad" for index in range(1, 7)], "gripper_closedness"]
DEFAULT_ORBBEC_SDK_ROOT = Path(
    r"C:\Users\13302\Downloads\OrbbecSDK_C_C++_v1.10.37_20260707_3f75820b8_win_x64_release\OrbbecSDK_v1.10.37"
)

T = TypeVar("T")


@dataclass(frozen=True)
class TimedValue(Generic[T]):
    """A value expressed only in the host computer's monotonic clock domain."""

    value: T
    timestamp_ns: int


@dataclass(frozen=True)
class ArmState:
    # [joint_1_rad, ..., joint_6_rad, gripper_closedness]
    state: np.ndarray
    raw: dict[str, Any]


def gripper_closedness(width_m: float, open_width_m: float) -> float:
    return float(1.0 - np.clip(width_m / open_width_m, 0.0, 1.0))


@dataclass(frozen=True)
class AlignedObservation:
    target_timestamp_ns: int
    base_rgb: np.ndarray             # D435i, external/base camera, RGB uint8.
    wrist_rgb: np.ndarray            # DaBai DC1, wrist camera, RGB uint8.
    arm_state: np.ndarray            # float32[7]
    base_skew_ns: int
    wrist_skew_ns: int
    arm_skew_ns: int
    arm_raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProducerFailure:
    source: str
    error: BaseException


class TimeWindowBuffer(Generic[T]):
    """Thread-safe history retaining only the latest BUFFER_WINDOW_NS."""

    def __init__(self, window_ns: int = BUFFER_WINDOW_NS) -> None:
        self._window_ns = window_ns
        self._values: deque[TimedValue[T]] = deque()
        self._lock = threading.Lock()

    def append(self, value: T, timestamp_ns: int) -> None:
        item = TimedValue(value=value, timestamp_ns=timestamp_ns)
        with self._lock:
            self._values.append(item)
            cutoff = timestamp_ns - self._window_ns
            while self._values and self._values[0].timestamp_ns < cutoff:
                self._values.popleft()

    def first_timestamp_ns(self) -> int | None:
        with self._lock:
            return self._values[0].timestamp_ns if self._values else None

    def nearest(self, target_timestamp_ns: int) -> TimedValue[T] | None:
        with self._lock:
            if not self._values:
                return None
            # Buffers are short (two seconds), so linear search is clear and cheap.
            return min(self._values, key=lambda item: abs(item.timestamp_ns - target_timestamp_ns))


class CameraAdapter(ABC):
    """Common RGB-only interface; implementations own their SDK lifecycle."""

    camera_name: str

    @abstractmethod
    def start(self) -> None:
        """Open one camera, preferably selected by its serial number."""

    @abstractmethod
    def read_rgb(self) -> TimedValue[np.ndarray]:
        """Block for one RGB frame and timestamp its arrival with monotonic_ns."""

    @abstractmethod
    def close(self) -> None:
        """Release SDK resources."""


class RealSenseD435iAdapter(CameraAdapter):
    """Intel RealSense D435i RGB source through pyrealsense2.

    The requested output is RGB8. Some Windows driver/profile combinations only
    expose BGR8 through librealsense; that fallback is converted here so the
    collector interface always remains RGB uint8 HWC.
    """

    camera_name = "intel_realsense_d435i"

    def __init__(self, serial_number: str | None = None) -> None:
        self.serial_number = serial_number
        self._rs: Any | None = None
        self._pipeline: Any | None = None
        self._started = False
        self._output_format: Any | None = None

    def start(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "D435i requires pyrealsense2 in D:/vscode/opencv/venv. "
                "Install the Intel RealSense Python SDK before using this adapter."
            ) from exc

        self._rs = rs
        self._pipeline = rs.pipeline()
        config = rs.config()
        if self.serial_number:
            config.enable_device(self.serial_number)
        # Prefer native RGB output. If unavailable, use BGR and convert in read_rgb.
        try:
            config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
            self._pipeline.start(config)
            self._output_format = rs.format.rgb8
        except RuntimeError:
            # A failed start may retain pipeline state in some librealsense builds.
            self._pipeline = rs.pipeline()
            config = rs.config()
            if self.serial_number:
                config.enable_device(self.serial_number)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self._pipeline.start(config)
            self._output_format = rs.format.bgr8
        self._started = True

    def read_rgb(self) -> TimedValue[np.ndarray]:
        if not self._started or self._pipeline is None or self._rs is None:
            raise RuntimeError("D435i adapter has not been started")
        while True:
            frames = self._pipeline.wait_for_frames(timeout_ms=1_000)
            host_arrival_ns = time.monotonic_ns()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            image = np.asanyarray(color_frame.get_data())
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise RuntimeError(f"D435i returned unsupported color image shape/dtype: {image.shape} {image.dtype}")
            if self._output_format == self._rs.format.bgr8:
                # Make an owned RGB copy before the SDK releases its frame buffer.
                image = image[:, :, ::-1]
            return TimedValue(np.array(image, dtype=np.uint8, copy=True, order="C"), host_arrival_ns)

    def close(self) -> None:
        if self._pipeline is not None and self._started:
            try:
                self._pipeline.stop()
            finally:
                self._started = False


class OrbbecDaBaiDC1Adapter(CameraAdapter):
    """DaBai DC1 RGB source through the installed Orbbec C SDK v1.10.37.

    It intentionally uses the host software time taken immediately after
    ``ob_pipeline_wait_for_frameset`` returns, not an Orbbec frame timestamp.
    """

    camera_name = "orbbec_dabai_dc1"

    # Constants verified against this SDK's ObTypes.h.
    _OB_SENSOR_COLOR = 2
    _OB_FORMAT_RGB = 22
    _OB_WIDTH_ANY = 0
    _OB_HEIGHT_ANY = 0
    _OB_FPS_ANY = 0

    def __init__(self, serial_number: str | None = None, sdk_root: Path = DEFAULT_ORBBEC_SDK_ROOT) -> None:
        self.serial_number = serial_number
        self.sdk_root = sdk_root
        self._dll: ctypes.WinDLL | None = None
        self._dll_directory: Any | None = None
        self._context: ctypes.c_void_p | None = None
        self._device: ctypes.c_void_p | None = None
        self._pipeline: ctypes.c_void_p | None = None
        self._config: ctypes.c_void_p | None = None
        self._profiles: ctypes.c_void_p | None = None
        self._color_profile: ctypes.c_void_p | None = None

    @staticmethod
    def _error_pointer() -> ctypes.c_void_p:
        return ctypes.c_void_p()

    def _bind(self, name: str, restype: Any, *argtypes: Any) -> None:
        assert self._dll is not None
        fn = getattr(self._dll, name)
        fn.restype = restype
        fn.argtypes = list(argtypes)

    def _load_sdk(self) -> None:
        if self._dll is not None:
            return
        dll_dir = self.sdk_root / "SDK" / "lib"
        dll_path = dll_dir / "OrbbecSDK.dll"
        if not dll_path.is_file():
            raise FileNotFoundError(f"OrbbecSDK.dll not found: {dll_path}")
        # Keep this handle alive for the adapter lifetime so dependent DLL lookup works.
        self._dll_directory = os.add_dll_directory(str(dll_dir))
        self._dll = ctypes.WinDLL(str(dll_path))
        void_p = ctypes.c_void_p
        err_pp = ctypes.POINTER(void_p)
        uint32 = ctypes.c_uint32

        self._bind("ob_error_message", ctypes.c_char_p, void_p)
        self._bind("ob_error_function", ctypes.c_char_p, void_p)
        self._bind("ob_delete_error", None, void_p)
        self._bind("ob_create_context", void_p, err_pp)
        self._bind("ob_delete_context", None, void_p, err_pp)
        self._bind("ob_query_device_list", void_p, void_p, err_pp)
        self._bind("ob_delete_device_list", None, void_p, err_pp)
        self._bind("ob_device_list_get_device_by_serial_number", void_p, void_p, ctypes.c_char_p, err_pp)
        self._bind("ob_delete_device", None, void_p, err_pp)
        self._bind("ob_create_pipeline", void_p, err_pp)
        self._bind("ob_create_pipeline_with_device", void_p, void_p, err_pp)
        self._bind("ob_delete_pipeline", None, void_p, err_pp)
        self._bind("ob_pipeline_get_stream_profile_list", void_p, void_p, ctypes.c_int, err_pp)
        self._bind("ob_pipeline_start_with_config", None, void_p, void_p, err_pp)
        self._bind("ob_pipeline_stop", None, void_p, err_pp)
        self._bind("ob_pipeline_wait_for_frameset", void_p, void_p, uint32, err_pp)
        self._bind("ob_create_config", void_p, err_pp)
        self._bind("ob_delete_config", None, void_p, err_pp)
        self._bind("ob_config_enable_stream", None, void_p, void_p, err_pp)
        self._bind(
            "ob_stream_profile_list_get_video_stream_profile",
            void_p,
            void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            err_pp,
        )
        self._bind("ob_delete_stream_profile_list", None, void_p, err_pp)
        self._bind("ob_delete_stream_profile", None, void_p, err_pp)
        self._bind("ob_frameset_color_frame", void_p, void_p, err_pp)
        self._bind("ob_delete_frame", None, void_p, err_pp)
        self._bind("ob_frame_data", void_p, void_p, err_pp)
        self._bind("ob_frame_data_size", uint32, void_p, err_pp)
        self._bind("ob_video_frame_width", uint32, void_p, err_pp)
        self._bind("ob_video_frame_height", uint32, void_p, err_pp)

    def _raise_if_error(self, error: ctypes.c_void_p, api_name: str) -> None:
        if not error.value:
            return
        assert self._dll is not None
        try:
            function = self._dll.ob_error_function(error)
            message = self._dll.ob_error_message(error)
            function_text = function.decode(errors="replace") if function else api_name
            message_text = message.decode(errors="replace") if message else "unknown SDK error"
        finally:
            self._dll.ob_delete_error(error)
        raise RuntimeError(f"Orbbec {function_text}: {message_text}")

    def _call(self, name: str, *args: Any) -> Any:
        assert self._dll is not None
        error = self._error_pointer()
        result = getattr(self._dll, name)(*args, ctypes.byref(error))
        self._raise_if_error(error, name)
        return result

    def _try_color_profile(self, width: int, height: int, fps: int) -> ctypes.c_void_p | None:
        assert self._dll is not None and self._profiles is not None
        error = self._error_pointer()
        profile = self._dll.ob_stream_profile_list_get_video_stream_profile(
            self._profiles, width, height, self._OB_FORMAT_RGB, fps, ctypes.byref(error)
        )
        if error.value:
            self._dll.ob_delete_error(error)
            return None
        return profile or None

    def start(self) -> None:
        self._load_sdk()
        if self.serial_number:
            self._context = self._call("ob_create_context")
            devices = self._call("ob_query_device_list", self._context)
            try:
                self._device = self._call(
                    "ob_device_list_get_device_by_serial_number", devices, self.serial_number.encode("utf-8")
                )
            finally:
                self._call("ob_delete_device_list", devices)
            self._pipeline = self._call("ob_create_pipeline_with_device", self._device)
        else:
            self._pipeline = self._call("ob_create_pipeline")

        self._config = self._call("ob_create_config")
        self._profiles = self._call("ob_pipeline_get_stream_profile_list", self._pipeline, self._OB_SENSOR_COLOR)
        # Prefer the dataset's planned 640x480@30. Only RGB profiles are accepted.
        self._color_profile = (
            self._try_color_profile(640, 480, 30)
            or self._try_color_profile(self._OB_WIDTH_ANY, self._OB_HEIGHT_ANY, 30)
            or self._try_color_profile(self._OB_WIDTH_ANY, self._OB_HEIGHT_ANY, self._OB_FPS_ANY)
        )
        if self._color_profile is None:
            raise RuntimeError("DaBai DC1 exposes no RGB color-stream profile")
        self._call("ob_config_enable_stream", self._config, self._color_profile)
        self._call("ob_pipeline_start_with_config", self._pipeline, self._config)

    def read_rgb(self) -> TimedValue[np.ndarray]:
        if self._pipeline is None:
            raise RuntimeError("DaBai DC1 adapter has not been started")
        while True:
            frameset = self._call("ob_pipeline_wait_for_frameset", self._pipeline, 100)
            if not frameset:
                continue
            host_arrival_ns = time.monotonic_ns()
            try:
                color = self._call("ob_frameset_color_frame", frameset)
                if not color:
                    continue
                width = int(self._call("ob_video_frame_width", color))
                height = int(self._call("ob_video_frame_height", color))
                data_size = int(self._call("ob_frame_data_size", color))
                expected_size = width * height * 3
                if data_size != expected_size:
                    raise RuntimeError(
                        f"DaBai DC1 RGB buffer has {data_size} bytes; expected {expected_size} for {width}x{height} RGB"
                    )
                data_ptr = self._call("ob_frame_data", color)
                if not data_ptr:
                    raise RuntimeError("DaBai DC1 returned a null RGB buffer")
                # Copy before deleting frameset; SDK owns the source buffer.
                rgb = np.frombuffer(ctypes.string_at(data_ptr, data_size), dtype=np.uint8).reshape(height, width, 3)
                return TimedValue(rgb.copy(), host_arrival_ns)
            finally:
                # As in the official sample, the frameset owns/reclaims its color frame.
                self._call("ob_delete_frame", frameset)

    def close(self) -> None:
        # Cleanup is intentionally best-effort so an earlier startup failure does not mask its cause.
        try:
            if self._pipeline is not None:
                self._call("ob_pipeline_stop", self._pipeline)
        except Exception:
            pass
        for name, attribute in (
            ("ob_delete_stream_profile", "_color_profile"),
            ("ob_delete_stream_profile_list", "_profiles"),
            ("ob_delete_config", "_config"),
            ("ob_delete_pipeline", "_pipeline"),
            ("ob_delete_device", "_device"),
            ("ob_delete_context", "_context"),
        ):
            value = getattr(self, attribute)
            if value is not None and self._dll is not None:
                try:
                    self._call(name, value)
                except Exception:
                    pass
                setattr(self, attribute, None)
        if self._dll_directory is not None:
            self._dll_directory.close()
            self._dll_directory = None


class PiperCanAdapter:
    """Piper CAN reader integrated here; it never enables or commands the arm."""

    def __init__(self, gripper_open_width_m: float = DEFAULT_GRIPPER_OPEN_WIDTH_M) -> None:
        self._gripper_open_width_m = gripper_open_width_m
        cfg = create_agx_arm_config(
            robot="piper_x", comm="can", channel="0", interface="agx_cando"
        )
        self._robot = AgxArmFactory.create_arm(cfg)
        self._effector: Any | None = None

    def start(self) -> dict[str, Any]:
        self._robot.connect()
        if not self._robot.is_connected() or not self._robot.is_ok():
            raise RuntimeError("CAN connection was established but Piper is not healthy")
        self._effector = self._robot.init_effector(self._robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        return {
            "firmware": self._robot.get_firmware(),
            "joint_enabled_at_start": [self._robot.get_joint_enable_status(i) for i in range(1, 7)],
            "arm_status_at_start": repr(self._robot.get_arm_status().msg),
        }

    def read_state(self) -> TimedValue[ArmState]:
        if self._effector is None:
            raise RuntimeError("Piper CAN reader has not been started")
        joints = self._robot.get_joint_angles()
        gripper = self._effector.get_gripper_status()
        if joints is None or joints.msg is None or len(joints.msg) != 6:
            raise RuntimeError("Missing six-axis joint feedback")
        if gripper is None or gripper.msg is None:
            raise RuntimeError("Missing gripper feedback; do not record a 7-D dataset without it")
        width_m = float(gripper.msg.value)
        state = np.asarray(
            [*joints.msg, gripper_closedness(width_m, self._gripper_open_width_m)], dtype=np.float32
        )
        raw = {
            # These source timestamps are diagnostic only, not used for synchronization.
            "joint_sdk_timestamp_s": joints.timestamp,
            "gripper_sdk_timestamp_s": gripper.timestamp,
            "gripper_width_m": width_m,
            "gripper_force_n": float(gripper.msg.force),
            "gripper_mode": gripper.msg.mode,
        }
        # The current pyAgxArm getter provides the latest parsed CAN cache.
        # This is deliberately the host time at which that state becomes available.
        return TimedValue(ArmState(state=state, raw=raw), time.monotonic_ns())

    def close(self) -> None:
        self._robot.disconnect()


def _camera_worker(
    adapter: CameraAdapter,
    buffer: TimeWindowBuffer[np.ndarray],
    stop_event: threading.Event,
    failures: queue.Queue[ProducerFailure],
) -> None:
    """Adapters timestamp immediately after their SDK returns a frame."""
    try:
        adapter.start()
        while not stop_event.is_set():
            frame = adapter.read_rgb()
            rgb = frame.value
            if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError(f"{adapter.camera_name} must return RGB uint8 HWC, got {type(rgb)} {getattr(rgb, 'shape', None)}")
            buffer.append(rgb, frame.timestamp_ns)
    except BaseException as exc:
        failures.put(ProducerFailure(adapter.camera_name, exc))
        stop_event.set()
    finally:
        adapter.close()


def _can_worker(
    adapter: PiperCanAdapter,
    buffer: TimeWindowBuffer[ArmState],
    stop_event: threading.Event,
    failures: queue.Queue[ProducerFailure],
) -> None:
    """Poll latest CAN feedback in a dedicated thread and timestamp availability."""
    try:
        adapter.start()
        while not stop_event.is_set():
            arm_state = adapter.read_state()
            buffer.append(arm_state.value, arm_state.timestamp_ns)
            # Measured feedback is about 50 Hz. Avoid spinning on the cached SDK value.
            time.sleep(0.01)
    except BaseException as exc:
        failures.put(ProducerFailure("piper_can", exc))
        stop_event.set()
    finally:
        adapter.close()


class AlignedScheduler:
    """Produces only complete, three-stream observations at a fixed logical FPS."""

    def __init__(
        self,
        base: TimeWindowBuffer[np.ndarray],
        wrist: TimeWindowBuffer[np.ndarray],
        arm: TimeWindowBuffer[ArmState],
    ) -> None:
        self.base = base
        self.wrist = wrist
        self.arm = arm

    def _select(self, target_ns: int) -> AlignedObservation | None:
        base = self.base.nearest(target_ns)
        wrist = self.wrist.nearest(target_ns)
        arm = self.arm.nearest(target_ns)
        if base is None or wrist is None or arm is None:
            return None
        base_skew = base.timestamp_ns - target_ns
        wrist_skew = wrist.timestamp_ns - target_ns
        arm_skew = arm.timestamp_ns - target_ns
        if max(abs(base_skew), abs(wrist_skew), abs(arm_skew)) > MAX_SKEW_NS:
            return None
        return AlignedObservation(
            target_timestamp_ns=target_ns,
            base_rgb=base.value,
            wrist_rgb=wrist.value,
            arm_state=arm.value.state,
            base_skew_ns=base_skew,
            wrist_skew_ns=wrist_skew,
            arm_skew_ns=arm_skew,
            arm_raw=arm.value.raw,
        )

    def wait_for_t0(self, stop_event: threading.Event) -> int:
        """Return the first valid target time: common_start + 100 ms."""
        while not stop_event.is_set():
            first_timestamps = [
                self.base.first_timestamp_ns(),
                self.wrist.first_timestamp_ns(),
                self.arm.first_timestamp_ns(),
            ]
            if all(ts is not None for ts in first_timestamps):
                common_start_ns = max(ts for ts in first_timestamps if ts is not None)
                target_ns = common_start_ns + STARTUP_DELAY_NS
                # Waiting past the right edge makes nearest-neighbour matching symmetric.
                while not stop_event.is_set() and time.monotonic_ns() < target_ns + MAX_SKEW_NS:
                    time.sleep(0.001)
                if self._select(target_ns) is not None:
                    return target_ns
                # A startup source may have been late; retry from the newest common start.
            time.sleep(0.005)
        raise RuntimeError("Capture stopped before all sources produced a valid first observation")

    def observations(self, stop_event: threading.Event) -> Iterator[AlignedObservation]:
        target_ns = self.wait_for_t0(stop_event)
        while not stop_event.is_set():
            # A fixed 30 ms output delay permits nearest samples from both sides of target_ns.
            while not stop_event.is_set() and time.monotonic_ns() < target_ns + MAX_SKEW_NS:
                time.sleep(0.001)
            observation = self._select(target_ns)
            if observation is not None:
                yield observation
            # Missing ticks are deliberately not filled with stale images.
            target_ns += PERIOD_NS


def start_sources(
    base_camera: CameraAdapter,
    wrist_camera: CameraAdapter,
    piper: PiperCanAdapter,
    stop_event: threading.Event,
) -> tuple[AlignedScheduler, list[threading.Thread], queue.Queue[ProducerFailure]]:
    """Start producer threads. The caller owns stop_event and joins the threads."""
    base_buffer: TimeWindowBuffer[np.ndarray] = TimeWindowBuffer()
    wrist_buffer: TimeWindowBuffer[np.ndarray] = TimeWindowBuffer()
    arm_buffer: TimeWindowBuffer[ArmState] = TimeWindowBuffer()
    failures: queue.Queue[ProducerFailure] = queue.Queue()
    threads = [
        threading.Thread(
            name="D435iReader", target=_camera_worker, args=(base_camera, base_buffer, stop_event, failures), daemon=True
        ),
        threading.Thread(
            name="DaBaiDC1Reader", target=_camera_worker, args=(wrist_camera, wrist_buffer, stop_event, failures), daemon=True
        ),
        threading.Thread(
            name="PiperCanReader", target=_can_worker, args=(piper, arm_buffer, stop_event, failures), daemon=True
        ),
    ]
    for thread in threads:
        thread.start()
    return AlignedScheduler(base_buffer, wrist_buffer, arm_buffer), threads, failures


def lerobot_features(base_rgb: np.ndarray, wrist_rgb: np.ndarray, use_videos: bool) -> dict[str, dict[str, Any]]:
    """Build features from the cameras' actual native resolutions, with no resize/crop."""
    image_dtype = "video" if use_videos else "image"

    def image_feature(rgb: np.ndarray) -> dict[str, Any]:
        height, width, channels = rgb.shape
        if channels != 3:
            raise ValueError(f"Expected RGB HWC image, got {rgb.shape}")
        return {
            "dtype": image_dtype,
            "shape": (3, height, width),
            "names": ["channels", "height", "width"],
        }

    return {
        "observation.images.base": image_feature(base_rgb),
        "observation.images.wrist": image_feature(wrist_rgb),
        "observation.state": {"dtype": "float32", "shape": (7,), "names": [STATE_NAMES]},
        "action": {"dtype": "float32", "shape": (7,), "names": [STATE_NAMES]},
    }


class LeRobotAlignedEpisodeWriter:
    """Writes aligned samples directly to one LeRobot episode.

    A frame at t uses the next aligned follower state as its action label. The
    final frame uses its own state as action, per the agreed boundary policy.
    """

    def __init__(self, repo_id: str, root: Path, task: str, use_videos: bool = True) -> None:
        self.repo_id = repo_id
        self.root = root.resolve()
        self.task = task
        self.use_videos = use_videos
        self.dataset: LeRobotDataset | None = None
        self.raw_log: TextIO | None = None
        self.frame_count = 0

    def _create(self, first: AlignedObservation) -> None:
        if self.root.exists():
            raise FileExistsError(f"Dataset root already exists: {self.root}")
        self.root.parent.mkdir(parents=True, exist_ok=True)
        self.dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=FPS,
            root=self.root,
            robot_type="piper_x",
            features=lerobot_features(first.base_rgb, first.wrist_rgb, self.use_videos),
            use_videos=self.use_videos,
        )
        raw_dir = self.root / "piper_raw"
        raw_dir.mkdir()
        self.raw_log = (raw_dir / "episode_000000_alignment.jsonl").open("w", encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "fps": FPS,
            "clock": "host time.monotonic_ns() only",
            "base_camera": "Intel RealSense D435i",
            "wrist_camera": "Orbbec DaBai DC1",
            "max_alignment_skew_ms": MAX_SKEW_NS / 1_000_000,
            "action_source": "next_observation_proxy",
            "state": "[joint_1..joint_6 rad, gripper_closedness]",
            "action": "absolute next follower state; final frame uses its own state",
            "gripper_closedness": "1 - clip(width_m / 0.1001, 0, 1)",
            "image": "native-resolution RGB uint8; no crop or resize in collector",
        }
        (self.root / "piper_capture_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def append(self, observation: AlignedObservation, action: np.ndarray) -> None:
        if self.dataset is None:
            self._create(observation)
        assert self.dataset is not None
        if observation.base_rgb.shape != tuple(self.dataset.features["observation.images.base"]["shape"])[1:] + (3,):
            raise ValueError("D435i resolution changed within an episode")
        if observation.wrist_rgb.shape != tuple(self.dataset.features["observation.images.wrist"]["shape"])[1:] + (3,):
            raise ValueError("DaBai DC1 resolution changed within an episode")
        self.dataset.add_frame(
            {
                "observation.images.base": observation.base_rgb,
                "observation.images.wrist": observation.wrist_rgb,
                "observation.state": observation.arm_state.astype(np.float32, copy=False),
                "action": action.astype(np.float32, copy=False),
                "task": self.task,
                # Omit timestamp: this LeRobot revision creates frame_index / FPS itself.
            }
        )
        assert self.raw_log is not None
        self.raw_log.write(
            json.dumps(
                {
                    "frame_index": self.frame_count,
                    "target_timestamp_ns": observation.target_timestamp_ns,
                    "base_skew_ns": observation.base_skew_ns,
                    "wrist_skew_ns": observation.wrist_skew_ns,
                    "arm_skew_ns": observation.arm_skew_ns,
                    "arm_raw": observation.arm_raw,
                }
            )
            + "\n"
        )
        self.frame_count += 1

    def finish(self) -> None:
        if self.raw_log is not None:
            self.raw_log.close()
            self.raw_log = None
        if self.dataset is not None and self.frame_count:
            self.dataset.save_episode()


def record_aligned_episode(
    scheduler: AlignedScheduler,
    writer: LeRobotAlignedEpisodeWriter,
    stop_event: threading.Event,
    duration_s: float,
) -> int:
    """Consume synchronized observations and save one directly usable LeRobot episode."""
    previous: AlignedObservation | None = None
    started_ns: int | None = None
    try:
        for current in scheduler.observations(stop_event):
            if started_ns is None:
                started_ns = current.target_timestamp_ns
            if previous is not None:
                writer.append(previous, current.arm_state)  # next_observation_proxy
            previous = current
            if current.target_timestamp_ns - started_ns >= round(duration_s * 1_000_000_000):
                break
        if previous is not None:
            writer.append(previous, previous.arm_state)
        writer.finish()
        return writer.frame_count
    finally:
        stop_event.set()
