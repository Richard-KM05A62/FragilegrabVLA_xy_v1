"""Linux-only, read-only adapters for Piper X input devices.

This file intentionally lives beside ``aligned_capture.py`` and does not alter
the Windows implementation.  It contains no arm enable or motion calls.
"""

from __future__ import annotations

import ctypes
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np


DEFAULT_GRIPPER_OPEN_WIDTH_M = 0.1001
T = TypeVar("T")


@dataclass(frozen=True)
class TimedValue(Generic[T]):
    value: T
    timestamp_ns: int


@dataclass(frozen=True)
class ArmState:
    state: np.ndarray
    raw: dict[str, Any]


def gripper_closedness(width_m: float, open_width_m: float) -> float:
    return float(1.0 - np.clip(width_m / open_width_m, 0.0, 1.0))


class CameraAdapter(ABC):
    camera_name: str

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def read_rgb(self, stop_event: threading.Event | None = None) -> TimedValue[np.ndarray]: ...

    @abstractmethod
    def close(self) -> None: ...


class RealSenseD435iAdapterLinux(CameraAdapter):
    """D435i RGB source through the Linux librealsense Python binding."""

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
                "D435i requires the Linux librealsense Python binding in the device harness environment"
            ) from exc
        self._rs = rs
        self._pipeline = rs.pipeline()
        config = rs.config()
        if self.serial_number:
            config.enable_device(self.serial_number)
        try:
            config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
            self._pipeline.start(config)
            self._output_format = rs.format.rgb8
        except RuntimeError:
            self._pipeline = rs.pipeline()
            config = rs.config()
            if self.serial_number:
                config.enable_device(self.serial_number)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self._pipeline.start(config)
            self._output_format = rs.format.bgr8
        self._started = True

    def read_rgb(self, stop_event: threading.Event | None = None) -> TimedValue[np.ndarray]:
        if not self._started or self._pipeline is None or self._rs is None:
            raise RuntimeError("D435i adapter has not been started")
        while stop_event is None or not stop_event.is_set():
            frames: Any | None = None
            color_frame: Any | None = None
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1_000)
                arrival_ns = time.monotonic_ns()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                image = np.asanyarray(color_frame.get_data())
                if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                    raise RuntimeError(f"D435i returned invalid RGB data: {image.shape} {image.dtype}")
                if self._output_format == self._rs.format.bgr8:
                    image = image[:, :, ::-1]
                return TimedValue(np.array(image, dtype=np.uint8, copy=True, order="C"), arrival_ns)
            finally:
                del color_frame
                del frames
        raise InterruptedError("D435i acquisition stopped")

    def close(self) -> None:
        if self._pipeline is not None and self._started:
            try:
                self._pipeline.stop()
            finally:
                self._started = False


def find_orbbec_library(sdk_root: Path | str | None = None) -> Path:
    """Resolve libOrbbecSDK.so without relying on a Windows SDK layout."""

    root_value = sdk_root or os.environ.get("ORBBEC_SDK_ROOT")
    if not root_value:
        raise RuntimeError("Set ORBBEC_SDK_ROOT or pass --orbbec-sdk-root for the Linux Orbbec SDK")
    root = Path(root_value).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Orbbec Linux SDK root not found: {root}")
    candidates = (
        root / "libOrbbecSDK.so",
        root / "lib" / "libOrbbecSDK.so",
        root / "SDK" / "lib" / "libOrbbecSDK.so",
        root / "lib" / "linux_x64" / "libOrbbecSDK.so",
        root / "lib" / "x64" / "libOrbbecSDK.so",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    recursive = sorted(root.rglob("libOrbbecSDK.so"))
    if len(recursive) == 1:
        return recursive[0]
    if recursive:
        raise RuntimeError(f"Multiple libOrbbecSDK.so files under {root}; pass their exact parent as SDK root")
    raise FileNotFoundError(f"libOrbbecSDK.so not found under Orbbec SDK root: {root}")


class OrbbecDaBaiDC1AdapterLinux(CameraAdapter):
    """DaBai DC1 RGB source through the Orbbec Linux C SDK."""

    camera_name = "orbbec_dabai_dc1"
    _OB_SENSOR_COLOR = 2
    _OB_FORMAT_RGB = 22
    _OB_WIDTH_ANY = 0
    _OB_HEIGHT_ANY = 0
    _OB_FPS_ANY = 0
    _OB_LOG_SEVERITY_OFF = 5

    def __init__(self, serial_number: str | None = None, sdk_root: Path | str | None = None) -> None:
        self.serial_number = serial_number
        self.sdk_root = sdk_root
        self._lib: ctypes.CDLL | None = None
        self._context: ctypes.c_void_p | None = None
        self._device: ctypes.c_void_p | None = None
        self._pipeline: ctypes.c_void_p | None = None
        self._config: ctypes.c_void_p | None = None
        self._profiles: ctypes.c_void_p | None = None
        self._color_profile: ctypes.c_void_p | None = None

    def _bind(self, name: str, restype: Any, *argtypes: Any) -> None:
        assert self._lib is not None
        fn = getattr(self._lib, name)
        fn.restype = restype
        fn.argtypes = list(argtypes)

    def _load_sdk(self) -> None:
        if self._lib is not None:
            return
        path = find_orbbec_library(self.sdk_root)
        try:
            self._lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            raise RuntimeError(
                f"Failed to load {path}; install its dependencies or add {path.parent} to LD_LIBRARY_PATH"
            ) from exc
        void_p = ctypes.c_void_p
        err_pp = ctypes.POINTER(void_p)
        uint32 = ctypes.c_uint32
        bindings = (
            ("ob_error_message", ctypes.c_char_p, (void_p,)),
            ("ob_error_function", ctypes.c_char_p, (void_p,)),
            ("ob_delete_error", None, (void_p,)),
            ("ob_set_logger_severity", None, (ctypes.c_int, err_pp)),
            ("ob_create_context", void_p, (err_pp,)),
            ("ob_delete_context", None, (void_p, err_pp)),
            ("ob_query_device_list", void_p, (void_p, err_pp)),
            ("ob_device_list_device_count", uint32, (void_p, err_pp)),
            ("ob_device_list_get_device_serial_number", ctypes.c_char_p, (void_p, uint32, err_pp)),
            ("ob_delete_device_list", None, (void_p, err_pp)),
            ("ob_device_list_get_device_by_serial_number", void_p, (void_p, ctypes.c_char_p, err_pp)),
            ("ob_delete_device", None, (void_p, err_pp)),
            ("ob_create_pipeline", void_p, (err_pp,)),
            ("ob_create_pipeline_with_device", void_p, (void_p, err_pp)),
            ("ob_delete_pipeline", None, (void_p, err_pp)),
            ("ob_pipeline_get_stream_profile_list", void_p, (void_p, ctypes.c_int, err_pp)),
            ("ob_pipeline_start_with_config", None, (void_p, void_p, err_pp)),
            ("ob_pipeline_stop", None, (void_p, err_pp)),
            ("ob_pipeline_wait_for_frameset", void_p, (void_p, uint32, err_pp)),
            ("ob_create_config", void_p, (err_pp,)),
            ("ob_delete_config", None, (void_p, err_pp)),
            ("ob_config_enable_stream", None, (void_p, void_p, err_pp)),
            ("ob_stream_profile_list_get_video_stream_profile", void_p,
             (void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, err_pp)),
            ("ob_delete_stream_profile_list", None, (void_p, err_pp)),
            ("ob_delete_stream_profile", None, (void_p, err_pp)),
            ("ob_frameset_color_frame", void_p, (void_p, err_pp)),
            ("ob_delete_frame", None, (void_p, err_pp)),
            ("ob_frame_data", void_p, (void_p, err_pp)),
            ("ob_frame_data_size", uint32, (void_p, err_pp)),
            ("ob_video_frame_width", uint32, (void_p, err_pp)),
            ("ob_video_frame_height", uint32, (void_p, err_pp)),
        )
        for name, restype, argtypes in bindings:
            self._bind(name, restype, *argtypes)
        self._call("ob_set_logger_severity", self._OB_LOG_SEVERITY_OFF)

    def _raise_if_error(self, error: ctypes.c_void_p, api_name: str) -> None:
        if not error.value:
            return
        assert self._lib is not None
        try:
            function = self._lib.ob_error_function(error)
            message = self._lib.ob_error_message(error)
            fn = function.decode(errors="replace") if function else api_name
            msg = message.decode(errors="replace") if message else "unknown SDK error"
        finally:
            self._lib.ob_delete_error(error)
        raise RuntimeError(f"Orbbec {fn}: {msg}")

    def _call(self, name: str, *args: Any) -> Any:
        assert self._lib is not None
        error = ctypes.c_void_p()
        result = getattr(self._lib, name)(*args, ctypes.byref(error))
        self._raise_if_error(error, name)
        return result

    def _try_color_profile(self, width: int, height: int, fps: int) -> ctypes.c_void_p | None:
        assert self._lib is not None and self._profiles is not None
        error = ctypes.c_void_p()
        profile = self._lib.ob_stream_profile_list_get_video_stream_profile(
            self._profiles, width, height, self._OB_FORMAT_RGB, fps, ctypes.byref(error)
        )
        if error.value:
            self._lib.ob_delete_error(error)
            return None
        return profile or None

    def start(self) -> None:
        self._load_sdk()
        if self.serial_number:
            self._context = self._call("ob_create_context")
            devices = self._call("ob_query_device_list", self._context)
            try:
                count = int(self._call("ob_device_list_device_count", devices))
                sdk_serials = [
                    (self._call("ob_device_list_get_device_serial_number", devices, index) or b"").decode(
                        errors="replace"
                    )
                    for index in range(count)
                ]
                if self.serial_number not in sdk_serials:
                    shown = [value or "<empty>" for value in sdk_serials]
                    raise RuntimeError(
                        f"DaBai DC1 serial {self.serial_number!r} is not exposed by the Orbbec SDK; "
                        f"SDK serials={shown}. Omit the serial when only one Orbbec camera is connected."
                    )
                self._device = self._call(
                    "ob_device_list_get_device_by_serial_number", devices, self.serial_number.encode()
                )
                self._pipeline = self._call("ob_create_pipeline_with_device", self._device)
            finally:
                self._call("ob_delete_device_list", devices)
        else:
            self._pipeline = self._call("ob_create_pipeline")
        self._config = self._call("ob_create_config")
        self._profiles = self._call("ob_pipeline_get_stream_profile_list", self._pipeline, self._OB_SENSOR_COLOR)
        self._color_profile = (
            self._try_color_profile(640, 480, 30)
            or self._try_color_profile(self._OB_WIDTH_ANY, self._OB_HEIGHT_ANY, 30)
            or self._try_color_profile(self._OB_WIDTH_ANY, self._OB_HEIGHT_ANY, self._OB_FPS_ANY)
        )
        if self._color_profile is None:
            raise RuntimeError("DaBai DC1 exposes no RGB color-stream profile")
        self._call("ob_config_enable_stream", self._config, self._color_profile)
        self._call("ob_pipeline_start_with_config", self._pipeline, self._config)

    def read_rgb(self, stop_event: threading.Event | None = None) -> TimedValue[np.ndarray]:
        if self._pipeline is None:
            raise RuntimeError("DaBai DC1 adapter has not been started")
        while stop_event is None or not stop_event.is_set():
            frameset = self._call("ob_pipeline_wait_for_frameset", self._pipeline, 100)
            if not frameset:
                continue
            arrival_ns = time.monotonic_ns()
            color: ctypes.c_void_p | None = None
            try:
                color = self._call("ob_frameset_color_frame", frameset)
                if not color:
                    continue
                width = int(self._call("ob_video_frame_width", color))
                height = int(self._call("ob_video_frame_height", color))
                size = int(self._call("ob_frame_data_size", color))
                if size != width * height * 3:
                    raise RuntimeError(f"DC1 RGB buffer size {size} does not match {width}x{height}x3")
                data = self._call("ob_frame_data", color)
                if not data:
                    raise RuntimeError("DaBai DC1 returned a null RGB buffer")
                rgb = np.frombuffer(ctypes.string_at(data, size), dtype=np.uint8).reshape(height, width, 3)
                return TimedValue(rgb.copy(), arrival_ns)
            finally:
                try:
                    if color:
                        self._call("ob_delete_frame", color)
                finally:
                    self._call("ob_delete_frame", frameset)
        raise InterruptedError("DC1 acquisition stopped")

    def close(self) -> None:
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
            if value is not None and self._lib is not None:
                try:
                    self._call(name, value)
                except Exception:
                    pass
                setattr(self, attribute, None)


class PiperCanAdapterLinux:
    """Configurable Linux Piper CAN reader; never enables or commands the arm."""

    def __init__(
        self,
        gripper_open_width_m: float = DEFAULT_GRIPPER_OPEN_WIDTH_M,
        *,
        channel: str | None = None,
        interface: str | None = None,
    ) -> None:
        try:
            from pyAgxArm import AgxArmFactory, create_agx_arm_config
        except ImportError as exc:
            raise RuntimeError("Install the Piper/pyAgxArm Linux SDK in the device harness environment") from exc
        self.channel = channel or os.environ.get("PIPER_CAN_CHANNEL", "can0")
        self.interface = interface or os.environ.get("PIPER_CAN_INTERFACE", "socketcan")
        self._gripper_open_width_m = gripper_open_width_m
        cfg = create_agx_arm_config(
            robot="piper_x", comm="can", channel=self.channel, interface=self.interface
        )
        self._robot = AgxArmFactory.create_arm(cfg)
        self._effector: Any | None = None

    def start(self) -> dict[str, Any]:
        self._robot.connect()
        if not self._robot.is_connected() or not self._robot.is_ok():
            raise RuntimeError(
                f"Piper CAN connection is unhealthy (channel={self.channel!r}, interface={self.interface!r})"
            )
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
            raise RuntimeError("Missing gripper feedback")
        width_m = float(gripper.msg.value)
        state = np.asarray(
            [*joints.msg, gripper_closedness(width_m, self._gripper_open_width_m)], dtype=np.float32
        )
        raw = {
            "joint_sdk_timestamp_s": joints.timestamp,
            "gripper_sdk_timestamp_s": gripper.timestamp,
            "gripper_width_m": width_m,
            "gripper_force_n": float(gripper.msg.force),
            "gripper_mode": gripper.msg.mode,
            "can_channel": self.channel,
            "can_interface": self.interface,
        }
        return TimedValue(ArmState(state=state, raw=raw), time.monotonic_ns())

    def close(self) -> None:
        self._robot.disconnect()
