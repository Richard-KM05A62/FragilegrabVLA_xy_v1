"""Bounded OpenPI websocket transport, with mandatory deployment contract checks."""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any

from .rtc_core import RuntimeConfig


def stats_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_metadata(metadata: dict[str, Any], config: RuntimeConfig, expected_stats: str) -> None:
    expected = {"rtc_enabled": True, "horizon": config.horizon, "action_dim": 7,
                "fps": config.fps, "action_space": "absolute_joint_position",
                "joint_unit": "rad", "gripper": "closedness", "norm_stats_sha256": expected_stats}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"RTC server contract mismatch: {key}: expected {value!r}, got {metadata.get(key)!r}")


class WebsocketRTCBackend:
    """One request at a time. No implicit infinite connection retry or response wait."""
    def __init__(self, host: str, port: int, config: RuntimeConfig, norm_stats_path: Path,
                 *, connect_timeout: float = 5, response_timeout: float = 180) -> None:
        if connect_timeout <= 0 or response_timeout <= 0:
            raise ValueError("Transport timeouts must be positive")
        try:
            from openpi_client import msgpack_numpy
            from websockets.sync.client import connect
        except ImportError as exc:
            raise RuntimeError(
                "Missing RTC client dependencies. Install the local OpenPI packages/openpi-client "
                "package in D:/vscode/opencv/venv; see RTC_RUNTIME.md."
            ) from exc
        self._codec, self._connect_fn = msgpack_numpy, connect
        self._uri = f"ws://{host}:{port}"
        self.config = config
        self._expected_stats = stats_sha256(norm_stats_path)
        self.connect_timeout, self.response_timeout = connect_timeout, response_timeout
        self._closed = threading.Event()
        self._lock = threading.Lock()
        self._ws = None
        self._connecting_ws = None
        self.metadata: dict[str, Any] = {}
        self._connect()

    def _connect(self) -> Any:
        if self._closed.is_set():
            raise RuntimeError("RTC transport is closed")
        ws = self._connect_fn(self._uri, open_timeout=self.connect_timeout, close_timeout=1,
                              compression=None, max_size=32 * 1024 * 1024)
        try:
            with self._lock:
                if self._closed.is_set():
                    raise RuntimeError("RTC transport was closed during connection")
                self._connecting_ws = ws
            message = ws.recv(timeout=self.connect_timeout)
            if not isinstance(message, bytes):
                raise ValueError("RTC server did not send binary metadata")
            metadata = self._codec.unpackb(message)
            if not isinstance(metadata, dict):
                raise ValueError("RTC metadata must be a mapping")
            validate_metadata(metadata, self.config, self._expected_stats)
            with self._lock:
                if self._closed.is_set():
                    raise RuntimeError("RTC transport was closed during connection")
                self.metadata, self._ws = metadata, ws
                self._connecting_ws = None
            return ws
        except BaseException:
            with self._lock:
                if self._connecting_ws is ws:
                    self._connecting_ws = None
            ws.close()
            raise

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._closed.is_set():
            raise RuntimeError("RTC transport is closed")
        with self._lock:
            ws = self._ws
        if ws is None:
            ws = self._connect()  # One bounded attempt per controller retry.
        try:
            ws.send(self._codec.packb(payload))
            message = ws.recv(timeout=self.response_timeout)
            if isinstance(message, str):
                raise RuntimeError(f"RTC model server error: {message[-2000:]}")
            result = self._codec.unpackb(message)
            if not isinstance(result, dict):
                raise ValueError("RTC inference response must be a mapping")
            return result
        except BaseException:
            # A timed-out response cannot be reused as the next request's result.
            with self._lock:
                if self._ws is ws:
                    self._ws = None
            ws.close()
            raise

    def close(self) -> None:
        self._closed.set()
        with self._lock:
            sockets = (self._ws, self._connecting_ws)
            self._ws = self._connecting_ws = None
        for ws in sockets:
            # Also interrupts an outstanding recv() on the predictor worker.
            if ws is not None:
                ws.close()
