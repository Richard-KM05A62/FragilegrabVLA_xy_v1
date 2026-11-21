"""Read-only live source reuse and explicitly synthetic demo sources."""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .rtc_core import DataBuffer, NullCommandSink, RuntimeConfig, SECOND_NS, TimeGrid


class SyntheticBackend:
    """Timing demo only. This is NOT a model or an RTC velocity-field sampler."""
    def __init__(self, config: RuntimeConfig, latency_ms: float) -> None:
        self.config, self.latency_ms = config, latency_ms
        self._closed = threading.Event()

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._closed.wait(self.latency_ms / 1000):
            raise RuntimeError("Synthetic backend closed")
        state = np.asarray(payload["observation/state"], dtype=np.float32)
        actions = np.repeat(state[None], self.config.horizon, axis=0)
        actions[:, 0] += np.arange(self.config.horizon, dtype=np.float32) * 0.001
        weights = payload["rtc/reference_weights"]
        reference = payload["rtc/reference_actions"]
        actions = weights * reference + (1 - weights) * actions
        return {"actions": actions, "policy_timing": {"infer_ms": self.latency_ms}}

    def close(self) -> None:
        self._closed.set()


class Sources:
    def __init__(self, data: DataBuffer, stop: threading.Event) -> None:
        self.data, self.stop = data, stop
        self.errors: queue.Queue[tuple[str, BaseException]] = queue.Queue(maxsize=3)
        self.threads: list[threading.Thread] = []

    def _launch(self, source: str, function) -> None:
        def guarded() -> None:
            try:
                function()
            except BaseException as exc:
                if not self.stop.is_set():
                    self.errors.put_nowait((source, exc))
                    self.stop.set()
        thread = threading.Thread(target=guarded, name=f"RTCSource-{source}", daemon=True)
        self.threads.append(thread)
        thread.start()

    def start_demo(self, sink: NullCommandSink) -> None:
        def produce() -> None:
            grid = TimeGrid(time.monotonic_ns())
            rgb = np.zeros((48, 64, 3), dtype=np.uint8)
            state = np.zeros(7, dtype=np.float32)
            step = 0
            while not self.stop.is_set():
                if sink.targets:
                    joints, width = sink.targets[-1]
                    state[:6] = joints
                    state[6] = np.clip(1 - width / self.data.config.open_width_m, 0, 1)
                now = time.monotonic_ns()
                self.data.append_camera("base", rgb, now)
                self.data.append_camera("wrist", rgb, now)
                self.data.append_state(state, now)
                step += 1
                self.stop.wait(max(0, (grid.at(step) - time.monotonic_ns()) / SECOND_NS))
        self._launch("synthetic", produce)

    def start_live(self, base_serial: str | None, wrist_serial: str | None,
                   orbbec_sdk_root: Path | None = None) -> None:
        # Import only in explicit live mode. No robot control methods are called.
        from capture.aligned_capture import OrbbecDaBaiDC1Adapter, PiperCanAdapter, RealSenseD435iAdapter
        base = RealSenseD435iAdapter(base_serial)
        wrist_options = {"sdk_root": orbbec_sdk_root} if orbbec_sdk_root is not None else {}
        wrist = OrbbecDaBaiDC1Adapter(wrist_serial, **wrist_options)
        arm = PiperCanAdapter(gripper_open_width_m=self.data.config.open_width_m)

        def camera(source: str, adapter) -> None:
            try:
                adapter.start()
                while not self.stop.is_set():
                    frame = adapter.read_rgb(stop_event=self.stop)
                    self.data.append_camera(source, frame.value, frame.timestamp_ns)
            except InterruptedError:
                if not self.stop.is_set():
                    raise
            finally:
                adapter.close()

        def can() -> None:
            try:
                arm.start()
                while not self.stop.is_set():
                    value = arm.read_state()
                    self.data.append_state(value.value.state, value.timestamp_ns)
                    self.stop.wait(0.01)
            finally:
                arm.close()

        self._launch("base", lambda: camera("base", base))
        self._launch("wrist", lambda: camera("wrist", wrist))
        self._launch("CAN", can)

    def failure(self) -> tuple[str, BaseException] | None:
        try:
            return self.errors.get_nowait()
        except queue.Empty:
            return None

    def join(self) -> list[str]:
        for thread in self.threads:
            thread.join(timeout=2)
        return [thread.name for thread in self.threads if thread.is_alive()]
