"""Linux read-only device sources for the RTC harness.

The original ``rtc_sources.py`` remains unchanged.  Synthetic behavior is
inherited; only live device construction is replaced.
"""

from __future__ import annotations

import threading
from pathlib import Path

from capture.device_adapters_linux import OrbbecDaBaiDC1AdapterLinux
from capture.device_adapters_linux import PiperCanAdapterLinux
from capture.device_adapters_linux import RealSenseD435iAdapterLinux

from .rtc_sources import Sources as _Sources


class SourcesLinux(_Sources):
    def start_live(
        self,
        base_serial: str | None,
        wrist_serial: str | None,
        orbbec_sdk_root: Path | None = None,
    ) -> None:
        base = RealSenseD435iAdapterLinux(base_serial)
        wrist = OrbbecDaBaiDC1AdapterLinux(wrist_serial, sdk_root=orbbec_sdk_root)
        arm = PiperCanAdapterLinux(gripper_open_width_m=self.data.config.open_width_m)

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


# The Linux launcher patches this name into run_piper_rtc without editing the
# existing cross-platform/demo entry point.
Sources = SourcesLinux

