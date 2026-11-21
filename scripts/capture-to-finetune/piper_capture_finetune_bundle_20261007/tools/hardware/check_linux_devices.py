"""Read-only Linux probe for one Piper input device at a time.

This script never enables or commands the robot. Run one probe only after its
vendor SDK and udev permissions have been installed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from capture.device_adapters_linux import OrbbecDaBaiDC1AdapterLinux
from capture.device_adapters_linux import PiperCanAdapterLinux
from capture.device_adapters_linux import RealSenseD435iAdapterLinux


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("device", choices=("d435i", "dc1", "can"))
    parser.add_argument("--serial")
    parser.add_argument("--orbbec-sdk-root", type=Path)
    parser.add_argument("--can-interface")
    parser.add_argument("--can-channel")
    parser.add_argument("--frames", type=int, default=30)
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")

    if args.device == "d435i":
        adapter = RealSenseD435iAdapterLinux(args.serial)
    elif args.device == "dc1":
        adapter = OrbbecDaBaiDC1AdapterLinux(args.serial, sdk_root=args.orbbec_sdk_root)
    else:
        adapter = PiperCanAdapterLinux(channel=args.can_channel, interface=args.can_interface)

    print("READ-ONLY probe: no enable or motion calls.", flush=True)
    started = time.monotonic()
    try:
        metadata = adapter.start()
        if metadata is not None:
            print("metadata:", metadata, flush=True)
        previous_ns = None
        for index in range(args.frames):
            value = adapter.read_rgb() if args.device != "can" else adapter.read_state()
            interval_ms = None if previous_ns is None else (value.timestamp_ns - previous_ns) / 1e6
            previous_ns = value.timestamp_ns
            if args.device == "can":
                payload = value.value
                print(index, "state=", payload.state.tolist(), "interval_ms=", interval_ms, flush=True)
            else:
                image = value.value
                print(index, "shape=", image.shape, "dtype=", image.dtype,
                      "interval_ms=", interval_ms, flush=True)
    finally:
        adapter.close()
    print("READ_ONLY_DEVICE_PROBE_OK elapsed_s=", round(time.monotonic() - started, 3), flush=True)


if __name__ == "__main__":
    main()

