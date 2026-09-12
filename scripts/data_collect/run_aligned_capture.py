"""One-command, read-only Piper X + D435i + DaBai DC1 LeRobot recorder.

No arm enable, motion, gripper command, reset, or calibration is performed.
"""

from __future__ import annotations

import argparse
import queue
import threading
from pathlib import Path

from aligned_capture import (
    LeRobotAlignedEpisodeWriter,
    OrbbecDaBaiDC1Adapter,
    PiperCanAdapter,
    ProducerFailure,
    RealSenseD435iAdapter,
    record_aligned_episode,
    start_sources,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only, time-aligned Piper X LeRobot recorder")
    parser.add_argument("--root", type=Path, required=True, help="New output directory for one LeRobot dataset")
    parser.add_argument("--task", required=True, help="Natural-language task label for this episode")
    parser.add_argument("--repo-id", default="local/piper_x_aligned")
    parser.add_argument("--seconds", type=float, default=30.0, help="Duration after first aligned observation")
    parser.add_argument("--d435i-serial", default=None, help="Optional D435i serial number")
    parser.add_argument("--dc1-serial", default=None, help="Optional DaBai DC1 serial number")
    parser.add_argument("--images", action="store_true", help="Store image frames rather than encoded videos")
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    args.root = args.root.resolve()
    if args.root.exists():
        parser.error(f"--root must not already exist: {args.root}")
    return args


def drain_failures(failures: queue.Queue[ProducerFailure]) -> list[ProducerFailure]:
    result: list[ProducerFailure] = []
    while True:
        try:
            result.append(failures.get_nowait())
        except queue.Empty:
            return result


def main() -> None:
    args = parse_args()
    stop_event = threading.Event()
    base_camera = RealSenseD435iAdapter(serial_number=args.d435i_serial)
    wrist_camera = OrbbecDaBaiDC1Adapter(serial_number=args.dc1_serial)
    piper = PiperCanAdapter()
    writer = LeRobotAlignedEpisodeWriter(
        repo_id=args.repo_id,
        root=args.root,
        task=args.task,
        use_videos=not args.images,
    )
    scheduler, threads, failures = start_sources(base_camera, wrist_camera, piper, stop_event)
    capture_error: BaseException | None = None
    try:
        print("Starting D435i, DaBai DC1, and read-only Piper CAN streams...")
        print("Waiting for all streams, then T0 = common_start + 100 ms. No robot commands are sent.")
        frames = record_aligned_episode(scheduler, writer, stop_event, args.seconds)
        print(f"Saved {frames} aligned frames to {args.root}")
    except KeyboardInterrupt:
        print("Capture cancelled by user.")
    except BaseException as exc:
        capture_error = exc
    finally:
        stop_event.set()
        for thread in threads:
            thread.join(timeout=2.0)

    producer_failures = drain_failures(failures)
    if producer_failures:
        detail = "\n".join(f"- {failure.source}: {failure.error}" for failure in producer_failures)
        raise RuntimeError(f"A producer failed before or during capture:\n{detail}")
    if capture_error is not None:
        raise capture_error


if __name__ == "__main__":
    main()
