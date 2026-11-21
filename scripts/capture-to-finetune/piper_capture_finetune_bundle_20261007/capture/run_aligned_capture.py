"""Interactive, read-only Piper X + D435i + DaBai DC1 LeRobot recorder.

No arm enable, motion, gripper command, reset, or calibration is performed.
"""

from __future__ import annotations

import argparse
import queue
import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .aligned_capture import ProducerFailure


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive, time-aligned Piper X LeRobot recorder")
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "data" / "piper_x_recordings",
        help="Recording directory; created at startup and reused on later runs",
    )
    parser.add_argument("--repo-id", default="local/piper_x_aligned")
    parser.add_argument("--d435i-serial", default=None, help="Optional D435i serial number")
    parser.add_argument("--dc1-serial", default=None, help="Optional DaBai DC1 serial number")
    parser.add_argument("--images", action="store_true", help="Store image frames rather than encoded videos")
    args = parser.parse_args()
    args.root = args.root.resolve()
    return args


def drain_failures(failures: queue.Queue[ProducerFailure]) -> list[ProducerFailure]:
    result: list[ProducerFailure] = []
    while True:
        try:
            result.append(failures.get_nowait())
        except queue.Empty:
            return result


def wait_for_enter_to_stop(episode_stop_event: threading.Event) -> None:
    """Block in a helper thread so aligned recording can continue until Enter is pressed."""
    try:
        input("Recording now. Press Enter to stop recording and begin saving... ")
    except EOFError:
        print("Standard input closed; saving the current trajectory.")
    episode_stop_event.set()


def main() -> None:
    args = parse_args()
    # Parse --help before importing camera/CAN and LeRobot dependencies.
    from .aligned_capture import (
        LeRobotAlignedEpisodeWriter,
        OrbbecDaBaiDC1Adapter,
        PiperCanAdapter,
        RealSenseD435iAdapter,
        record_aligned_episode,
        start_sources,
    )

    stop_event = threading.Event()
    base_camera = RealSenseD435iAdapter(serial_number=args.d435i_serial)
    wrist_camera = OrbbecDaBaiDC1Adapter(serial_number=args.dc1_serial)
    piper = PiperCanAdapter()
    writer = LeRobotAlignedEpisodeWriter(
        repo_id=args.repo_id,
        root=args.root,
        use_videos=not args.images,
    )
    # Make the requested recording destination visible immediately. The first
    # saved aligned frame will turn this empty directory into a LeRobot dataset.
    args.root.mkdir(parents=True, exist_ok=True)
    scheduler, threads, failures = start_sources(base_camera, wrist_camera, piper, stop_event)
    capture_error: BaseException | None = None
    try:
        print("Starting D435i, DaBai DC1, and read-only Piper CAN streams...")
        existing_episodes = writer.existing_episode_count()
        print(f"Dataset: {args.root}")
        print(f"Existing trajectories: {existing_episodes}; next saved trajectory will be #{existing_episodes:06d}.")
        print("No robot commands are sent. Each recording begins at Enter + 100 ms after three-stream alignment.")

        while not stop_event.is_set():
            command = input("Press Enter to start the next trajectory, or type q then Enter to quit: ").strip().lower()
            if command in {"q", "quit", "exit"}:
                break
            if command:
                print("Unrecognized command. Press Enter to record, or type q to quit.")
                continue

            while True:
                task = input(f"Task prompt for trajectory #{existing_episodes:06d}: ").strip()
                if task:
                    break
                print("A non-empty task prompt is required; this trajectory has not started.")

            episode_stop_event = threading.Event()
            input_thread = threading.Thread(
                target=wait_for_enter_to_stop, args=(episode_stop_event,), name="EpisodeEnterStop", daemon=True
            )
            input_thread.start()
            frames = record_aligned_episode(scheduler, writer, stop_event, episode_stop_event, task)
            input_thread.join()
            producer_failures = drain_failures(failures)
            if producer_failures:
                detail = "\n".join(f"- {failure.source}: {failure.error}" for failure in producer_failures)
                raise RuntimeError(f"A producer failed during capture:\n{detail}")
            if frames:
                completed_index = existing_episodes
                existing_episodes += 1
                print(f"Saved trajectory #{completed_index:06d}: {frames} aligned frames. Task: {task!r}")
            else:
                print("No aligned frame arrived before Enter; no trajectory was saved.")
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
