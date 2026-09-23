#!/usr/bin/env python3
"""Run ten compressed dual-view MuJoCo diagnostic episodes directly on a USB drive."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fragilegrab.benchmark.runner import run_episode, save_json


def usb_mount(*, label: str | None, uuid: str | None, explicit_mount: Path | None) -> Path:
    if explicit_mount is not None:
        mount = explicit_mount.expanduser().resolve()
        if not mount.is_dir():
            raise RuntimeError(f"explicit USB mount does not exist: {mount}")
        return mount
    result = subprocess.run(
        ["lsblk", "--json", "--output", "LABEL,UUID,MOUNTPOINTS"],
        check=True,
        capture_output=True,
        text=True,
    )
    matches: list[Path] = []

    def visit(device: dict[str, Any]) -> None:
        if ((label is not None and device.get("label") == label)
                or (uuid is not None and device.get("uuid") == uuid)):
            matches.extend(Path(value) for value in device.get("mountpoints") or [] if value)
        for child in device.get("children") or []:
            visit(child)

    for device in json.loads(result.stdout).get("blockdevices", []):
        visit(device)
    unique = sorted({path.resolve() for path in matches})
    if len(unique) != 1:
        identity = f"label={label!r}, uuid={uuid!r}"
        raise RuntimeError(f"USB {identity} must resolve to exactly one mounted path, got {unique}")
    mount = unique[0]
    return mount


def verify_writable(mount: Path) -> None:
    probe = mount / f".fragilegrab-write-test-{os.getpid()}"
    try:
        probe.write_text("ok\n", encoding="utf-8")
    finally:
        probe.unlink(missing_ok=True)


def video_info(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    try:
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frames <= 0 or fps <= 0 or width <= 0 or height <= 0:
            raise RuntimeError(f"video is not decodable: {path}")
        return {"frames": frames, "fps": fps, "width": width, "height": height}
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    usb = parser.add_mutually_exclusive_group()
    usb.add_argument("--usb-uuid", default=None)
    usb.add_argument("--usb-label", default=None)
    usb.add_argument("--usb-mount", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--first-seed", type=int, default=7)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--task", choices=["qualification", "F1"], default="qualification")
    parser.add_argument(
        "--object-config",
        type=Path,
        default=ROOT / "configs/benchmark/object.libero_spatial_bowl_to_plate.yaml",
    )
    args = parser.parse_args()
    if args.usb_uuid is None and args.usb_label is None and args.usb_mount is None:
        args.usb_uuid = "D850-5AAE"
    if not 1 <= args.episodes <= 100:
        raise ValueError("--episodes must be in [1, 100]")

    mount = usb_mount(label=args.usb_label, uuid=args.usb_uuid, explicit_mount=args.usb_mount)
    verify_writable(mount)
    if shutil.disk_usage(mount).free < 2 * 1024**3:
        raise RuntimeError("USB has less than 2 GiB free; refusing to start")
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"piper_bowl_diagnostic_pilot_{args.episodes}ep_{timestamp}"
    output = mount / "FragileGrabVLA" / "pilot_data" / run_name
    output.mkdir(parents=True, exist_ok=False)

    summary: dict[str, Any] = {
        "schema_version": "fragilegrab.diagnostic_pilot/0.1",
        "status": "running",
        "created_at_local": dt.datetime.now().astimezone().isoformat(),
        "purpose": "diagnostic_only_not_yet_approved_for_training_or_norm_stats",
        "usb": {"label": args.usb_label, "uuid": args.usb_uuid, "mountpoint": str(mount)},
        "output": str(output),
        "object_config": str(args.object_config),
        "task": args.task,
        "episodes_requested": args.episodes,
        "episodes": [],
    }
    save_json(output / "pilot_summary.json", summary)

    for index in range(args.episodes):
        seed = args.first_seed + index
        episode_dir = output / f"episode_{index:03d}_seed_{seed}"
        print(f"[{index + 1}/{args.episodes}] seed={seed} -> {episode_dir}", flush=True)
        try:
            manifest = run_episode(
                ROOT / "configs/benchmark",
                ROOT / "assets/vendor/agx_arm_sim",
                episode_dir,
                args.task,
                seed,
                object_config=args.object_config,
                recording_options={
                    "save_raw_rgb_frames": False,
                    "compress_timeline": True,
                    "record_full_physics_timeline": False,
                    "generate_preview_video": False,
                },
            )
            videos = {
                role: video_info(episode_dir / filename)
                for role, filename in (("front", "base_rgb.mp4"), ("wrist", "wrist_rgb.mp4"))
            }
            item = {
                "index": index,
                "seed": seed,
                "directory": episode_dir.name,
                "status": "complete",
                "task_success": bool(manifest["result"]["task_success"]),
                "qualification": manifest["result"]["qualification"],
                "videos": videos,
                "size_bytes": sum(p.stat().st_size for p in episode_dir.rglob("*") if p.is_file()),
            }
        except Exception as exc:
            item = {
                "index": index,
                "seed": seed,
                "directory": episode_dir.name,
                "status": "failed_to_record",
                "error": f"{type(exc).__name__}: {exc}",
            }
        summary["episodes"].append(item)
        save_json(output / "pilot_summary.json", summary)

    complete = [item for item in summary["episodes"] if item["status"] == "complete"]
    successful = [item for item in complete if item.get("task_success")]
    summary.update(
        {
            "status": "complete",
            "completed_at_local": dt.datetime.now().astimezone().isoformat(),
            "episodes_recorded": len(complete),
            "task_successes": len(successful),
            "training_admission": "rejected_pending_review",
            "next_step": "Review both videos and scene/action timing; only validated successful episodes may enter a LeRobot train split.",
        }
    )
    save_json(output / "pilot_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if len(complete) != args.episodes:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
