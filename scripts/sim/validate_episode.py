#!/usr/bin/env python3
"""Validate structural integrity of a FragileGrabVLA simulation episode v0.1."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


EPISODE_SCHEMA = "fragilegrab.episode/0.1"
REQUIRED_STREAMS = {
    "observation.image.base_rgb",
    "observation.image.wrist_rgb",
    "robot_state",
    "object_state",
    "contact",
    "joint_effort",
}


class EpisodeError(ValueError):
    """The episode is missing required evidence or contains invalid references."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise EpisodeError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EpisodeError(f"JSON root must be a mapping: {path}")
    return value


def _existing_file(episode_dir: Path, relative_ref: Any, label: str) -> Path:
    if not isinstance(relative_ref, str) or not relative_ref:
        raise EpisodeError(f"{label} must be a non-empty relative path")
    path = Path(relative_ref)
    if path.is_absolute() or ".." in path.parts:
        raise EpisodeError(f"{label} must stay inside the episode directory")
    resolved = episode_dir / path
    if not resolved.is_file():
        raise EpisodeError(f"missing {label}: {relative_ref}")
    return resolved


def validate_episode(episode_dir: Path) -> dict[str, Any]:
    episode_dir = episode_dir.resolve()
    if (episode_dir / ".incomplete").exists():
        raise EpisodeError("episode still contains .incomplete")

    manifest = _load_json(episode_dir / "episode_manifest.json")
    if manifest.get("episode_schema_version") != EPISODE_SCHEMA:
        raise EpisodeError(f"episode_schema_version must be {EPISODE_SCHEMA}")
    if manifest.get("source", {}).get("kind") != "simulation":
        raise EpisodeError("source.kind must be simulation")
    if manifest.get("action", {}).get("source") != "scripted_joint_position_targets":
        raise EpisodeError("this validator only accepts the policy-free scripted baseline")
    if manifest.get("termination", {}).get("status") != "complete":
        raise EpisodeError("only complete episodes pass baseline validation")

    config_record = manifest.get("experiment_config")
    if not isinstance(config_record, dict):
        raise EpisodeError("experiment_config descriptor is missing")
    config_path = _existing_file(
        episode_dir, config_record.get("storage_ref"), "experiment config snapshot"
    )
    if _sha256(config_path) != config_record.get("sha256"):
        raise EpisodeError("experiment config SHA-256 does not match manifest")

    streams = manifest.get("streams")
    if not isinstance(streams, dict):
        raise EpisodeError("streams must be a mapping")
    missing_streams = REQUIRED_STREAMS - set(streams)
    if missing_streams:
        raise EpisodeError(f"missing required streams: {sorted(missing_streams)}")

    timeline_ref = manifest.get("artifacts", {}).get("timeline")
    timeline_path = _existing_file(episode_dir, timeline_ref, "timeline")
    observations = 0
    executions = 0
    previous_episode_step = -1
    previous_sim_time = float("-inf")
    first_image_shapes: dict[str, list[int]] = {}

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on validation host
        raise EpisodeError("NumPy is required to validate recorded RGB arrays") from exc

    with timeline_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EpisodeError(f"invalid timeline JSON at line {line_number}: {exc}") from exc
            record_type = record.get("record_type")
            if record_type == "execution":
                executions += 1
                requested = record.get("requested_action")
                executed = record.get("executed_action")
                if requested != executed:
                    raise EpisodeError(
                        f"execution line {line_number} changed requested action without recording a rule"
                    )
                if record.get("dispatch_status") != "submitted_to_articulation_position_targets":
                    raise EpisodeError(f"execution line {line_number} has invalid dispatch status")
                continue
            if record_type != "observation":
                raise EpisodeError(f"unknown timeline record type at line {line_number}")

            observations += 1
            episode_step = record.get("episode_physics_step")
            sim_time = record.get("simulation_time_s")
            if not isinstance(episode_step, int) or episode_step < previous_episode_step:
                raise EpisodeError(f"non-monotonic episode physics step at line {line_number}")
            if not isinstance(sim_time, (int, float)) or sim_time < previous_sim_time:
                raise EpisodeError(f"non-monotonic simulation time at line {line_number}")
            previous_episode_step = episode_step
            previous_sim_time = float(sim_time)

            robot_state = record.get("robot_state")
            if not isinstance(robot_state, dict) or robot_state.get("is_valid") is not True:
                raise EpisodeError(f"invalid robot_state at line {line_number}")
            for field in ("dof_names", "positions", "velocities", "efforts", "dof_types"):
                if field not in robot_state:
                    raise EpisodeError(f"robot_state.{field} missing at line {line_number}")
            dof_count = len(robot_state["dof_names"])
            if any(
                len(robot_state[field]) != dof_count
                for field in ("positions", "velocities", "efforts", "dof_types")
            ):
                raise EpisodeError(f"robot_state vector length mismatch at line {line_number}")

            object_state = record.get("object_state")
            if not isinstance(object_state, dict):
                raise EpisodeError(f"object_state missing at line {line_number}")
            contact = record.get("contact")
            if not isinstance(contact, dict) or contact.get("is_valid") is not True:
                raise EpisodeError(f"contact sensor is invalid at line {line_number}")

            images = record.get("images")
            if not isinstance(images, dict) or set(images) != {"base_rgb", "wrist_rgb"}:
                raise EpisodeError(f"two camera roles are required at line {line_number}")
            for role, image_ref in images.items():
                image_path = _existing_file(
                    episode_dir, image_ref, f"{role} image at line {line_number}"
                )
                frame = np.load(image_path, allow_pickle=False)
                if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] not in (3, 4):
                    raise EpisodeError(
                        f"{role} image at line {line_number} must be HWC uint8 RGB/RGBA"
                    )
                declared = streams[f"observation.image.{role}"].get("shape")
                if list(frame.shape) != declared:
                    raise EpisodeError(
                        f"{role} image shape at line {line_number} differs from stream descriptor"
                    )
                first_image_shapes.setdefault(role, list(frame.shape))

    integrity = manifest.get("integrity", {})
    if observations == 0:
        raise EpisodeError("timeline contains no observations")
    if executions == 0:
        raise EpisodeError("timeline contains no execution records")
    if integrity.get("observation_count") != observations:
        raise EpisodeError("observation count does not match manifest")
    if integrity.get("execution_count") != executions:
        raise EpisodeError("execution count does not match manifest")

    videos = manifest.get("artifacts", {}).get("videos")
    if not isinstance(videos, dict) or set(videos) != {"base_rgb", "wrist_rgb"}:
        raise EpisodeError("both camera video references are required")
    for role, video_ref in videos.items():
        video_path = _existing_file(episode_dir, video_ref, f"{role} video")
        if video_path.stat().st_size == 0:
            raise EpisodeError(f"{role} video is empty")

    return {
        "status": "ok",
        "episode_id": manifest.get("episode_id"),
        "observations": observations,
        "executions": executions,
        "image_shapes": first_image_shapes,
        "termination": manifest["termination"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_dir", type=Path)
    args = parser.parse_args()
    try:
        result = validate_episode(args.episode_dir)
    except EpisodeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
