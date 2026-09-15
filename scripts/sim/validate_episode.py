#!/usr/bin/env python3
"""Validate structural integrity of a FragileGrabVLA simulation episode v0.1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def _stream_descriptor(streams: dict[str, Any], name: str) -> dict[str, Any]:
    descriptor = streams.get(name)
    if not isinstance(descriptor, dict):
        raise EpisodeError(f"stream descriptor must be a mapping: {name}")
    return descriptor


def _finite_numeric_vector(value: Any, expected_length: int, label: str) -> None:
    if not isinstance(value, list) or len(value) != expected_length:
        raise EpisodeError(f"{label} must contain {expected_length} values")
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        for item in value
    ):
        raise EpisodeError(f"{label} must contain only finite numeric values")


def _validate_video(
    path: Path,
    role: str,
    expected_frames: int,
    expected_shape: list[int],
    expected_fps: float,
    cv2: Any,
) -> None:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise EpisodeError(f"{role} video is not a decodable video file")
        actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(actual_fps) or actual_fps <= 0 or not math.isclose(
            actual_fps, expected_fps, rel_tol=1e-3, abs_tol=1e-3
        ):
            raise EpisodeError(
                f"{role} video FPS {actual_fps} differs from manifest {expected_fps}"
            )

        decoded_frames = 0
        while True:
            available, frame = capture.read()
            if not available:
                break
            if (
                frame is None
                or frame.ndim != 3
                or list(frame.shape[:2]) != expected_shape[:2]
            ):
                raise EpisodeError(
                    f"{role} video frame {decoded_frames} has an unexpected shape"
                )
            decoded_frames += 1
    finally:
        capture.release()

    if decoded_frames != expected_frames:
        raise EpisodeError(
            f"{role} video contains {decoded_frames} decodable frames; "
            f"expected {expected_frames}"
        )


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
    source_snapshot = config_record.get("source_snapshot")
    if source_snapshot is not None:
        if not isinstance(source_snapshot, dict):
            raise EpisodeError("source config snapshot descriptor must be a mapping")
        source_config_path = _existing_file(
            episode_dir,
            source_snapshot.get("storage_ref"),
            "source config snapshot",
        )
        if _sha256(source_config_path) != source_snapshot.get("sha256"):
            raise EpisodeError("source config snapshot SHA-256 does not match manifest")

    streams = manifest.get("streams")
    if not isinstance(streams, dict):
        raise EpisodeError("streams must be a mapping")
    missing_streams = REQUIRED_STREAMS - set(streams)
    if missing_streams:
        raise EpisodeError(f"missing required streams: {sorted(missing_streams)}")
    for stream_name in REQUIRED_STREAMS:
        _stream_descriptor(streams, stream_name)

    robot_stream = _stream_descriptor(streams, "robot_state")
    expected_dof_names = robot_stream.get("dof_names")
    expected_dof_types = robot_stream.get("dof_types")
    if (
        not isinstance(expected_dof_names, list)
        or not expected_dof_names
        or any(not isinstance(name, str) or not name for name in expected_dof_names)
        or len(set(expected_dof_names)) != len(expected_dof_names)
    ):
        raise EpisodeError("robot_state stream must declare unique non-empty dof_names")
    if not isinstance(expected_dof_types, list) or len(expected_dof_types) != len(
        expected_dof_names
    ):
        raise EpisodeError("robot_state stream must declare one dof_type per DOF")
    if robot_stream.get("shape") != [len(expected_dof_names)]:
        raise EpisodeError("robot_state stream shape does not match declared DOFs")

    effort_stream = _stream_descriptor(streams, "joint_effort")
    if (
        effort_stream.get("shape") != [len(expected_dof_names)]
        or effort_stream.get("dof_names") != expected_dof_names
        or effort_stream.get("dof_types") != expected_dof_types
    ):
        raise EpisodeError("joint_effort stream does not match robot_state DOF contract")

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
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on validation host
        raise EpisodeError("OpenCV is required to validate recorded videos") from exc

    with timeline_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EpisodeError(
                    f"invalid timeline JSON at line {line_number}: {exc}"
                ) from exc
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
            if robot_state["dof_names"] != expected_dof_names:
                raise EpisodeError(f"robot_state DOF order changed at line {line_number}")
            if robot_state["dof_types"] != expected_dof_types:
                raise EpisodeError(f"robot_state DOF types changed at line {line_number}")
            for field in ("positions", "velocities", "efforts"):
                _finite_numeric_vector(
                    robot_state[field],
                    len(expected_dof_names),
                    f"robot_state.{field} at line {line_number}",
                )

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
                if (
                    frame.dtype != np.uint8
                    or frame.ndim != 3
                    or frame.shape[2] not in (3, 4)
                ):
                    raise EpisodeError(
                        f"{role} image at line {line_number} must be HWC uint8 RGB/RGBA"
                    )
                declared = _stream_descriptor(
                    streams, f"observation.image.{role}"
                ).get("shape")
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
    video_encoding = manifest.get("artifacts", {}).get("video_encoding")
    if not isinstance(video_encoding, dict):
        raise EpisodeError("video_encoding descriptor is required")
    video_fps = video_encoding.get("fps")
    if (
        isinstance(video_fps, bool)
        or not isinstance(video_fps, (int, float))
        or not math.isfinite(float(video_fps))
        or float(video_fps) <= 0
    ):
        raise EpisodeError("video_encoding.fps must be a positive finite number")
    for role, video_ref in videos.items():
        video_path = _existing_file(episode_dir, video_ref, f"{role} video")
        declared_shape = _stream_descriptor(
            streams, f"observation.image.{role}"
        ).get("shape")
        if (
            not isinstance(declared_shape, list)
            or len(declared_shape) != 3
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0
                for item in declared_shape
            )
        ):
            raise EpisodeError(f"{role} stream must declare a positive HWC shape")
        _validate_video(
            video_path,
            role,
            observations,
            declared_shape,
            float(video_fps),
            cv2,
        )

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
