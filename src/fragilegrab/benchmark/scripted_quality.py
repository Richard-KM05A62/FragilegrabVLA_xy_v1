"""Minimal structural validation and non-blocking metrics for scripted episodes."""

from __future__ import annotations

from typing import Any

import numpy as np

SCRIPTED_TASK_TYPES = ("pick", "pick_and_place", "relocate", "container_place")
TASK_GOAL_KINDS = {
    "pick": "none",
    "pick_and_place": "tabletop",
    "relocate": "tabletop",
    "container_place": "body",
}


def validate_scripted_dataset_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "task_set_id",
        "dataset_repo_id",
        "norm_stats_id",
        "resolution_hw",
        "control_hz",
        "video_fourcc",
        "video_extension",
        "quality_control",
        "variations",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"scripted dataset config is missing: {missing}")
    if config["schema_version"] != "fragilegrab.scripted_provisional/0.4":
        raise ValueError("unsupported scripted dataset schema_version")
    if not str(config["dataset_repo_id"]).startswith("local/"):
        raise ValueError("provisional dataset_repo_id must use the local/ namespace")
    if not str(config["norm_stats_id"]).startswith("piper_x_"):
        raise ValueError("norm_stats_id must start with piper_x_")
    if float(config["control_hz"]) != 20.0:
        raise ValueError("the confirmed provisional dataset control rate is 20 Hz")
    if config["video_fourcc"] != "mp4v" or config["video_extension"] != "mp4":
        raise ValueError("provisional review videos must use mp4v in an .mp4 container")
    if len(config["resolution_hw"]) != 2 or any(int(x) <= 0 for x in config["resolution_hw"]):
        raise ValueError("resolution_hw must contain two positive dimensions")
    qc = config["quality_control"]
    if qc != {"metric_thresholds_enabled": False, "metric_thresholds": {}}:
        raise ValueError("provisional metric thresholds must remain centralized, empty and disabled")
    variations = config["variations"]
    ids = [item.get("id") for item in variations]
    seeds = [item.get("seed") for item in variations]
    if not variations or len(ids) != len(set(ids)) or None in ids:
        raise ValueError("variation ids must be non-empty and unique")
    if len(seeds) != len(set(seeds)) or any(type(seed) is not int for seed in seeds):
        raise ValueError("variation seeds must be unique integers")
    for variation in variations:
        missing_fields = {
            "id", "task_type", "instruction", "seed", "source_xy_m", "goal_kind",
            "object_yaw_rad", "wrist_yaw_rad", "robot_joint1_reset_delta_rad",
        } - variation.keys()
        if missing_fields:
            raise ValueError(f"variation {variation.get('id')} is missing: {sorted(missing_fields)}")
        task_type = variation["task_type"]
        if task_type not in SCRIPTED_TASK_TYPES:
            raise ValueError(f"unsupported scripted task_type: {task_type}")
        expected_goal_kind = TASK_GOAL_KINDS[task_type]
        if variation["goal_kind"] != expected_goal_kind:
            raise ValueError(
                f"variation {variation['id']} requires goal_kind={expected_goal_kind}"
            )
        if not isinstance(variation["instruction"], str) or not variation["instruction"].strip():
            raise ValueError(f"variation {variation['id']} instruction must be non-empty")
        source_xy = np.asarray(variation["source_xy_m"], dtype=np.float64)
        if source_xy.shape != (2,) or not np.isfinite(source_xy).all():
            raise ValueError(f"variation {variation['id']} source_xy_m must be finite xy")
        if expected_goal_kind == "none":
            if "goal_xy_m" in variation:
                raise ValueError(f"pick variation {variation['id']} must not define goal_xy_m")
        else:
            goal_xy = np.asarray(variation.get("goal_xy_m"), dtype=np.float64)
            if goal_xy.shape != (2,) or not np.isfinite(goal_xy).all():
                raise ValueError(f"variation {variation['id']} goal_xy_m must be finite xy")
        scalar_fields = (
            "object_yaw_rad", "wrist_yaw_rad", "robot_joint1_reset_delta_rad"
        )
        if any(not np.isfinite(float(variation[field])) for field in scalar_fields):
            raise ValueError(f"variation {variation['id']} contains a non-finite scalar")


def validate_episode_structure(
    config: dict[str, Any], diagnostics: dict[str, Any], videos: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Only check conditions that make serialization or alignment unusable."""
    lengths = {name: int(value) for name, value in diagnostics["stream_lengths"].items()}
    timestamps = np.asarray(diagnostics["timestamps_s"], dtype=np.float64)
    frame_count = int(diagnostics["frame_count"])
    checks = {
        "non_empty_episode": frame_count > 0,
        "finite_state_action": bool(diagnostics["finite_state_action"]),
        "shape_valid": bool(diagnostics["shape_valid"]),
        "timestamp_count_matches": timestamps.shape == (frame_count,),
        "timestamps_finite_and_strictly_increasing": bool(
            timestamps.shape == (frame_count,)
            and np.isfinite(timestamps).all()
            and (frame_count == 1 or np.all(np.diff(timestamps) > 0))
        ),
        "image_state_action_lengths_match": bool(
            lengths and set(lengths.values()) == {frame_count}
        ),
        "action_within_model_hard_limits": bool(diagnostics["action_within_hard_limits"]),
    }
    for role in ("front", "wrist"):
        video = videos[role]
        checks[f"{role}_video_length_matches"] = int(video["frames"]) == frame_count
        checks[f"{role}_video_resolution_matches"] = [
            int(video["height"]),
            int(video["width"]),
        ] == list(config["resolution_hw"])
        checks[f"{role}_video_fps_matches"] = float(video["fps"]) == float(config["control_hz"])
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "frame_count": frame_count,
        "stream_lengths": lengths,
    }


def summarize_episode_metrics(result: dict[str, Any], diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Return descriptive metrics only; no metric here can reject an episode."""
    tracking = np.asarray(diagnostics["tracking_error_samples"], dtype=np.float64)
    tracking_p95 = (
        np.quantile(tracking, 0.95, axis=0).tolist()
        if tracking.ndim == 2 and tracking.shape[0] > 0 and tracking.shape[1] == 7
        else None
    )
    warnings = []
    if result.get("task_success") is not True:
        warnings.append("task_success_false")
    return {
        "quality_flag": "warning" if warnings else "unreviewed",
        "rejection_reason": None,
        "warnings": warnings,
        "task_result": result,
        "max_joint_action_step_rad": float(diagnostics["max_joint_action_step_rad"]),
        "max_gripper_action_step_closedness": float(
            diagnostics["max_gripper_action_step_closure"]
        ),
        "tracking_error_p95": tracking_p95,
        "bilateral_contact_physics_steps": int(
            diagnostics["bilateral_contact_physics_steps"]
        ),
        "lifted_control_ticks": int(diagnostics["lifted_control_ticks"]),
        "image_stats": diagnostics["image_stats"],
        "thresholds_applied": False,
    }
