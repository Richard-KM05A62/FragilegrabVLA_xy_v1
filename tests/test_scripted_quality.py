from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import yaml

from fragilegrab.benchmark.scripted_quality import (
    summarize_episode_metrics,
    validate_episode_structure,
    validate_scripted_dataset_config,
)


def config() -> dict:
    return {
        "schema_version": "fragilegrab.scripted_provisional/0.4",
        "task_set_id": "test_task_set",
        "dataset_repo_id": "local/test",
        "norm_stats_id": "piper_x_test",
        "control_hz": 20,
        "resolution_hw": [48, 64],
        "video_fourcc": "mp4v",
        "video_extension": "mp4",
        "quality_control": {
            "metric_thresholds_enabled": False,
            "metric_thresholds": {},
        },
        "variations": [{
            "id": "one",
            "task_type": "pick",
            "instruction": "Pick up the object",
            "seed": 1,
            "source_xy_m": [0.0, 0.0],
            "goal_kind": "none",
            "object_yaw_rad": 0.0,
            "wrist_yaw_rad": 0.0,
            "robot_joint1_reset_delta_rad": 0.0,
        }],
    }


def structural_inputs(frames: int = 4):
    diagnostics = {
        "frame_count": frames,
        "finite_state_action": True,
        "shape_valid": True,
        "action_within_hard_limits": True,
        "timestamps_s": [index * 0.05 for index in range(frames)],
        "stream_lengths": {name: frames for name in ("front", "wrist", "state", "action", "timestamp")},
        "max_joint_action_step_rad": 5.0,
        "max_gripper_action_step_closure": 1.0,
        "tracking_error_samples": np.ones((frames, 7)).tolist(),
        "bilateral_contact_physics_steps": 0,
        "lifted_control_ticks": 0,
        "image_stats": {
            "front": {"mean_std": 0.0, "mean_temporal_absdiff": 0.0},
            "wrist": {"mean_std": 0.0, "mean_temporal_absdiff": 0.0},
        },
    }
    videos = {
        role: {"frames": frames, "fps": 20.0, "height": 48, "width": 64}
        for role in ("front", "wrist")
    }
    return diagnostics, videos


def test_only_structural_errors_block_episode() -> None:
    cfg = config()
    diagnostics, videos = structural_inputs()
    report = validate_episode_structure(cfg, diagnostics, videos)
    assert report["passed"]
    # Deliberately poor motion/contact/image metrics are recorded but cannot reject.
    metrics = summarize_episode_metrics({"task_success": False}, diagnostics)
    assert metrics["quality_flag"] == "warning"
    assert metrics["rejection_reason"] is None
    assert metrics["warnings"] == ["task_success_false"]
    assert metrics["thresholds_applied"] is False


def test_mismatched_stream_length_is_blocking() -> None:
    cfg = config()
    diagnostics, videos = structural_inputs()
    diagnostics = copy.deepcopy(diagnostics)
    diagnostics["stream_lengths"]["action"] -= 1
    report = validate_episode_structure(cfg, diagnostics, videos)
    assert not report["passed"]
    assert report["failed_checks"] == ["image_state_action_lengths_match"]


def test_config_requires_metric_thresholds_to_be_disabled_and_empty() -> None:
    cfg = config()
    validate_scripted_dataset_config(cfg)
    cfg["quality_control"] = {
        "metric_thresholds_enabled": True,
        "metric_thresholds": {"invented": 1.0},
    }
    try:
        validate_scripted_dataset_config(cfg)
    except ValueError as exc:
        assert "disabled" in str(exc)
    else:
        raise AssertionError("enabled provisional QC threshold was accepted")


def test_config_validation_rejects_duplicate_variations() -> None:
    cfg = config()
    cfg["variations"] = [{"id": "same", "seed": 1}, {"id": "same", "seed": 2}]
    try:
        validate_scripted_dataset_config(cfg)
    except ValueError as exc:
        assert "variation ids" in str(exc)
    else:
        raise AssertionError("duplicate variation ids were accepted")


def test_task_type_requires_matching_goal_semantics() -> None:
    cfg = config()
    cfg["variations"][0]["goal_kind"] = "body"
    try:
        validate_scripted_dataset_config(cfg)
    except ValueError as exc:
        assert "goal_kind=none" in str(exc)
    else:
        raise AssertionError("pick task with a body goal was accepted")


def test_repository_multitask_config_has_balanced_24_episode_quota() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(
        (root / "configs/benchmark/gentle_demo_pilot.yaml").read_text(encoding="utf-8")
    )["values"]
    validate_scripted_dataset_config(cfg)
    counts = {
        task_type: sum(v["task_type"] == task_type for v in cfg["variations"])
        for task_type in ("pick", "pick_and_place", "relocate", "container_place")
    }
    assert len(cfg["variations"]) == 24
    assert counts == {
        "pick": 6,
        "pick_and_place": 6,
        "relocate": 6,
        "container_place": 6,
    }

