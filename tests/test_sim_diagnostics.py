from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fragilegrab.benchmark.diagnostics import (
    analyze_pi05_timeline,
    audit_piper_transform_alignment,
)


def _asymmetric_stats() -> dict:
    return {
        "state": {"q01": [-2, -1, -3, -4, -2, -1, -0.2], "q99": [3, 4, 2, 1, 5, 2, 1.2]},
        "actions": {"q01": [-0.3, -0.2, -0.4, -0.1, -0.5, -0.6, -0.1], "q99": [0.7, 0.5, 0.3, 0.8, 0.2, 0.4, 1.1]},
    }


def test_transform_alignment_preserves_joint_sign_and_inverts_gripper() -> None:
    report = audit_piper_transform_alignment(
        state=np.array([0.4, -0.2, 0.7, -0.5, 0.1, 0.2, 0.25]),
        norm_stats=_asymmetric_stats(),
        norm_stats_id="piper_x_test_v0.1",
        normalization_purpose="dataset",
        joint_probe_rad=0.05,
        gripper_closures=[0.0, 0.5, 1.0],
    )
    assert report["status"] == "passed"
    assert all(row["passed"] for row in report["joint_alignment"])
    assert report["gripper_alignment"]["passed"] is True
    assert [row["actuator_target_m"] for row in report["gripper_alignment"]["samples"]] == [
        0.05,
        0.025,
        0.0,
    ]
    assert report["padding"]["passed"] is True


def test_transform_alignment_rejects_reversed_quantile_range() -> None:
    stats = _asymmetric_stats()
    stats["actions"]["q99"][2] = stats["actions"]["q01"][2]
    with np.testing.assert_raises_regex(ValueError, "quantile range"):
        audit_piper_transform_alignment(
            state=np.zeros(7),
            norm_stats=stats,
            norm_stats_id="piper_x_test_v0.1",
            normalization_purpose="dataset",
            joint_probe_rad=0.05,
            gripper_closures=[0.0, 1.0],
        )


def test_trace_inspection_reports_range_gripper_and_chunk_boundary(tmp_path: Path) -> None:
    limits = [[-1.0, 1.0]] * 6 + [[0.0, 0.05]]
    (tmp_path / "model_audit.json").write_text(json.dumps({"actuator_ctrlrange": limits}))
    first = np.zeros((50, 7))
    first[:, 6] = np.linspace(0.0, 1.0, 50)
    decoded_first = first.copy()
    decoded_first[:, 6] = 0.05 - 0.05 * first[:, 6]
    second = first.copy()
    second[:, 0] = 2.0
    decoded_second = decoded_first.copy()
    decoded_second[:, 0] = 2.0
    rows = [
        {"record_type": "observation", "joint_position": [0.0] * 8},
        {
            "record_type": "policy_query",
            "query_id": "q0",
            "simulation_time_s": 0.0,
            "client_round_trip_ms": 90.0,
            "predicted_action_chunk": first.tolist(),
            "decoded_actuator_chunk": decoded_first.tolist(),
        },
        {"record_type": "execution", "query_id": "q0", "executed_action": decoded_first[0].tolist()},
        {"record_type": "observation", "joint_position": [0.1] * 8},
        {
            "record_type": "policy_query",
            "query_id": "q1",
            "simulation_time_s": 0.05,
            "client_round_trip_ms": 95.0,
            "predicted_action_chunk": second.tolist(),
            "decoded_actuator_chunk": decoded_second.tolist(),
        },
    ]
    (tmp_path / "timeline.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    report = analyze_pi05_timeline(tmp_path)
    assert report["query_count"] == 2
    assert report["normalized_raw_output_available"] is False
    assert report["queries"][0]["gripper_closure_max"] == 1.0
    assert report["queries"][1]["range_violation_count"] == 50
    assert report["queries"][1]["first_proposal_minus_previous_executed"][0] == 2.0
