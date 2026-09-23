from __future__ import annotations

import numpy as np

from fragilegrab.benchmark.evaluator import GraspTendencyEvaluator


CFG = {
    "approach_progress_m": 0.05,
    "pregrasp_distance_m": 0.10,
    "alignment_tolerance_deg": 35.0,
    "desired_tool_z_world": [0.0, 0.0, -1.0],
    "close_start_closure": 0.20,
    "close_near_object_distance_m": 0.10,
    "contact_epsilon_N": 0.0001,
    "lift_attempt_m": 0.01,
    "stable_lift_m": 0.04,
    "stable_lift_hold_s": 0.10,
    "trace_period_s": 0.05,
}


def _state(time_s, distance, closure=0.0, contacts=(), lift=0.0):
    return {
        "time_s": time_s,
        "object_position_m": np.array([0.0, 0.0, lift]),
        "grasp_position_m": np.array([distance, 0.0, lift]),
        "wrist_rotation": np.diag([1.0, -1.0, -1.0]),
        "gripper_closure": closure,
        "contacts": [
            {"body1": finger, "body2": 10, "normal_force_N": 1.0} for finger in contacts
        ],
    }


def test_grasp_tendency_reaches_l4_for_near_close_and_contact():
    evaluator = GraspTendencyEvaluator(
        CFG,
        {"object_position_m": [0, 0, 0], "grasp_position_m": [0.3, 0, 0]},
        target_id=10,
        finger_ids=[20, 21],
    )
    evaluator.update(_state(0.05, 0.08, closure=0.5, contacts=(20,)), 0.05)
    result = evaluator.result()
    assert result["highest_stage"] == 4
    assert result["gripper_close_near_object"] is True
    assert result["finger_object_contact"] is True


def test_grasp_tendency_reaches_l6_after_bilateral_lift_hold():
    evaluator = GraspTendencyEvaluator(
        CFG,
        {"object_position_m": [0, 0, 0], "grasp_position_m": [0.3, 0, 0]},
        target_id=10,
        finger_ids=[20, 21],
    )
    evaluator.update(_state(0.05, 0.06, closure=0.7, contacts=(20, 21), lift=0.05), 0.05)
    evaluator.update(_state(0.10, 0.06, closure=0.7, contacts=(20, 21), lift=0.05), 0.05)
    result = evaluator.result()
    assert result["highest_stage"] == 6
    assert result["stable_lift"] is True
    assert result["object_maximum_lift_m"] == 0.05
