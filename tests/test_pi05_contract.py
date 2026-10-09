from __future__ import annotations

import numpy as np
import pytest

from fragilegrab.benchmark.pi05_client import _packb, _unpackb
from fragilegrab.benchmark.pi05_contract import (
    ACTION_SEMANTICS,
    CONTRACT_ID,
    OPENPI_COMMIT,
    OPENPI_INTERFACE_BASE_COMMIT,
    Pi05ContractError,
    PiperPi05Contract,
)
from fragilegrab.benchmark.pi05_runner import project_diagnostic_target


def contract() -> PiperPi05Contract:
    return PiperPi05Contract("piper_x_test_stats")


def metadata() -> dict:
    return {
        "openpi_commit": OPENPI_COMMIT,
        "openpi_interface_base_commit": OPENPI_INTERFACE_BASE_COMMIT,
        "contract_id": CONTRACT_ID,
        "norm_stats_id": "piper_x_test_stats",
        "normalization_purpose": "piper_x_dataset_calibration",
        "execution_permitted": True,
        "action_horizon": 50,
        "effective_action_dim": 7,
        "action_semantics": ACTION_SEMANTICS,
        "image_roles": {
            "base_0_rgb": "present",
            "left_wrist_0_rgb": "present",
            "right_wrist_0_rgb": "masked",
        },
    }


def test_observation_maps_two_fingers_to_normalized_closure() -> None:
    value = contract()
    observation = {
        "images": {
            "base_rgb": np.zeros((16, 24, 3), dtype=np.uint8),
            "wrist_rgb": np.ones((16, 24, 3), dtype=np.uint8),
        },
        "joint_position": np.array([0, 1, 2, 3, 4, 5, 0.05, -0.05]),
    }
    request = value.encode_observation(observation, "pick up the object")
    np.testing.assert_array_equal(request["observation/state"], [0, 1, 2, 3, 4, 5, 0])
    assert request["observation/image"].shape == (16, 24, 3)
    np.testing.assert_array_equal(request["observation/wrist_image"], 1)

    observation["joint_position"][-2:] = 0
    request = value.encode_observation(observation, "pick up the object")
    assert request["observation/state"][-1] == pytest.approx(1.0)


def test_observation_requires_real_wrist_image() -> None:
    value = contract()
    observation = {
        "images": {"base_rgb": np.zeros((16, 24, 3), dtype=np.uint8)},
        "joint_position": np.array([0, 1, 2, 3, 4, 5, 0.05, -0.05]),
    }
    with pytest.raises(Pi05ContractError, match="wrist_rgb"):
        value.encode_observation(observation, "pick up")


def test_observation_tolerates_tiny_gripper_servo_overshoot() -> None:
    value = contract()
    state = value.policy_state(np.array([0, 0, 0, 0, 0, 0, 0.05001, -0.05001]))
    assert state[-1] == pytest.approx(0.0)
    state = value.policy_state(np.array([0, 0, 0, 0, 0, 0, -0.0002, 0.0002]))
    assert state[-1] == pytest.approx(1.0)

    with pytest.raises(Pi05ContractError, match="观测位置越界"):
        value.policy_state(np.array([0, 0, 0, 0, 0, 0, 0.052, -0.052]))
    with pytest.raises(Pi05ContractError, match="观测位置越界"):
        value.policy_state(np.array([0, 0, 0, 0, 0, 0, -0.002, 0.002]))


def test_action_chunk_maps_closure_back_to_meter_actuator() -> None:
    value = contract()
    actions = np.zeros((50, 7))
    actions[:25, 6] = 0.0
    actions[25:, 6] = 1.0
    limits = np.array([[-10.0, 10.0]] * 6 + [[0.0, 0.05]])
    decoded = value.decode_action_chunk({"actions": actions}, limits)
    np.testing.assert_allclose(decoded[:25, 6], 0.05)
    np.testing.assert_allclose(decoded[25:, 6], 0.0)


def test_horizon_20_lora_contract() -> None:
    value = PiperPi05Contract("piper_x_embodiment_norm20_v0_1", action_horizon=20)
    actions = np.zeros((20, 7))
    decoded = value.decode_action_chunk_proposal({"actions": actions})
    assert decoded.shape == (20, 7)
    expected = metadata()
    expected["norm_stats_id"] = "piper_x_embodiment_norm20_v0_1"
    expected["action_horizon"] = 20
    value.validate_server_metadata(expected)


def test_diagnostic_proposal_preserves_out_of_range_values_for_explicit_projection() -> None:
    value = contract()
    actions = np.zeros((50, 7))
    actions[0, 0] = 99.0
    actions[0, 6] = 1.2
    decoded = value.decode_action_chunk_proposal({"actions": actions})
    assert decoded[0, 0] == pytest.approx(99.0)
    assert decoded[0, 6] == pytest.approx(-0.01)

    limits = np.array([[-1.0, 1.0]] * 6 + [[0.0, 0.05]])
    with pytest.raises(Pi05ContractError, match=r"\[0, 1\]"):
        value.decode_action_chunk({"actions": actions}, limits)


def test_contract_rejects_wrong_server_and_unsafe_actions() -> None:
    value = contract()
    wrong = metadata()
    wrong["norm_stats_id"] = "other"
    with pytest.raises(Pi05ContractError, match="契约不匹配"):
        value.validate_server_metadata(wrong)

    actions = np.zeros((50, 7))
    actions[:, 0] = 99
    limits = np.array([[-1.0, 1.0]] * 6 + [[0.0, 0.05]])
    with pytest.raises(Pi05ContractError, match="ctrlrange"):
        value.decode_action_chunk({"actions": actions}, limits)

    with pytest.raises(Pi05ContractError, match="单步变化过大"):
        value.validate_execution_step(
            np.zeros(7),
            np.array([0.3, 0, 0, 0, 0, 0, 0]),
            max_joint_delta_rad=0.2,
            max_gripper_delta_m=0.01,
        )


def test_msgpack_protocol_round_trip_matches_numpy_shapes_and_dtypes() -> None:
    source = {
        "image": np.arange(24, dtype=np.uint8).reshape(2, 4, 3),
        "state": np.arange(7, dtype=np.float64),
        "scalar": np.float32(1.5),
    }
    restored = _unpackb(_packb(source))
    np.testing.assert_array_equal(restored["image"], source["image"])
    np.testing.assert_array_equal(restored["state"], source["state"])
    assert restored["scalar"].dtype == np.dtype("float32")


def test_valid_server_metadata() -> None:
    contract().validate_server_metadata(metadata())


def test_transport_only_normalization_is_rejected_by_execution_paths() -> None:
    transport_metadata = metadata()
    transport_metadata["normalization_purpose"] = "transport_only"
    transport_metadata["execution_permitted"] = False

    contract().validate_server_metadata(transport_metadata, require_execution=False)
    with pytest.raises(Pi05ContractError, match="仅允许传输测试"):
        contract().validate_server_metadata(transport_metadata)


def test_diagnostic_projection_records_range_and_rate_changes() -> None:
    previous = np.zeros(7)
    proposed = np.array([99.0, -0.03, 0.0, 0.0, 0.0, 0.0, -0.01])
    limits = np.array([[-1.0, 1.0]] * 6 + [[0.0, 0.05]])
    projected, evidence = project_diagnostic_target(
        previous,
        proposed,
        limits,
        max_joint_delta_rad=0.05,
        max_gripper_delta_m=0.0035,
    )
    np.testing.assert_allclose(projected, [0.05, -0.03, 0, 0, 0, 0, 0])
    assert evidence["range_changed_dims"] == [0, 6]
    assert evidence["rate_changed_dims"] == [0]
    assert evidence["changed"] is True


def test_non_piper_x_norm_stats_id_is_rejected() -> None:
    with pytest.raises(Pi05ContractError, match="piper_x_"):
        PiperPi05Contract("unscoped_calibration_v0.1")
