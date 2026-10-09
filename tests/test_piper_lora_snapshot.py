import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.policy.piper_lora_snapshot import (
    SnapshotIdentity,
    load_lora_snapshot,
    overlay_lora_params,
    save_lora_snapshot,
    select_adapter_params,
    select_lora_params,
)


def params() -> dict:
    return {
        "llm": {
            "kernel": np.ones((2, 2), dtype=np.float32),
            "attn_lora_a": np.arange(6, dtype=np.float32).reshape(2, 3),
        },
        "llm_1": {"ffn_lora_b": np.full((3, 2), 2, dtype=np.float32)},
    }


def identity(step: int = 500) -> SnapshotIdentity:
    return SnapshotIdentity(
        step=step,
        openpi_commit="15a9616",
        base_params_sha256="base-sha",
        dataset_id="local/piper",
        norm_stats_sha256="norm-sha",
        action_horizon=20,
        batch_size=8,
    )


def test_selects_only_lora_leaves() -> None:
    selected = select_lora_params(params())
    assert set(selected) == {"llm/attn_lora_a", "llm_1/ffn_lora_b"}


def test_adapter_selection_keeps_official_non_llm_trainable_leaves() -> None:
    selected = select_adapter_params(
        {
            "llm": {"attn_lora_a": np.ones((2, 2), dtype=np.float32)},
            "action_in_proj": {"kernel": np.ones((2, 2), dtype=np.float32)},
        }
    )
    assert set(selected) == {"llm/attn_lora_a", "action_in_proj/kernel"}


def test_snapshot_roundtrip_and_overlay(tmp_path: Path) -> None:
    snapshot = tmp_path / "step_000500"
    manifest = save_lora_snapshot(
        snapshot,
        {
            "llm": {"attn_lora_a": params()["llm"]["attn_lora_a"]},
            "llm_1": params()["llm_1"],
            "action_in_proj": {"kernel": np.ones((2, 2), dtype=np.float32)},
        },
        identity(),
        minimum_free_bytes_after_save=1,
    )
    arrays, loaded_manifest = load_lora_snapshot(snapshot)
    assert loaded_manifest == manifest
    assert manifest["step"] == 500
    assert manifest["parameter_leaf_count"] == 3

    target = params()
    target["action_in_proj"] = {"kernel": np.zeros((2, 2), dtype=np.float32)}
    target["llm"]["attn_lora_a"][:] = -1
    overlaid = overlay_lora_params(target, snapshot)
    np.testing.assert_array_equal(overlaid["llm"]["attn_lora_a"], arrays["llm/attn_lora_a"])
    np.testing.assert_array_equal(overlaid["llm"]["kernel"], np.ones((2, 2), dtype=np.float32))
    np.testing.assert_array_equal(overlaid["action_in_proj"]["kernel"], np.ones((2, 2), dtype=np.float32))


def test_overlay_preserves_abstract_reference_shape(tmp_path: Path) -> None:
    class AbstractLeaf:
        def __init__(self, shape: tuple[int, ...]):
            self.shape = shape

    snapshot = tmp_path / "step_000500"
    saved = np.arange(6, dtype=np.float32).reshape(2, 3)
    save_lora_snapshot(
        snapshot,
        {"llm": {"attn_lora_a": saved}},
        identity(),
        minimum_free_bytes_after_save=1,
    )
    target: dict[str, Any] = {
        "llm": {
            "kernel": np.ones((2, 2), dtype=np.float32),
            "attn_lora_a": AbstractLeaf((2, 3)),
        }
    }
    overlaid = overlay_lora_params(target, snapshot)
    np.testing.assert_array_equal(overlaid["llm"]["attn_lora_a"], saved)
    np.testing.assert_array_equal(overlaid["llm"]["kernel"], np.ones((2, 2), dtype=np.float32))


def test_rejects_nonfinite_lora_leaf() -> None:
    value = params()
    value["llm"]["attn_lora_a"][0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        select_lora_params(value)


def test_rejects_contract_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="action_horizon=20"):
        save_lora_snapshot(
            tmp_path / "bad",
            {"llm": {"attn_lora_a": params()["llm"]["attn_lora_a"]}},
            dataclasses.replace(identity(), action_horizon=16),
            minimum_free_bytes_after_save=1,
        )
