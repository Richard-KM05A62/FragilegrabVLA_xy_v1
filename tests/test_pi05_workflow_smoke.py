from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "create_pi05_workflow_smoke_stats",
    ROOT / "scripts/policy/create_pi05_workflow_smoke_stats.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_smoke_stats_are_transport_only_and_openpi_shaped(tmp_path: Path) -> None:
    output = tmp_path / "smoke"
    provenance = MODULE.create_smoke_stats(output, "piper_x_workflow_smoke_test")
    document = json.loads((output / "norm_stats.json").read_text(encoding="utf-8"))
    stats = document["norm_stats"]

    assert provenance["purpose"] == "transport_only"
    assert provenance["execution_permitted"] is False
    assert set(stats) == {"state", "actions"}
    for value in stats.values():
        assert set(value) == {"mean", "std", "q01", "q99"}
        np.testing.assert_array_equal(value["mean"], np.zeros(7))
        np.testing.assert_array_equal(value["std"], np.ones(7))


def test_smoke_stats_refuse_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "smoke"
    MODULE.create_smoke_stats(output, "piper_x_workflow_smoke_test")
    try:
        MODULE.create_smoke_stats(output, "piper_x_workflow_smoke_test")
    except FileExistsError:
        pass
    else:
        raise AssertionError("smoke stats must not overwrite an existing directory")
