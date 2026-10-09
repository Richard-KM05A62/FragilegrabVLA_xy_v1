#!/usr/bin/env python3
"""Disk-conscious adapter snapshots for the locked OpenPI LoRA runtime.

OpenPI's official LoRA freeze filter trains LoRA leaves plus small non-LLM
projection/head leaves.  The caller must pass the state selected by
``TrainConfig.trainable_filter``; every supplied leaf is stored.  Frozen base
parameters are never written.  Optimizer state is intentionally outside this
format.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping
import uuid

import numpy as np


SCHEMA_VERSION = "fragilegrab.openpi_adapter_snapshot/0.2"


@dataclasses.dataclass(frozen=True)
class SnapshotIdentity:
    step: int
    openpi_commit: str
    base_params_sha256: str
    dataset_id: str
    norm_stats_sha256: str
    action_horizon: int
    batch_size: int
    model_config: str = "pi05_piper_x_lora_h20"

    def validate(self) -> None:
        if self.step < 0:
            raise ValueError("step must be non-negative")
        if self.action_horizon != 20:
            raise ValueError("this snapshot contract requires action_horizon=20")
        if self.batch_size != 8:
            raise ValueError("this snapshot contract requires batch_size=8")
        for name in ("openpi_commit", "base_params_sha256", "dataset_id", "norm_stats_sha256"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")


def _as_pure_mapping(params: Any) -> Mapping[str, Any]:
    if hasattr(params, "to_pure_dict"):
        params = params.to_pure_dict()
    if not isinstance(params, Mapping):
        raise TypeError("params must be a mapping or expose to_pure_dict()")
    return params


def _flatten(tree: Mapping[str, Any], prefix: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
    flat: dict[str, np.ndarray] = {}
    for key, value in tree.items():
        path = (*prefix, str(key))
        if isinstance(value, Mapping):
            flat.update(_flatten(value, path))
        else:
            flat["/".join(path)] = np.asarray(value)
    return flat


def _flatten_raw(tree: Mapping[str, Any], prefix: tuple[str, ...] = ()) -> dict[str, Any]:
    """Flatten without coercing abstract JAX leaves to NumPy object scalars."""
    flat: dict[str, Any] = {}
    for key, value in tree.items():
        path = (*prefix, str(key))
        if isinstance(value, Mapping):
            flat.update(_flatten_raw(value, path))
        else:
            flat["/".join(path)] = value
    return flat


def select_lora_params(params: Any) -> dict[str, np.ndarray]:
    selected = {
        path: value
        for path, value in _flatten(_as_pure_mapping(params)).items()
        if "lora" in path.lower()
    }
    if not selected:
        raise ValueError("no LoRA parameter leaves were found")
    for path, value in selected.items():
        if value.dtype == object:
            raise TypeError(f"LoRA leaf {path!r} has unsupported object dtype")
        if not np.issubdtype(value.dtype, np.number):
            raise TypeError(f"LoRA leaf {path!r} is not numeric: {value.dtype}")
        if not np.isfinite(value).all():
            raise ValueError(f"LoRA leaf {path!r} contains NaN or Inf")
    return selected


def select_adapter_params(trainable_params: Any) -> dict[str, np.ndarray]:
    """Validate all caller-supplied trainable leaves and retain them all."""
    selected = _flatten(_as_pure_mapping(trainable_params))
    if not selected:
        raise ValueError("trainable parameter set is empty")
    if not any("lora" in path.lower() for path in selected):
        raise ValueError("trainable parameter set contains no LoRA leaves")
    for path, value in selected.items():
        if value.dtype == object:
            raise TypeError(f"trainable leaf {path!r} has unsupported object dtype")
        if not np.issubdtype(value.dtype, np.number):
            raise TypeError(f"trainable leaf {path!r} is not numeric: {value.dtype}")
        if not np.isfinite(value).all():
            raise ValueError(f"trainable leaf {path!r} contains NaN or Inf")
    return selected


def estimate_snapshot_bytes(params: Any) -> int:
    return sum(value.nbytes for value in select_adapter_params(params).values())


def require_free_space(directory: Path, required_bytes: int) -> None:
    if required_bytes <= 0:
        raise ValueError("required_bytes must be positive")
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    available = shutil.disk_usage(directory).free
    if available < required_bytes:
        raise OSError(
            f"insufficient free space at {directory}: required={required_bytes}, available={available}"
        )


def _array_digest(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def save_lora_snapshot(
    output: Path,
    params: Any,
    identity: SnapshotIdentity,
    *,
    minimum_free_bytes_after_save: int = 1024**3,
) -> dict[str, Any]:
    identity.validate()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"snapshot already exists: {output}")

    arrays = select_adapter_params(params)
    raw_bytes = sum(value.nbytes for value in arrays.values())
    # NPZ may temporarily coexist with buffers and the atomic staging directory.
    require_free_space(output.parent, raw_bytes * 2 + minimum_free_bytes_after_save)

    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        archive_path = staging / "adapter_params.npz"
        np.savez_compressed(archive_path, **arrays)
        archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            **dataclasses.asdict(identity),
            "parameter_leaf_count": len(arrays),
            "parameter_selection": "openpi_train_config.trainable_filter",
            "uncompressed_parameter_bytes": raw_bytes,
            "archive_bytes": archive_path.stat().st_size,
            "archive_sha256": archive_sha256,
            "parameters": {
                path: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256": _array_digest(value),
                }
                for path, value in sorted(arrays.items())
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.rename(staging, output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_lora_snapshot(snapshot: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    snapshot = snapshot.resolve()
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported snapshot schema: {manifest.get('schema_version')!r}")
    archive_path = snapshot / "adapter_params.npz"
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != manifest.get("archive_sha256"):
        raise ValueError("LoRA archive SHA-256 does not match manifest")
    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    expected = manifest.get("parameters", {})
    if set(arrays) != set(expected):
        raise ValueError("LoRA parameter paths do not match manifest")
    for path, value in arrays.items():
        record = expected[path]
        if list(value.shape) != record["shape"] or str(value.dtype) != record["dtype"]:
            raise ValueError(f"LoRA parameter metadata mismatch: {path}")
        if _array_digest(value) != record["sha256"]:
            raise ValueError(f"LoRA parameter SHA-256 mismatch: {path}")
    return arrays, manifest


def overlay_lora_params(params: Any, snapshot: Path) -> dict[str, Any]:
    pure = _as_pure_mapping(params)
    # CheckpointWeightLoader retains missing LoRA leaves as
    # jax.ShapeDtypeStruct references.  np.asarray(ShapeDtypeStruct) produces
    # a scalar object array, so overlay must preserve those leaves until the
    # real adapter arrays replace them.
    flat = _flatten_raw(pure)
    arrays, _ = load_lora_snapshot(snapshot)
    missing = set(arrays) - set(flat)
    if missing:
        raise KeyError(f"snapshot contains unknown parameter paths: {sorted(missing)[:5]}")
    for path, value in arrays.items():
        if flat[path].shape != value.shape:
            raise ValueError(
                f"LoRA parameter shape mismatch: {path}; "
                f"model={flat[path].shape}, snapshot={value.shape}"
            )
        flat[path] = value

    rebuilt: dict[str, Any] = {}
    for path, value in flat.items():
        cursor = rebuilt
        parts = path.split("/")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return rebuilt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    arrays, manifest = load_lora_snapshot(args.snapshot)
    print(json.dumps({
        "snapshot": str(args.snapshot.resolve()),
        "step": manifest["step"],
        "parameter_leaf_count": len(arrays),
        "archive_bytes": manifest["archive_bytes"],
    }, indent=2))


if __name__ == "__main__":
    main()
