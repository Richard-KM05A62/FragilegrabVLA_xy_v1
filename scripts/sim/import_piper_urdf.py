#!/usr/bin/env python3
"""Import a materialized Piper X URDF with Isaac Sim 6.1.0 public API.

All physics-affecting importer choices are explicit command-line inputs.  The
script refuses Xacro input because Isaac Sim's URDFImporter requires ``.urdf``;
materialize the upstream gripper Xacro before invoking this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


ISAAC_VERSION = "6.1.0"


def _boolean(value: str) -> bool:
    normalized = value.lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _number_or_mapping(value: str) -> float | dict[str, float]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            "expected a finite number or JSON object mapping joint regex to number"
        ) from exc
    if isinstance(parsed, bool):
        raise argparse.ArgumentTypeError("boolean is not a drive parameter")
    if isinstance(parsed, (int, float)):
        numeric = float(parsed)
        if not math.isfinite(numeric):
            raise argparse.ArgumentTypeError("drive parameter must be finite")
        return numeric
    if isinstance(parsed, dict) and parsed:
        result: dict[str, float] = {}
        for key, item in parsed.items():
            if not isinstance(key, str) or not key:
                raise argparse.ArgumentTypeError("drive mapping keys must be strings")
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise argparse.ArgumentTypeError("drive mapping values must be numbers")
            numeric = float(item)
            if not math.isfinite(numeric):
                raise argparse.ArgumentTypeError("drive mapping values must be finite")
            result[key] = numeric
        return result
    raise argparse.ArgumentTypeError(
        "expected a finite number or non-empty JSON object mapping joint regex to number"
    )


def _ros_package(value: str) -> dict[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("ROS package mapping must be NAME=/absolute/path")
    name, raw_path = value.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    if not name or not path.is_dir():
        raise argparse.ArgumentTypeError("ROS package mapping must reference an existing directory")
    return {name: str(path)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--fix-base", type=_boolean, required=True)
    parser.add_argument("--merge-fixed-joints", type=_boolean, required=True)
    parser.add_argument("--merge-mesh", type=_boolean, required=True)
    parser.add_argument("--collision-from-visuals", type=_boolean, required=True)
    parser.add_argument("--allow-self-collision", type=_boolean, required=True)
    parser.add_argument("--run-asset-transformer", type=_boolean, required=True)
    parser.add_argument("--run-multi-physics-conversion", type=_boolean, required=True)
    parser.add_argument(
        "--ros-package",
        type=_ros_package,
        action="append",
        default=[],
        metavar="NAME=/ABSOLUTE/PATH",
    )
    parser.add_argument("--joint-drive-type", choices=("force", "acceleration"))
    parser.add_argument("--joint-target-type", choices=("none", "position", "velocity"))
    parser.add_argument("--override-joint-stiffness", type=_number_or_mapping)
    parser.add_argument("--override-joint-damping", type=_number_or_mapping)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    urdf_path = args.urdf.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if urdf_path.suffix.lower() != ".urdf" or not urdf_path.is_file():
        print("error: --urdf must reference an existing materialized .urdf file", file=sys.stderr)
        return 2
    if output_dir.exists():
        if not output_dir.is_dir():
            print(f"error: output path is not a directory: {output_dir}", file=sys.stderr)
            return 2
        if any(output_dir.iterdir()):
            print(f"error: output directory must be empty: {output_dir}", file=sys.stderr)
            return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    simulation_app: Any = None
    try:
        from isaacsim import SimulationApp

        simulation_app = SimulationApp({"headless": args.headless})
        import isaacsim.core.experimental.utils.app as app_utils

        for extension in (
            "isaacsim.asset.importer.urdf",
            "isaacsim.core.version",
        ):
            enabled = app_utils.enable_extension(extension)
            if not enabled and not app_utils.is_extension_enabled(extension):
                raise RuntimeError(f"unable to enable Isaac Sim extension: {extension}")
            if not app_utils.is_extension_enabled(extension):
                raise RuntimeError(f"Isaac Sim extension is not enabled: {extension}")

        from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
        from isaacsim.core.version import get_version

        version_parts = get_version()
        actual_version = ".".join(version_parts[2:5])
        if actual_version != ISAAC_VERSION:
            raise RuntimeError(
                f"Isaac Sim runtime {actual_version} does not match required {ISAAC_VERSION}"
            )

        import_config = URDFImporterConfig(
            urdf_path=str(urdf_path),
            usd_path=str(output_dir),
            merge_fixed_joints=args.merge_fixed_joints,
            merge_mesh=args.merge_mesh,
            collision_from_visuals=args.collision_from_visuals,
            allow_self_collision=args.allow_self_collision,
            ros_package_paths=args.ros_package,
            fix_base=args.fix_base,
            joint_drive_type=args.joint_drive_type,
            joint_target_type=args.joint_target_type,
            override_joint_stiffness=args.override_joint_stiffness,
            override_joint_damping=args.override_joint_damping,
            run_asset_transformer=args.run_asset_transformer,
            run_multi_physics_conversion=args.run_multi_physics_conversion,
        )
        output_path = Path(URDFImporter(import_config).import_urdf())
        if not output_path.is_absolute():
            output_path = output_dir / output_path
        output_path = output_path.resolve()
        if not output_path.is_file():
            raise RuntimeError(f"URDFImporter returned a missing output: {output_path}")
        try:
            output_relative_path = output_path.relative_to(output_dir)
        except ValueError as exc:
            raise RuntimeError(
                f"URDFImporter wrote outside configured output directory: {output_path}"
            ) from exc

        generated_files = [
            {
                "path": str(path.relative_to(output_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(output_dir.rglob("*"))
            if path.is_file()
        ]

        manifest = {
            "schema_version": "fragilegrab.asset_import/0.1",
            "tool": "isaacsim.asset.importer.urdf.URDFImporter",
            "isaac_sim_version_components": list(version_parts),
            "source": {"path": str(urdf_path), "sha256": _sha256(urdf_path)},
            "output": {
                "path": str(output_path),
                "path_relative_to_manifest": str(output_relative_path),
                "sha256": _sha256(output_path),
                "generated_files": generated_files,
            },
            "config": {
                "fix_base": args.fix_base,
                "merge_fixed_joints": args.merge_fixed_joints,
                "merge_mesh": args.merge_mesh,
                "collision_from_visuals": args.collision_from_visuals,
                "allow_self_collision": args.allow_self_collision,
                "run_asset_transformer": args.run_asset_transformer,
                "run_multi_physics_conversion": args.run_multi_physics_conversion,
                "ros_package_paths": args.ros_package,
                "joint_drive_type": args.joint_drive_type,
                "joint_target_type": args.joint_target_type,
                "override_joint_stiffness": args.override_joint_stiffness,
                "override_joint_damping": args.override_joint_damping,
            },
        }
        manifest_path = output_dir / "piper_urdf_import_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        print(output_path)
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
