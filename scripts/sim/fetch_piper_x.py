#!/usr/bin/env python3
"""Fetch the pinned official AgileX PiPER-X + D435 simulation assets."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import yaml


ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=ROOT / "assets/vendor/agx_arm_sim")
parser.add_argument("--robot-config", type=Path, default=ROOT / "configs/benchmark/robot.yaml")
args = parser.parse_args()
robot = yaml.safe_load(args.robot_config.read_text(encoding="utf-8"))["values"]
args.output.mkdir(parents=True, exist_ok=False)


def git(*values: str) -> None:
    subprocess.run(["git", "-C", str(args.output), *values], check=True)


git("init")
git("remote", "add", "origin", robot["repository"])
git("fetch", "--depth=1", "--filter=blob:none", "origin", robot["commit"])
git("checkout", "--detach", "FETCH_HEAD")
git("submodule", "update", "--init", "--depth=1", robot["urdf_submodule"])
actual = subprocess.check_output(
    ["git", "-C", str(args.output / robot["urdf_submodule"]), "rev-parse", "HEAD"],
    text=True,
).strip()
if actual != robot["urdf_submodule_commit"]:
    raise RuntimeError(
        f"agx_arm_urdf submodule mismatch: expected {robot['urdf_submodule_commit']}, got {actual}"
    )
print(f"Fetched official PiPER-X + D435 assets at {robot['commit']} to {args.output}")
