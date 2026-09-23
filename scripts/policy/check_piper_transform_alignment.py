#!/usr/bin/env python3
"""离线逐轴审计 π0.5–PiPER normalization 与 robot-specific transform。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fragilegrab.benchmark.diagnostics import audit_piper_transform_alignment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--norm-stats-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"拒绝覆盖：{args.output}")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))["values"]
    stats_doc = json.loads((args.norm_stats_dir / "norm_stats.json").read_text(encoding="utf-8"))
    provenance = json.loads((args.norm_stats_dir / "provenance.json").read_text(encoding="utf-8"))
    if config["norm_stats_id"] != provenance["norm_stats_id"]:
        raise ValueError("config 与 provenance 的 norm_stats_id 不一致")
    if config["contract_id"] != provenance["contract_id"]:
        raise ValueError("config 与 provenance 的 contract_id 不一致")

    result = audit_piper_transform_alignment(
        state=config["state"],
        norm_stats=stats_doc["norm_stats"],
        norm_stats_id=provenance["norm_stats_id"],
        normalization_purpose=provenance["purpose"],
        joint_probe_rad=float(config["joint_probe_rad"]),
        gripper_closures=config["gripper_closures"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if result["status"] == "failed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
