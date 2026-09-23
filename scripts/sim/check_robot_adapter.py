#!/usr/bin/env python3
"""不调用 VLA，逐维验证 PiPER-X MuJoCo action/controller/gripper 映射。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fragilegrab.benchmark.diagnostics import run_robot_adapter_sanity  # noqa: E402
from fragilegrab.benchmark.environment import load_configs  # noqa: E402
from fragilegrab.benchmark.runner import native  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-assets", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs/benchmark")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"拒绝覆盖：{args.output}")
    result = run_robot_adapter_sanity(load_configs(args.config_dir), args.robot_assets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(native(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
