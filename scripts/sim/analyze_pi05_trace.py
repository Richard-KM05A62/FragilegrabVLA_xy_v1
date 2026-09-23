#!/usr/bin/env python3
"""离线检查 π0.5 chunk 范围、连续性、gripper 和 chunk boundary。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fragilegrab.benchmark.diagnostics import analyze_pi05_timeline  # noqa: E402
from fragilegrab.benchmark.runner import native  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"拒绝覆盖：{args.output}")
    result = analyze_pi05_timeline(args.episode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(native(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
