#!/usr/bin/env python3
"""为已有 MuJoCo episode 的原始 RGB 帧生成 VS Code 可播放的 WebM。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from fragilegrab.benchmark.video import export_webm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_dir", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.episode_dir / "episode_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("profile") != "mujoco_qualification/0.1" or manifest.get("termination", {}).get("status") != "complete":
        raise ValueError("只接受已完成的 MuJoCo Qualification/F1 episode")
    rgb = manifest["streams"]["base_rgb"]
    metadata = export_webm(args.episode_dir, manifest["integrity"]["frames"],
                           rgb["shape"][:2], rgb["fps"])
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
