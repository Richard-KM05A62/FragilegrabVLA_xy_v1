"""从未压缩 RGB 帧生成 VS Code 可预览的 VP8/WebM 派生产物。"""
from pathlib import Path
import hashlib
import json

import cv2
import numpy as np


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_webm(episode_dir, expected_frames, resolution_hw, fps, fourcc="VP80"):
    """不修改原始 MP4、RGB 或 episode manifest。已有预览不静默覆盖。"""
    episode_dir = Path(episode_dir)
    if fourcc != "VP80":
        raise ValueError("VS Code 预览仅使用 VP8（VP80）编码")
    if expected_frames <= 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("帧数和帧率必须为正")
    h, w = resolution_hw
    sources = sorted((episode_dir / "streams/base_rgb").glob("*.npy"))
    if len(sources) != expected_frames or any(p.name != f"{i:06d}.npy" for i, p in enumerate(sources)):
        raise ValueError("原始 RGB 帧数量或序号不符，拒绝生成预览")
    target = episode_dir / "base_rgb.webm"
    if target.exists():
        raise FileExistsError(f"预览已存在，不覆盖：{target}")
    partial = episode_dir / "base_rgb.partial.webm"
    if partial.exists():
        raise FileExistsError(f"上次预览生成未完成，请检查后移走：{partial}")
    writer = cv2.VideoWriter(str(partial), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("此 OpenCV 构建没有 VP8/WebM 编码器")
    try:
        for path in sources:
            rgb = np.load(path, allow_pickle=False)
            if rgb.dtype != np.uint8 or rgb.shape != (h, w, 3):
                raise ValueError(f"原始 RGB 帧格式不符：{path}")
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    except BaseException:
        writer.release()
        partial.unlink(missing_ok=True)
        raise
    writer.release()
    capture = cv2.VideoCapture(str(partial))
    decoded = 0
    codec_code = int(capture.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((codec_code >> (i * 8)) & 255) for i in range(4))
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame.shape[:2] != (h, w):
            capture.release()
            partial.unlink(missing_ok=True)
            raise RuntimeError("预览视频解码尺寸不符")
        decoded += 1
    capture.release()
    if decoded != expected_frames or codec != fourcc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"预览视频解码失败：{decoded}/{expected_frames} 帧，codec={codec}")
    partial.rename(target)
    metadata = {"source": "streams/base_rgb/*.npy", "video": target.name,
                "codec": "VP8", "container": "WebM", "frames": decoded,
                "fps": fps, "resolution_hw": [h, w], "sha256": _sha256(target),
                "role": "显示预览；原始物理与指标证据仍见 episode_manifest.json"}
    (episode_dir / "preview_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata
