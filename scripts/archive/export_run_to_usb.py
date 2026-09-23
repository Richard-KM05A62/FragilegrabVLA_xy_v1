#!/usr/bin/env python3
"""将完整 run 安全迁移到 U 盘，并在本地留下轻量摘要和预览。"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any
import uuid


RECEIPT_NAME = "archived_run.json"
ARCHIVE_SCHEMA = "fragilegrab.archive_receipt/0.1"


class ArchiveError(RuntimeError):
    """安全迁移无法继续。"""


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArchiveError(f"无法读取 JSON：{path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArchiveError(f"JSON 顶层必须是对象：{path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inventory(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ArchiveError(f"run 中不允许符号链接：{path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ArchiveError(f"run 中存在非普通文件：{path}")
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return records


def _tree_sha256(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["size_bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(record["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _episode_summaries(source: Path) -> list[dict[str, Any]]:
    manifests = sorted(source.rglob("episode_manifest.json"))
    if not manifests:
        raise ArchiveError("目录中没有 episode_manifest.json，不是可识别的 run")
    summaries = []
    for manifest_path in manifests:
        manifest = _json(manifest_path)
        termination = manifest.get("termination") or {}
        if termination.get("status") != "complete":
            raise ArchiveError(f"episode 尚未完整完成：{manifest_path}")
        metrics_path = manifest_path.parent / "metrics.json"
        result = _json(metrics_path) if metrics_path.exists() else manifest.get("result")
        task = manifest.get("task") or {}
        summaries.append(
            {
                "relative_directory": manifest_path.parent.relative_to(source).as_posix() or ".",
                "episode_id": manifest.get("episode_id"),
                "profile": manifest.get("profile"),
                "task_id": task.get("id") if isinstance(task, dict) else task,
                "termination": termination,
                "result": result,
                "manifest_integrity": manifest.get("integrity"),
            }
        )
    return summaries


def _mount_for_label(label: str) -> Path:
    try:
        result = subprocess.run(
            ["lsblk", "--json", "--output", "LABEL,MOUNTPOINTS"],
            check=True,
            capture_output=True,
            text=True,
        )
        devices = json.loads(result.stdout).get("blockdevices", [])
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise ArchiveError(f"无法读取 U 盘挂载信息：{exc}") from exc

    matches: list[Path] = []

    def visit(device: dict[str, Any]) -> None:
        if device.get("label") == label:
            for mountpoint in device.get("mountpoints") or []:
                if mountpoint:
                    matches.append(Path(mountpoint))
        for child in device.get("children") or []:
            visit(child)

    for device in devices:
        visit(device)
    unique = sorted({path.resolve() for path in matches})
    if not unique:
        raise ArchiveError(f"未找到已挂载且卷标为 {label!r} 的 U 盘")
    if len(unique) != 1:
        raise ArchiveError(f"卷标 {label!r} 对应多个挂载点：{unique}")
    return unique[0]


def _verify_external_mount(root: Path) -> None:
    try:
        result = subprocess.run(
            ["findmnt", "--noheadings", "--output", "TARGET,SOURCE", "--target", str(root)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ArchiveError(f"无法验证 U 盘挂载点：{exc}") from exc
    fields = result.stdout.strip().split(maxsplit=1)
    if len(fields) != 2 or Path(fields[0]).resolve() != root.resolve():
        raise ArchiveError(f"指定目录不是独立挂载点，拒绝写入：{root}")
    source = fields[1]
    if source in {"overlay", "/dev/root"} or source.startswith("/dev/mapper/"):
        raise ArchiveError(f"指定目录看起来是本机系统盘，拒绝写入：{root} ({source})")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def archive_run(
    source: Path,
    usb_root: Path,
    *,
    usb_label: str | None,
    keep_preview: bool = True,
    verify_mount: bool = True,
) -> dict[str, Any]:
    source = source.resolve()
    usb_root = usb_root.resolve()
    if not source.is_dir():
        raise ArchiveError(f"run 目录不存在：{source}")
    if (source / RECEIPT_NAME).exists():
        raise ArchiveError(f"该目录已经是归档占位目录：{source}")
    if (source / ".incomplete").exists() or (source / "failure.json").exists():
        raise ArchiveError("run 含 .incomplete 或 failure.json，拒绝删除本地证据")
    if verify_mount:
        _verify_external_mount(usb_root)
    if _is_relative_to(source, usb_root) or _is_relative_to(usb_root, source):
        raise ArchiveError("run 源目录与 U 盘目标目录不能互相包含")

    episodes = _episode_summaries(source)
    source_inventory = _inventory(source)
    if not source_inventory:
        raise ArchiveError("run 目录为空")
    total_bytes = sum(record["size_bytes"] for record in source_inventory)
    free_bytes = shutil.disk_usage(usb_root).free
    reserve_bytes = max(64 * 1024 * 1024, total_bytes // 20)
    if free_bytes < total_bytes + reserve_bytes:
        raise ArchiveError(
            f"U 盘空间不足：需要至少 {total_bytes + reserve_bytes} bytes，当前 {free_bytes} bytes"
        )

    archive_parent = usb_root / "FragileGrabVLA" / "runs"
    archive_parent.mkdir(parents=True, exist_ok=True)
    destination = archive_parent / source.name
    if destination.exists():
        raise ArchiveError(f"U 盘目标已经存在，拒绝覆盖：{destination}")
    partial = archive_parent / f".{source.name}.partial-{uuid.uuid4().hex}"

    local_preview_temp = Path(tempfile.mkdtemp(prefix=f".{source.name}.preview-", dir=source.parent))
    preview_records: list[str] = []
    source_delete_started = False
    try:
        if keep_preview:
            for preview in sorted(source.rglob("base_rgb.webm")):
                relative = preview.relative_to(source)
                target = local_preview_temp / "previews" / relative.parent / preview.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(preview, target)
                preview_records.append(target.relative_to(local_preview_temp).as_posix())

        shutil.copytree(source, partial, copy_function=shutil.copy2)
        destination_inventory = _inventory(partial)
        if destination_inventory != source_inventory:
            raise ArchiveError("U 盘副本逐文件 SHA-256 校验失败；本地 run 保持不变")
        os.sync()
        partial.rename(destination)
        os.sync()

        receipt = {
            "schema_version": ARCHIVE_SCHEMA,
            "status": "archived_and_source_removed",
            "archived_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "original_source_path": str(source),
            "destination": {
                "usb_label": usb_label,
                "mountpoint": str(usb_root),
                "path": str(destination),
                "relative_to_mount": destination.relative_to(usb_root).as_posix(),
            },
            "archive": {
                "file_count": len(source_inventory),
                "total_bytes": total_bytes,
                "tree_sha256": _tree_sha256(source_inventory),
            },
            "episodes": episodes,
            "local_preview_files": preview_records,
            "restore_note": "完整 run 只保存在 destination；本地目录是轻量摘要，不可作为完整 episode。",
        }

        # 先把本地轻量占位目录完整准备好，再删除源目录并原子改名。
        # 因而删除后不会经历“摘要尚未写出”的半成品状态。
        _write_json(local_preview_temp / RECEIPT_NAME, receipt)
        source_delete_started = True
        shutil.rmtree(source)
        local_preview_temp.rename(source)
        return receipt
    except BaseException:
        if partial.exists():
            shutil.rmtree(partial)
        # 在尚未开始删除本地源目录时，清掉已经提交的远端副本，允许安全重试。
        # 一旦源目录删除已经开始，则无论后续发生什么都保留已验证的 U 盘副本。
        if destination.exists() and not source_delete_started:
            shutil.rmtree(destination)
        raise
    finally:
        shutil.rmtree(local_preview_temp, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="要迁移的完整 run 目录")
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--usb-label", help="已挂载 U 盘的卷标，例如 FRAGILE_DATA")
    location.add_argument("--usb-root", type=Path, help="U 盘挂载点的绝对路径")
    parser.add_argument(
        "--no-local-preview",
        action="store_true",
        help="本地只保留 archived_run.json，不保留 WebM 预览",
    )
    args = parser.parse_args()
    usb_root = _mount_for_label(args.usb_label) if args.usb_label else args.usb_root
    assert usb_root is not None
    receipt = archive_run(
        args.run,
        usb_root,
        usb_label=args.usb_label,
        keep_preview=not args.no_local_preview,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
