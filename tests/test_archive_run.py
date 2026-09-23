from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.archive.export_run_to_usb import ArchiveError, archive_run


def _complete_run(root: Path) -> Path:
    run = root / "run_a"
    episode = run / "0"
    frames = episode / "streams" / "base_rgb"
    frames.mkdir(parents=True)
    (frames / "000000.npy").write_bytes(b"large raw frame" * 100)
    (episode / "timeline.jsonl").write_text('{"record_type":"observation"}\n')
    (episode / "base_rgb.webm").write_bytes(b"small preview")
    (episode / "metrics.json").write_text(
        json.dumps({"task_success": True, "peak_contact_force_N": 1.2})
    )
    (episode / "episode_manifest.json").write_text(
        json.dumps(
            {
                "episode_id": "episode-a",
                "profile": "mujoco_qualification/0.1",
                "task": {"id": "F1"},
                "termination": {"status": "complete", "reason": "script_horizon"},
                "integrity": {"steps": 10, "frames": 1},
            }
        )
    )
    (run / "smoke_result.json").write_text(json.dumps({"pass": True}))
    return run


def test_archive_verifies_then_replaces_source_with_summary_and_preview(tmp_path: Path) -> None:
    source = _complete_run(tmp_path / "local")
    usb = tmp_path / "usb"
    usb.mkdir()

    receipt = archive_run(
        source,
        usb,
        usb_label="TEST_USB",
        keep_preview=True,
        verify_mount=False,
    )

    destination = usb / "FragileGrabVLA" / "runs" / "run_a"
    assert (destination / "0" / "streams" / "base_rgb" / "000000.npy").exists()
    assert (destination / "0" / "timeline.jsonl").exists()
    assert receipt["archive"]["file_count"] == 6
    assert receipt["episodes"][0]["result"]["task_success"] is True

    assert sorted(path.relative_to(source).as_posix() for path in source.rglob("*") if path.is_file()) == [
        "archived_run.json",
        "previews/0/base_rgb.webm",
    ]
    local_receipt = json.loads((source / "archived_run.json").read_text())
    assert local_receipt["status"] == "archived_and_source_removed"
    assert local_receipt["destination"]["usb_label"] == "TEST_USB"
    assert not (source / "0" / "streams").exists()


def test_incomplete_run_is_not_copied_or_deleted(tmp_path: Path) -> None:
    source = _complete_run(tmp_path / "local")
    (source / ".incomplete").touch()
    usb = tmp_path / "usb"
    usb.mkdir()

    with pytest.raises(ArchiveError, match="incomplete"):
        archive_run(source, usb, usb_label="TEST_USB", verify_mount=False)

    assert (source / "0" / "streams" / "base_rgb" / "000000.npy").exists()
    assert not (usb / "FragileGrabVLA").exists()


def test_existing_destination_is_never_overwritten(tmp_path: Path) -> None:
    source = _complete_run(tmp_path / "local")
    destination = tmp_path / "usb" / "FragileGrabVLA" / "runs" / "run_a"
    destination.mkdir(parents=True)
    marker = destination / "keep.txt"
    marker.write_text("existing")

    with pytest.raises(ArchiveError, match="已经存在"):
        archive_run(source, tmp_path / "usb", usb_label="TEST_USB", verify_mount=False)

    assert marker.read_text() == "existing"
    assert (source / "0" / "timeline.jsonl").exists()


def test_archived_stub_cannot_be_archived_again(tmp_path: Path) -> None:
    source = tmp_path / "local" / "run_a"
    source.mkdir(parents=True)
    (source / "archived_run.json").write_text("{}")
    usb = tmp_path / "usb"
    usb.mkdir()

    with pytest.raises(ArchiveError, match="归档占位"):
        archive_run(source, usb, usb_label="TEST_USB", verify_mount=False)
