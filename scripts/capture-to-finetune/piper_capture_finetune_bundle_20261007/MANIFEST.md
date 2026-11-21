# Bundle manifest

Created: 2026-10-07

## Included project material

- Complete `capture/` Python sources, excluding bytecode caches.
- Modified OpenPI `src/`, `scripts/`, `packages/`, `examples/`, and documentation.
- OpenPI environment locks and metadata: `pyproject.toml`, `uv.lock`, `.python-version`, licenses, README, and submodule declaration.
- Data-collection design and capture/fine-tuning runbooks.
- Linux and Windows environment notes needed by the workflow.
- Read-only CAN/camera probes, Linux camera udev rules, and the Linux adapter test.

## Key project modifications

- `openpi/src/openpi/policies/piper_policy.py`
- `openpi/src/openpi/training/config.py`
- `capture/aligned_capture.py`
- `capture/device_adapters_linux.py`
- `capture/run_aligned_capture.py`
- `capture/generate_fake_lerobot_dataset.py`

## External material not embedded

- π0.5 base model and fine-tuned checkpoints.
- Real or synthetic LeRobot dataset directories; synthetic data are generated using the bundled script.
- Python wheels and package caches; reproduce them from `openpi/uv.lock` and the pinned versions in the runbook.
- Orbbec SDK binaries. The separately retained Windows archive contains v1.10.37; Linux needs `libOrbbecSDK.so` from the corresponding Linux SDK.
- pyAgxArm source/package, RealSense system rules beyond the bundled udev file, and OS-level CAN drivers.

## Safety boundary

The bundled collection path is read-only with respect to the robot. Motion-control helpers were intentionally excluded. Configure and validate one device at a time before recording.
