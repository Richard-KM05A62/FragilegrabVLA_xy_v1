# Piper X data collection and OpenPI fine-tuning bundle

This source bundle contains the project-specific code and documentation needed to reproduce the Piper X data-collection and π0.5 fine-tuning workflow.

Start with `docs/training/CAPTURE_FINETUNE_RUNBOOK.md`. It covers Windows collection, cloud setup, synthetic-data smoke testing, real-data configuration, normalization statistics, training, and the remaining Linux collection-entry migration work.

## Scope

- `capture/`: aligned D435i + DaBai DC1 + Piper CAN collection and fake LeRobot dataset generation.
- `openpi/`: the modified OpenPI source tree required by the Piper dataset and fine-tuning configuration.
- `docs/`: collection contract, environment records, and fine-tuning runbooks.
- `tools/hardware/`: read-only device probes and Linux udev rules.
- `tests/`: Linux device-adapter unit test.

The Linux device adapters have passed hardware tests, but `capture/run_aligned_capture.py` still needs to select those adapters before the formal Linux LeRobot recorder is considered end-to-end ready. See runbook section 13.

## Deliberately excluded

Real datasets, model weights, checkpoints, logs, virtual environments, Git history, caches, RTC inference code, robot-control scripts, camera Viewer applications, and third-party ALOHA/LIBERO simulation assets are not included.

The Windows Orbbec SDK is distributed as a separate optional archive. Linux requires the matching Linux Orbbec SDK and cannot use the Windows DLL archive.

## Integrity

`SHA256SUMS` records every file in this bundle except itself. Verify it before configuring or running hardware.
