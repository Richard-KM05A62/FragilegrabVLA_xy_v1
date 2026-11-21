# Piper X：数采、OpenPI 微调与 RTC 推理

工作区已按功能整理（2026-09-18）。代码路径调整不改变数据格式、模型权重或控制算法。

```text
piper_x_real/
├── capture/              双相机/CAN 只读读取、时间对齐、LeRobot 写入、假数据生成
├── inference/            RTC 缓存/调度/传输、模型策略、模型服务与运行入口
├── tests/                不连接设备的离线单元测试
├── tools/
│   ├── hardware/         人工硬件检查；部分脚本会使能机械臂
│   └── legacy/           早期采集样例，不是正式 LeRobot 数采入口
├── docs/
│   ├── collection/       数采实现与数据格式
│   ├── training/         数据上传、微调与历史冒烟说明
│   ├── inference/        RTC 设计与实现
│   └── environment/      环境路径、Linux 模型加载交接
├── logs/archive/         既有终端输出及云端目录树备份
├── Log/                  Orbbec SDK 生成的日志（保留原位置）
├── data/                 已有数据集，未移动或修改
├── models/               选定的原 5k checkpoint，未移动或修改
└── openpi/               OpenPI 仓库与 Piper 自定义配置，内部结构保持不变
```

## 启动方式

所有命令从**项目根目录**执行。以前根目录下 `python run_aligned_capture.py` 等命令，现改为 `python -m capture.run_aligned_capture` 等模块方式；不直接执行包内 `.py` 文件。

Windows 数采/离线测试继续使用原统一解释器，不重新安装环境：

```powershell
Set-Location -LiteralPath 'D:\piper_x_real'

# 正式数采：会连接相机和 CAN，但不使能、不控制运动；按 Enter 操作并逐条输入 task prompt。
& 'D:\vscode\opencv\venv\Scripts\python.exe' -m capture.run_aligned_capture `
  --root 'D:\piper_x_real\data\piper_x_recordings' --repo-id local/piper_x_recordings_v000

# 只查看数采参数，不连接设备。
& 'D:\vscode\opencv\venv\Scripts\python.exe' -m capture.run_aligned_capture --help

# 离线测试与模拟 RTC：不加载真实模型，不连接设备，不下发指令。
& 'D:\vscode\opencv\venv\Scripts\python.exe' -m unittest discover -s tests -v
& 'D:\vscode\opencv\venv\Scripts\python.exe' -m inference.run_piper_rtc --demo --duration-s 5
```

假数据生成入口为 `python -m capture.generate_fake_lerobot_dataset`；先用 `--help` 检查输出目录，不覆盖正式数据。早期单相机入口为 `python -m capture.collect_piper_lerobot`，不是当前双相机正式入口。

Linux 模型加载/服务使用 `openpi/.venv/bin/python`，不是 Windows 数采解释器。请先按下面的环境交接配置并验证 GPU，之后在项目根目录运行：

```bash
openpi/.venv/bin/python -m inference.serve_piper_rtc \
  --config pi05_piper_finetune --checkpoint models \
  --host 127.0.0.1 --port 8000 --num-steps 10 --max-guidance 5
```

模型加载与 Linux GPU/RTC 编译仍需实际验证。`inference.run_piper_rtc` 目前只有模拟/只读空执行接口，不会让机械臂运动；Linux 相机/CAN 适配也尚未完成。

## 文档入口

- [数采实现与数据格式](docs/collection/DATA_COLLECTION_DESIGN.md)
- [从真实数据到微调](docs/training/REAL_DATA_TO_FINETUNE.md)
- [数采与云端微调 Runbook](docs/training/CAPTURE_FINETUNE_RUNBOOK.md)
- [历史假数据微调说明](docs/training/OPENPI_PIPER_FINETUNE.md)
- [RTC H40 / s20 / 8-step 重新微调计划](docs/training/RTC_H40_FINETUNE_PLAN.md)
- [H40 / s20 / 8-step 推理端修改清单](docs/inference/H40_INFERENCE_CHANGES.md)
- [RTC 详细设计](docs/inference/RTC_INFERENCE_DESIGN.md)
- [RTC 已实现功能与验证边界](docs/inference/RTC_IMPLEMENTATION.md)
- [环境与路径记录](docs/environment/ENVIRONMENT_AND_PATHS.md)
- [新 agent 的 Linux 环境/5k 模型加载测试说明](docs/environment/LINUX_MODEL_LOAD_TEST.md)
- [Linux 真机设备适配清单](docs/environment/LINUX_DEVICE_ADAPTATION.md)

人工硬件脚本见 [tools/README.md](tools/README.md)。未经用户确认不要运行使能或主从示教脚本；也不要通过导入它们探测接口，因为旧脚本在模块顶层就连接/使能设备。
