# Piper X 数采与 OpenPI π0.5 微调 Runbook

本文面向尚未配置环境的操作者，目标是从 Windows 真机数采开始，完成云端假数据冒烟和正式 π0.5 LoRA 微调。所有示例都使用本项目当前的数据契约与修改版 OpenPI。

## 1. 固定契约

不要在流程中临时改变以下定义：

| 项目 | 约定 |
|---|---|
| 数据格式 | LeRobot v2.1，30 Hz |
| 底座相机 | Intel RealSense D435i，`observation.images.base` |
| 腕部相机 | Orbbec DaBai DC1，`observation.images.wrist` |
| state | `float32[7] = [q1..q6(rad), gripper_closedness]` |
| 原始 action | 下一逻辑时刻的 state 代理，同为绝对值 `float32[7]` |
| 夹爪 | 0 全开、1 全闭；训练和推理均保持绝对闭合度 |
| 关节动作 | OpenPI 训练前转换为 `action_q - current_q`，推理后恢复绝对关节目标 |
| 图像预处理 | policy 内等比例补边到 `224×224`，不做中心裁剪 |
| prompt | 每条轨迹必须输入非空任务文本 |

本项目的正式采集入口只读取相机和 CAN 状态，不使能机械臂、不发送运动或夹爪指令。

## 2. Windows 数采环境

### 2.1 硬件与驱动

先安装并验证：

1. AgileX CAN 驱动和 USB-CAN 设备；当前代码使用 `channel="0"`、`interface="agx_cando"`。
2. Intel RealSense 驱动；用 RealSense Viewer 确认 D435i 能输出彩色图像。
3. Orbbec SDK C/C++ v1.10.37；确认 DaBai DC1 能被官方工具识别。

当前 Windows 适配器默认在下面的位置寻找 Orbbec SDK：

```text
C:\Users\13302\Downloads\OrbbecSDK_C_C++_v1.10.37_20260707_3f75820b8_win_x64_release\OrbbecSDK_v1.10.37
```

其他用户必须把 `capture/aligned_capture.py` 中的 `DEFAULT_ORBBEC_SDK_ROOT` 改为自己的实际 SDK 根目录；该目录下应存在 `SDK/lib/OrbbecSDK.dll`。

### 2.2 创建干净的 Python 环境

已验证的数采环境是 Python 3.12.9。不要复用包含 Isaac Sim 或其他大型软件栈的环境。

```powershell
Set-Location -LiteralPath 'D:\piper_x_real'
py -3.12 -m venv .venv-capture
& '.\.venv-capture\Scripts\python.exe' -m pip install --upgrade pip

& '.\.venv-capture\Scripts\python.exe' -m pip install `
  'datasets==3.6.0' `
  'lerobot==0.1.0' `
  'pyrealsense2==2.58.4.10922' `
  'av==18.1.0'

& '.\.venv-capture\Scripts\python.exe' -m pip install `
  'git+https://github.com/agilexrobotics/pyAgxArm.git@cc498c00af0bcb9e297943e94f4792c0e3ee5b2c'

& '.\.venv-capture\Scripts\python.exe' -m pip install `
  'git+https://github.com/agilexrobotics/python-can-agx-cando.git@b222c4027ad4f6599f7634c72c67184619177972'
```

验证关键依赖：

```powershell
& '.\.venv-capture\Scripts\python.exe' -c @'
import datasets, lerobot, pyrealsense2, av, pyAgxArm
print("datasets", datasets.__version__)
print("lerobot", lerobot.__version__)
print("capture imports OK")
'@
```

必须看到 `datasets 3.6.0` 和 `lerobot 0.1.0`。数据写入端与云端的 Datasets 版本不一致时，Parquet 元数据可能无法读取。

现有工作站也可继续使用已验证解释器：

```text
D:\vscode\opencv\venv\Scripts\python.exe
```

但新环境应优先独立创建，不复制旧虚拟环境目录。

## 3. 先做短轨迹数采

从项目根目录查看参数不会连接设备：

```powershell
& '.\.venv-capture\Scripts\python.exe' -m capture.run_aligned_capture --help
```

正式启动：

```powershell
& '.\.venv-capture\Scripts\python.exe' -m capture.run_aligned_capture `
  --root 'D:\piper_x_real\data\real\pick_cube_v001' `
  --repo-id 'local/piper_x_pick_cube_v001'
```

多台同型号相机时增加：

```text
--d435i-serial <D435i序列号> --dc1-serial <DC1序列号>
```

交互顺序：

1. 程序报告已有轨迹数和下一编号。
2. 按 Enter，输入本条轨迹的 task prompt。
3. 程序等待三路数据稳定，并从 Enter 后约 100 ms 开始记录。
4. 再按 Enter，保存本条轨迹。
5. 重复以上步骤；输入 `q` 退出。

同一数据集始终复用相同的 `--root` 和 `--repo-id`。采集器会自动从下一 episode 编号续写；不要手工拼接 Parquet 或视频文件。

首条短轨迹必须检查：

- 无 producer failure，帧数约为时长 × 30 Hz；
- 两路视频角色、RGB 颜色、视野和安装角度正确；
- `piper_capture_manifest.json` 为 30 Hz、host clock、±30 ms；
- `piper_raw/episode_*.jsonl` 三路 skew 均在 ±30 ms；
- 六关节单位为 rad，夹爪值位于 `[0,1]`；
- task prompt 与演示动作一致。

## 4. 生成假数据

假数据只验证读取、归一化、权重加载、反向传播和 checkpoint 保存，不能用于训练真实策略。目标目录必须尚不存在。

```powershell
& '.\.venv-capture\Scripts\python.exe' -m capture.generate_fake_lerobot_dataset `
  --root 'D:\piper_x_real\data\debug\piper_x_fake_zero' `
  --repo-id 'local/piper_x_fake_zero' `
  --task 'debug no-op task' `
  --episodes 2 `
  --frames-per-episode 64
```

预期输出为 2 episodes / 128 frames，并包含 `meta/`、`data/`、`videos/`、`piper_capture_manifest.json`。

## 5. 云端 OpenPI 环境

### 5.1 源码与 GPU

使用本工作区的修改版 `openpi/`，不要只 clone 原始上游仓库。基线 commit 为：

```text
15a9616a00943ada6c20a0f158e3adb39df2ccac
```

必须保留：

```text
src/openpi/policies/piper_policy.py
src/openpi/training/config.py
src/openpi/training/data_loader.py
pyproject.toml
uv.lock
```

建议云端路径：

```text
/root/autodl-tmp/openpi
```

先确认 `nvidia-smi` 能看到 GPU；不要用 CPU 回退冒充环境成功。

### 5.2 安装系统与 Python 依赖

```bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg

# 已有 uv 时跳过安装。
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"

cd /root/autodl-tmp/openpi
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 UV_LINK_MODE=copy uv sync --frozen --python 3.11
```

以后始终使用项目解释器，不使用裸 `python`：

```text
/root/autodl-tmp/openpi/.venv/bin/python
```

验证：

```bash
cd /root/autodl-tmp/openpi
ffmpeg -version | head -n 1
.venv/bin/python - <<'PY'
import datasets, jax, torch, torchcodec
print("datasets", datasets.__version__)
print("jax", jax.__version__)
print("devices", jax.devices())
assert datasets.__version__ == "3.6.0"
assert jax.default_backend() == "gpu"
print("cloud environment OK")
PY
```

### 5.3 放置基础权重

将官方 π0.5 基础权重放到：

```text
/root/autodl-tmp/pi05_checkpoints/openpi-assets/checkpoints/pi05_base/params
```

`params/` 必须是实际 Orbax 参数目录，不能多嵌套一层 `pi05_base/params/params`。该权重不由 `uv sync` 下载。

## 6. 上传并验证假数据

LeRobot 的云端缓存路径由 `repo_id` 决定。假数据应放为：

```text
/root/autodl-tmp/.cache/huggingface/lerobot/local/piper_x_fake_zero/
```

其下必须直接出现 `meta/`、`data/` 和 `videos/`，不能多套一层本机目录名。可使用 SCP、SFTP 或云平台文件管理器上传。

修改版 OpenPI 已包含 `pi05_piper_debug_finetune`。执行：

```bash
cd /root/autodl-tmp/openpi
.venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi05_piper_debug_finetune

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
.venv/bin/python scripts/train.py pi05_piper_debug_finetune \
  --exp-name piper_fake_smoke
```

通过标准：

- norm stats 正常写出，无数据 schema、视频解码或 Parquet 错误；
- 20 个训练 step 全部完成，loss 和 grad norm 为有限值；
- checkpoint 中存在 `params/`、`assets/`、`train_state/`；
- GPU 有实际占用，无 CPU 回退或 OOM。

`--overwrite` 会删除同名实验目录，仅在明确要重跑同一个假实验时使用。正式训练优先更换 `exp-name`，不要习惯性覆盖。

## 7. 上传正式数据

假数据冒烟成功后，将正式数据上传到：

```text
/root/autodl-tmp/.cache/huggingface/lerobot/<repo_id>/
```

例如 `repo_id=local/piper_x_pick_cube_v001`：

```text
/root/autodl-tmp/.cache/huggingface/lerobot/local/piper_x_pick_cube_v001/
```

上传后检查 `meta/info.json`、Parquet 和两路视频均存在。不要上传本机用于安全续写的隐藏派生缓存目录。

## 8. 为新数据集添加训练配置

在 `openpi/src/openpi/training/config.py` 的 `_CONFIGS` 中复制一个 Piper 正式配置，并同时修改模型和 freeze filter。以下是 H=40 的模板：

```python
TrainConfig(
    name="pi05_piper_pick_cube_v001_h40",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=40,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ),
    data=LeRobotPiperDataConfig(
        repo_id="local/piper_x_pick_cube_v001",
        base_config=DataConfig(
            prompt_from_task=True,
            exclude_episodes=(),  # 填入确认损坏的零基 episode ID
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "/root/autodl-tmp/pi05_checkpoints/openpi-assets/checkpoints/pi05_base/params"
    ),
    batch_size=8,
    num_train_steps=5_000,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=100,
        peak_lr=6.25e-6,
        decay_steps=5_000,
        decay_lr=6.25e-7,
    ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    ema_decay=None,
    freeze_filter=pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=40,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter(),
    save_interval=500,
    keep_period=None,
    log_interval=100,
    wandb_enabled=False,
),
```

配置原则：

- `name` 和 `repo_id` 必须唯一且准确；
- `model` 与 `freeze_filter` 的结构参数必须完全一致；
- H=40 用于当前 RTC 方案；更改 horizon 后必须重新训练并重算 stats；
- 只在确认轨迹损坏后填写 `exclude_episodes`，编号从 0 开始；
- 数据加载器显示发现全部 data files 不代表坏 episode 被训练使用，最终以过滤后的 frame 数为准。

## 9. 计算正式统计量

每个新配置都要单独计算，不能复用假数据、旧数据集或不同 horizon 的统计量：

```bash
cd /root/autodl-tmp/openpi
.venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi05_piper_pick_cube_v001_h40
```

输出位置：

```text
assets/pi05_piper_pick_cube_v001_h40/local/piper_x_pick_cube_v001/norm_stats.json
```

检查 `state` 和 `actions` 的 `mean/std/q01/q99` 均为 7 维有限数，且 `std > 0`。以下任一项改变后都应重算：数据集、排除轨迹、action horizon、动作变换或字段定义。

## 10. 指定微调参数

参数写在 `TrainConfig` 中；命令行主要选择配置和实验名。

| 参数 | 建议起点 | 说明 |
|---|---:|---|
| `action_horizon` | 40 | 必须与部署端 horizon 一致 |
| `batch_size` | 8 | H=40 在 RTX 4090 24 GB 上实测约 22.1 GB；OOM 时降为 4 |
| `num_train_steps` | 5,000 | 本项目 18 条/3,717 帧数据的已验证设置 |
| `warmup_steps` | 100 | LoRA 学习率预热 |
| `peak_lr` | `6.25e-6` | 已验证的 LoRA 峰值学习率 |
| `decay_steps` | 与总步数一致 | 余弦衰减覆盖本轮训练 |
| `decay_lr` | `6.25e-7` | 最低学习率 |
| `log_interval` | 100 | 每 100 step 打印一次指标 |
| `save_interval` | 500 | 每 500 step 保存恢复点 |
| `keep_period` | `None` | 配合 checkpoint manager 只保留最新一个 |

不同数据规模可用 `num_train_steps × batch_size / 有效帧数` 估算样本遍历次数；先以约 5–15 次遍历作为试验范围，再依据真机验证和过拟合迹象调整。训练 loss 只能判断优化是否正常，不能替代任务成功率评估。

batch 从 8 降到 4 会使每 step 样本数减半；若希望保持相近样本曝光量，应相应增加 steps，并重新审视学习率衰减长度，而不是只改 batch。

## 11. 启动正式微调

先确认同名实验目录不存在。为了让 SSH 退出后训练继续，使用后台日志：

```bash
cd /root/autodl-tmp/openpi

CONFIG=pi05_piper_pick_cube_v001_h40
EXP=piper_pick_cube_v001_h40_5k
LOGDIR="logs/$EXP"
mkdir -p "$LOGDIR"

nohup env \
  PYTHONUNBUFFERED=1 \
  JAX_PLATFORMS=cuda \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  .venv/bin/python scripts/train.py "$CONFIG" --exp-name "$EXP" \
  > "$LOGDIR/train.log" 2>&1 < /dev/null &

echo $! > "$LOGDIR/train.pid"
```

监控：

```bash
tail -f "logs/$EXP/train.log"
ps -p "$(cat "logs/$EXP/train.pid")" -o pid,etime,stat,%cpu,%mem,cmd
nvidia-smi
```

checkpoint 位置：

```text
checkpoints/<CONFIG>/<EXP>/<step>/
```

训练循环会在最后一步强制保存；5,000 steps 的最终目录是 `4999/`。最终至少应包含：

```text
4999/
  _CHECKPOINT_METADATA
  params/
  assets/
  train_state/
```

推理部署只需 `params/`、`assets/` 和 `_CHECKPOINT_METADATA`；续训才需要 `train_state/`。下载后应核对文件数量、总大小、空文件和 `assets/.../norm_stats.json` 哈希。

## 12. 最短验收清单

按顺序完成，前一步失败不要继续：

1. Windows 两台相机与 CAN 分别可读。
2. 一条短轨迹可保存、回放，state/action/skew 合法。
3. 本机假数据为 2 episodes / 128 frames。
4. 云端 `uv` 环境、FFmpeg、GPU 和基础权重可用。
5. 假数据 norm stats 与 20-step 训练通过。
6. 正式数据上传路径和 `repo_id` 一致。
7. 新配置的 horizon、freeze filter、排除列表和权重路径正确。
8. 正式 norm stats 为 7 维有限数。
9. 正式训练首个 step、周期 loss、显存和滚动 checkpoint 正常。
10. 最终 checkpoint 保存完整后再进入推理验收。

## 13. 将正式数采迁移到 Linux

本节记录截至 2026-10-07 的准确状态。Linux 设备读取层已经完成，不应把“设备适配已验证”和“正式 LeRobot 数采入口已接通”混为一件事。

### 13.1 已经完成并通过真机验证

Linux 专用实现位于 `capture/device_adapters_linux.py`，已经验证：

- D435i 通过 Linux `pyrealsense2 2.58.4.10922` 连续读取、停止和重连；
- DaBai DC1 通过 Orbbec Linux SDK v1.10.37 的 `libOrbbecSDK.so` 连续读取、停止和重连；
- 两台相机可以同时输出原生分辨率的 RGB `uint8[H,W,3]`；
- Piper 通过 `socketcan`、`can0` 和 pyAgxArm 读取 6 关节及夹爪反馈；
- 三路数据可以同时进入时间缓存，使用主机 `time.monotonic_ns()` 时间戳；
- 相机 udev 规则已经安装，CAN、双相机和三路联合读取均已在真机通过。

这部分不需要改变原有 30 Hz 逻辑时钟、2 秒时间窗口、±30 ms 最近邻选择、`T0 = common_start + 100 ms`、state/action 定义或 episode/task 规则。

### 13.2 尚未完成的入口接线

正式保存 LeRobot 数据集的入口 `capture/run_aligned_capture.py` 当前仍创建 Windows 类：

```text
RealSenseD435iAdapter
OrbbecDaBaiDC1Adapter
PiperCanAdapter
```

因此该入口目前不能原样用于 Linux 正式数采。需要让它在 Linux 上改为创建：

```text
RealSenseD435iAdapterLinux
OrbbecDaBaiDC1AdapterLinux
PiperCanAdapterLinux
```

推荐保留一个入口并按 `sys.platform` 选择适配器，同时增加以下可选参数；也可以单独新增 Linux 入口，但不能复制时间对齐和数据写入逻辑：

```text
--orbbec-sdk-root
--can-channel       # 默认 can0
--can-interface     # 默认 socketcan
```

改完后必须保存一条短轨迹，验证 `meta/`、`data/`、`videos/`、`piper_raw/` 和 `piper_capture_manifest.json` 均正常生成，随后再做长时间录制。

### 13.3 Linux 数采解释器要求

设备读取和 LeRobot 写盘必须发生在同一个 Python 进程，因此运行正式数采的解释器必须同时包含：

- `pyAgxArm 1.0.0`（当前验证的官方 commit 为 `cc498c00`）；
- `python-can 4.6.1`；
- `pyrealsense2 2.58.4.10922`；
- `numpy 1.26.4`；
- 与本项目数据格式兼容的 LeRobot 版本；
- `datasets`、PyAV (`av`) 和系统 FFmpeg。

只包含相机/CAN 包的设备环境不能写完整 LeRobot 视频数据集；只包含 OpenPI/GPU 依赖的模型环境也不能默认视为具备设备访问能力。可以在既有设备环境中补齐 LeRobot 写盘依赖，但必须先确认不会改变上述已验证版本。OpenPI 推理服务仍可保留独立 GPU 环境，与数采不是同一进程。

### 13.4 SDK、权限和启动前配置

Windows 的 Orbbec ZIP、`OrbbecSDK.dll` 和 Windows Viewer 不能用于 Linux。当前已验证的 Linux SDK 根目录为：

```text
/home/bac-alive/.local/share/piper-x-device/orbbec/native-v1.10.37/OrbbecSDK_v1.10.37
```

采集前配置动态库和 CAN：

```bash
sudo ip link set can0 up type can bitrate 1000000

export ORBBEC_SDK_ROOT=/home/bac-alive/.local/share/piper-x-device/orbbec/native-v1.10.37/OrbbecSDK_v1.10.37
export LD_LIBRARY_PATH="$ORBBEC_SDK_ROOT/SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PIPER_CAN_CHANNEL=can0
export PIPER_CAN_INTERFACE=socketcan
```

若 `can0` 已经启动，重复执行 `ip link set ... up` 可能返回 busy；此时先用 `ip -details link show can0` 确认它确实为 1,000,000 bit/s 和 `ERROR-ACTIVE`，无需为了消除提示而重置正在工作的接口。

### 13.5 序列号与运行注意事项

- D435i 已验证序列号为 `261822072150`，可以传给 `--d435i-serial`；
- 当前 Orbbec Linux SDK 对这台 DC1 返回空 SDK 序列号。只连接一台 Orbbec 相机时必须省略 `--dc1-serial`，USB 描述符 `CC1N16201XW` 不能作为 SDK 选择参数；
- 保持 DC1 的 RGB buffer 独立复制，以及 color frame/frameset 的完整释放；这是避免长期录制耗尽 SDK frame buffer 的必要条件；
- 默认视频模式依赖 PyAV/FFmpeg。只有在排查编码问题时才用 `--images`，因为正式训练数据的存储模式应保持一致；
- Linux 迁移不改变数据契约：外部相机仍为 `observation.images.base`，腕部 DC1 仍为 `observation.images.wrist`，state/action 均为 7 维，图像采集端不裁剪、不缩放。

### 13.6 Linux 端到端验收顺序

1. 在正式数采解释器中导入 `pyrealsense2`、`pyAgxArm`、`lerobot`、`datasets` 和 `av`。
2. 分别复测 D435i、DC1 和 CAN，只读且不使能、不发送动作。
3. 运行双相机并发和三路缓存测试，确认退出后线程与相机资源释放。
4. 完成 `run_aligned_capture.py` 的 Linux 适配器接线。
5. 录制数秒短轨迹，检查帧数、RGB 通道、7 维 state/action、时间偏差和 task。
6. 关闭并重新启动进程，确认下一条 episode 可以在同一数据集内顺序追加。
7. 用与云端相同的数据读取配置加载该短数据集，计算 norm stats 或执行小规模假训练，确认没有跨平台编码差异。

