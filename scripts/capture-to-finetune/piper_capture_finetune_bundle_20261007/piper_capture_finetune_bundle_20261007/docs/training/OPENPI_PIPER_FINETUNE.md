# Piper X → OpenPI pi0.5 调试微调

本文件描述 `pi05_piper_debug_finetune` 的数据契约和云端冒烟流程。该配置定义在 OpenPI 的
`src/openpi/training/config.py`，配套策略转换在 `src/openpi/policies/piper_policy.py`。

## 数据契约

- LeRobot v2.1 特征：`observation.images.base`（D435i）、`observation.images.wrist`（DaBai DC1）、`observation.state`、`action`。
- 图像：RGB `uint8[H,W,3]`；OpenPI 的模型变换会使用等比例补边（`resize_with_pad`）缩放为 `224×224`。不做中心裁剪。真机推理必须传入未裁剪的原始 RGB 图像，由同一转换完成缩放。
- 本体状态：`float32[7] = [q1..q6, gripper_closedness]`；六个关节为弧度，夹爪为闭合度，0 是全开、1 是全闭。
- 原始动作：同样是 `float32[7]` 的绝对目标。本项目数采的动作是下一逻辑时刻的从臂状态代理。
- 训练输入：动作块的前六维由 `absolute_q - current_q` 转为关节增量；第七维夹爪始终保持绝对闭合度。推理输出在返回 Piper harness 前执行逆变换，重新得到 `[q1..q6, gripper_closedness]`。
- pi0.5 内部保留 32 维动作头；7 维 Piper state/action 在模型输入端补零，输出端只取前 7 维。

## 把本地假数据放到云端

当前调试数据的 `repo_id` 是 `local/piper_x_fake_zero`。OpenPI 创建 LeRobot 数据集时不传入 `root`，因此云端目录必须符合 LeRobot 的缓存规则：

```text
$HF_LEROBOT_HOME/local/piper_x_fake_zero/
  meta/
  data/
  videos/
  piper_capture_manifest.json
```

最直接的做法是在云端设置 `HF_LEROBOT_HOME`，并将本机的
`D:\piper_x_real\data\fake_zero_lerobot_v001` 整个目录上传/复制到上面的目标目录。也可以将数据上传为同名 Hugging Face 数据集；此时让 LeRobot 下载即可。

**版本约束：** 数据集写入端与云端必须使用兼容的 `datasets` 版本。实际验证发现，
本机的 `datasets 5.0.1` 写出的 `List` Parquet 元数据不能由云端 OpenPI/LeRobot 的
`datasets 3.6.0` 读取。正式数采前应将本机数采环境固定到 `datasets==3.6.0`，或同步升级
并验证云端全部 LeRobot 依赖；当前采用前者。云端还必须安装系统级 FFmpeg，以供 TorchCodec
解码 LeRobot 的 AV1 视频。

## 云端首次冒烟命令

在包含 Piper 配置代码的 OpenPI 根目录运行：

```bash
cd /root/autodl-tmp/openpi
.venv/bin/python scripts/compute_norm_stats.py --config-name pi05_piper_debug_finetune
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 .venv/bin/python scripts/train.py pi05_piper_debug_finetune --exp-name piper_fake_smoke --overwrite
```

配置默认只训练 20 步、不写 Weights & Biases，用来验证数据加载、归一化、权重加载、前向/反向与 checkpoint 保存。全零假数据允许统计量计算；量化归一化实现对零范围加了一个很小的稳定项。它不用于判断学习效果。

该命令已在云端 RTX 4090 上验证通过，生成的 checkpoint 位于
`checkpoints/pi05_piper_debug_finetune/piper_fake_smoke/{10,19}`。

## 切换到真实数据

保持 `LeRobotPiperDataConfig` 和 `PiperInputs/PiperOutputs` 不变，只需要：

1. 把真实数据上传为一个新的 LeRobot `repo_id`。
2. 将 `pi05_piper_debug_finetune` 复制为正式配置，修改 `name`、`repo_id`、训练步数、批量和 checkpoint 路径。
3. 用正式配置重新运行 `compute_norm_stats.py`，绝不复用全零数据的统计量。
4. 启动训练。

云端已验证的预训练权重路径是：

```text
/root/autodl-tmp/pi05_checkpoints/openpi-assets/checkpoints/pi05_base/params
```

正式配置应将 `CheckpointWeightLoader(...)` 指向该目录。
