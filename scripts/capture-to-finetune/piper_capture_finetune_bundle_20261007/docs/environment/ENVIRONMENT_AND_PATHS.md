# Piper X / OpenPI 环境与路径记录

更新时间：2026-09-18（本机 OpenPI 路径与 Linux 交接）。云端及早期微调部分是 2026-09-13 的历史记录，不代表云端当前状态；不记录 SSH 密码或其他访问凭据。

OpenPI 已整体移入本工作区的 `openpi/`，桌面原目录不再保留。Linux 环境和原 5k 权重的加载测试请先阅读 `LINUX_MODEL_LOAD_TEST.md`；本机 `models/` 是当前选定的原 5k checkpoint，不能误换为后续 10k 权重。尚未在 Linux 验证模型加载。

## 本机（Windows）

| 项目 | 已验证位置 / 版本 | 用途 |
|---|---|---|
| Piper 工作区 | `D:\piper_x_real` | 数采、时间对齐、假数据与项目文档 |
| 本机统一解释器 | `D:\vscode\opencv\venv\Scripts\python.exe` | Piper CAN / 相机 / LeRobot 数采脚本 |
| 本机统一虚拟环境 | `D:\vscode\opencv\venv` | 上述解释器的依赖环境 |
| 本机 Hugging Face Datasets | `3.6.0` | 已与云端训练环境对齐；用于写出可被云端读取的 LeRobot 数据。 |
| 本机 OpenPI 源码 | `D:\piper_x_real\openpi` | Piper 专用 OpenPI 配置；已包含 Git 记录及本地改动 |
| OpenPI commit | `15a9616` | 与云端 OpenPI 的 Git HEAD 一致 |
| Orbbec C SDK | `C:\Users\13302\Downloads\OrbbecSDK_C_C++_v1.10.37_20260707_3f75820b8_win_x64_release\OrbbecSDK_v1.10.37` | DaBai DC1 相机适配器使用的 SDK |

### 本机项目中的关键文件

| 文件 | 作用 |
|---|---|
| `capture/aligned_capture.py` | 双相机 + Piper CAN 的缓存、时间对齐与 LeRobot episode 写入核心；只读本体状态。 |
| `capture/run_aligned_capture.py` | 数采主循环入口：`python -m capture.run_aligned_capture`。 |
| `capture/collect_piper_lerobot.py` | 早期的单相机 LeRobot 数采实现。 |
| `capture/generate_fake_lerobot_dataset.py` | 生成与正式 schema 一致的全零调试数据集。 |
| `data\fake_zero_lerobot_v001` | 当前生成的 2 episode / 128 frame LeRobot v2.1 假数据。 |
| `docs/collection/DATA_COLLECTION_DESIGN.md` | 数采 schema、时间对齐策略与动作代理约定。 |
| `docs/training/OPENPI_PIPER_FINETUNE.md` | Piper 数据流入 OpenPI 的训练约定及早期运行说明。 |

### 本机 OpenPI 的 Piper 改动

| 文件 | 作用 |
|---|---|
| `D:\piper_x_real\openpi\src\openpi\policies\piper_policy.py` | 定义 Piper 图像/state 输入、7 维动作输出。 |
| `D:\piper_x_real\openpi\src\openpi\training\config.py` | `LeRobotPiperDataConfig`、调试配置及正式 `pi05_piper_finetune`。 |
| `D:\piper_x_real\openpi\src\openpi\training\data_loader.py` | 训练/统计量计算排除 episode 6、11。 |

早期部署时需要同步这些 Piper 改动；随后已完成真实数据微调。以下云端部分保留当时的环境记录，不应据此重新覆盖本机已选定模型。

## 云端（AutoDL）

| 项目 | 已验证位置 / 版本 | 用途 |
|---|---|---|
| OpenPI 根目录 | `/root/autodl-tmp/openpi` | 云端训练项目 |
| 云端解释器 | `/root/autodl-tmp/openpi/.venv/bin/python` | 必须用于 OpenPI 脚本；裸 `python` 缺少依赖。 |
| 云端 Python | `3.11.16` | OpenPI 项目环境 |
| 云端 LeRobot | `0.1.0` | OpenPI 的 LeRobot 数据加载依赖 |
| 云端 Hugging Face Datasets | `3.6.0`（由 `uv` 显式锁定） | 必须与数据写入端一致，才能加载 LeRobot v2.1 Parquet。 |
| 云端 OpenPI commit | `15a9616` | 与本机开发副本的 Git HEAD 一致 |
| GPU | NVIDIA GeForce RTX 4090，24GB | π0.5 LoRA 微调 |
| 数据盘 | `/root/autodl-tmp` | 云端代码、权重、数据应优先放在此处 |

### 云端已验证的 π0.5 权重

```text
/root/autodl-tmp/pi05_checkpoints/openpi-assets/checkpoints/pi05_base/params
```

该路径已实际通过 `CheckpointWeightLoader` 载入并与 π0.5 LoRA 模型参数结构校验通过。创建模型和加载权重的首次测试峰值约为 `9.5 GiB` 显存，低于 RTX 4090 的 24GB。

正式 Piper 配置中的权重加载器应使用：

```python
weight_loaders.CheckpointWeightLoader(
    "/root/autodl-tmp/pi05_checkpoints/openpi-assets/checkpoints/pi05_base/params"
)
```

### 云端数据缓存约定

当前已探查到的 LeRobot 缓存根目录是：

```text
/root/autodl-tmp/.cache/huggingface/lerobot
```

调试数据的 `repo_id` 是 `local/piper_x_fake_zero`。若将本机目录
`D:\piper_x_real\data\fake_zero_lerobot_v001` 上传至云端，应放为：

```text
/root/autodl-tmp/.cache/huggingface/lerobot/local/piper_x_fake_zero/
```

该目录下必须直接包含 `meta/`、`data/`、`videos/` 与 `piper_capture_manifest.json`。

> 更新：云端现在已经有一个活动的 `local/piper_x_fake_zero` 调试集。它由云端自身的
> `datasets==3.6.0` / LeRobot 环境重新生成，包含 2 个 episode、128 帧，并已通过完整训练读取。
> 最初从本机上传、由 `datasets==5.0.1` 写出的副本保留在
> `piper_x_fake_zero_datasets5_backup`。该副本的 Parquet 元数据使用 `List`，不能由云端当前
> LeRobot 读取。

### 数据集写入端版本约束

本机此前使用的 `datasets 5.0.1` 与云端 `3.6.0` 生成/读取的 LeRobot v2.1 Parquet 元数据不兼容：
云端会在读取 `List` 特征时失败。因此本机数采虚拟环境已固定为与云端相同的 `datasets==3.6.0`；
开始真实数采前仍应重新验证一段小型采集数据。

> 更新：本机数采虚拟环境现已切换到 `datasets==3.6.0`。该环境中已有 Isaac Sim 包，
> `pip check` 报告 Isaac Sim 的多个既有精确版本约束不满足，其中 `fsspec==2026.2.0` 与
> `datasets 3.6.0` 所需的 `fsspec<=2025.3.0` 直接冲突。Piper 数采所需的 LeRobot/Datasets
> 导入已验证正常；若之后需要在同一解释器运行 Isaac Sim，应建立独立 Isaac Sim 环境，不能再
> 将两者维持在同一依赖集合中。

云端另已安装系统级 FFmpeg（Ubuntu 包 `ffmpeg`）；这是 TorchCodec 解码 LeRobot AV1 视频的必要
运行时依赖，正式训练镜像也必须保留。

### 云端训练入口

OpenPI 提供：

```text
/root/autodl-tmp/openpi/scripts/compute_norm_stats.py
/root/autodl-tmp/openpi/scripts/train.py
/root/autodl-tmp/openpi/scripts/train_pytorch.py
```

当前 π0.5 权重已按 JAX 路线验证，因此默认采用 `train.py`。所有命令都从 OpenPI 根目录、通过项目虚拟环境执行：

```bash
cd /root/autodl-tmp/openpi
.venv/bin/python scripts/compute_norm_stats.py --config-name <config-name>
.venv/bin/python scripts/train.py <config-name> --exp-name <experiment-name> --overwrite
```

旧的 `train.log` 记录的是系统 Python 启动失败（`ModuleNotFoundError: etils`），不应作为环境健康状况的判断依据。

## 已确定的数据与变换契约

- 数据集格式：LeRobot v2.1。
- 图像字段：`observation.images.base`（D435i 底座视角）与 `observation.images.wrist`（DaBai DC1 腕部视角），RGB `uint8[H,W,3]`。
- 状态与原始动作：`float32[7] = [q1..q6, gripper_closedness]`；关节为弧度，夹爪 0 为全开、1 为全闭。
- 当前数采动作：下一逻辑时刻的从臂状态代理；最后一帧使用自身状态，故关节增量为 0。
- π0.5 训练前：前 6 个绝对关节动作减去当前关节 state，变为增量；夹爪保持绝对值。
- π0.5 推理后：逆变换恢复绝对关节目标；Piper harness 只接收模型 32 维动作头的前 7 维。
- 图像预处理：OpenPI 使用 `resize_with_pad` 等比例补边至 `224×224`，不进行中心裁剪。真机推理需要输入未裁剪 RGB 图像，由同一 policy transform 处理。

## 下一步部署顺序

1. 将本机 `piper_policy.py` 与改动后的 `training/config.py` 同步到云端同路径。
2. 将 `pi05_piper_debug_finetune` 的 `CheckpointWeightLoader` 改为上述本地云端权重路径。
3. 上传假 LeRobot 数据到指定缓存目录。
4. 执行统计量计算，再执行 20 步训练冒烟测试。
5. 成功后复制为真实数据的正式配置：替换 `repo_id`、数据统计量、步数和实验名，其他数据契约保持不变。

## 已完成的云端冒烟测试

- 已成功运行 `compute_norm_stats.py --config-name pi05_piper_debug_finetune`。
- 已成功从本地 pi0.5 `params` 权重加载并完成 20 个 LoRA 微调 step。
- 第 0 步指标：`loss=0.1875`、`grad_norm=0.8738`；全零数据仅用于流程验证，不代表模型质量。
- 训练过程显存峰值约为 `18.8 GiB`，低于 RTX 4090 的 24GB 容量。
- 已保存 checkpoint：
  `/root/autodl-tmp/openpi/checkpoints/pi05_piper_debug_finetune/piper_fake_smoke/10`
  与 `/root/autodl-tmp/openpi/checkpoints/pi05_piper_debug_finetune/piper_fake_smoke/19`。
