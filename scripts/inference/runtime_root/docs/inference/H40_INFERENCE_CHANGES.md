# H40 / s20 / 8-step：推理端修改清单

更新：2026-09-19。目标配置：`action_horizon=40`、`execution_steps=20`、`num_steps=8`、30 Hz。

本文描述新 H=40 checkpoint 训练完成后推理端需要同步的内容。旧 H=10 checkpoint、旧 stats 和当前已验证启动方式继续保留，不能与新模型混用。

## 1. 必须替换的部署资产

- checkpoint 根目录换成 H=40 新实验的最终 checkpoint；目录必须直接包含 `params/` 和 `assets/`。
- norm stats 必须使用同一个 H=40 checkpoint 内的
  `assets/local/piper_x_recordings_v000/norm_stats.json`。
- 模型配置改为 `pi05_piper_finetune_h40`。
- 不需要训练数据、优化器状态或旧基础权重；模型服务从 checkpoint 的完整 `params/` 恢复。

旧 H=10 stats 即使字段仍是 7 维，也不能用于新模型。客户端会比较 stats SHA256 和服务 metadata，应保留这项强制检查。

## 2. 模型服务参数

`inference/serve_piper_rtc.py` 已经按模型自身的 `action_horizon` 动态构造 reference、noise 和输出，不需要重写核心采样代码。启动时必须显式指定新配置、checkpoint 和 8 个积分步：

```bash
cd /path/to/piper_x_real
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

openpi/.venv/bin/python -m inference.serve_piper_rtc \
  --config pi05_piper_finetune_h40 \
  --checkpoint /path/to/h40/checkpoint \
  --host 127.0.0.1 --port 8000 \
  --num-steps 8 --max-guidance 5
```

暂不修改 `serve_piper_rtc.py` 的通用默认值，因为默认配置仍对应现有 H=10 checkpoint。新部署必须显式传参，避免静默加载错误组合。

服务 metadata 应报告：

```text
horizon          40
action_dim       7
model_action_dim 32
fps              30
action_space     absolute_joint_position
num_steps        8
norm_stats_sha256 = 新 checkpoint stats 的 SHA256
```

## 3. RTC harness 参数

Linux 真机入口继续使用 `inference.run_piper_rtc_linux`，它最终调用公共的 `run_piper_rtc` 参数解析与调度器。先只读验证：

```bash
openpi/.venv/bin/python -m inference.run_piper_rtc_linux \
  --read-only \
  --task "与 H40 微调数据一致的任务描述" \
  --host 127.0.0.1 --port 8000 \
  --horizon 40 --execution-steps 20 --latency-window 20 \
  --response-timeout-s 180 \
  --norm-stats /path/to/h40/checkpoint/assets/local/piper_x_recordings_v000/norm_stats.json \
  --orbbec-sdk-root /path/to/OrbbecSDK \
  --summary logs/rtc_h40_s20_n8_readonly.json
```

必须显式传 `--horizon 40 --execution-steps 20`。当前 `RuntimeConfig` 和公共 CLI 的默认值仍是 H=10/s=4，用于兼容旧模型；忘记传参时，WebSocket metadata 校验应拒绝 H=40 服务，而不是继续运行。

只读模式确认稳定进入 RTC、动作范围和夹爪语义后，才能在用户明确授权及现场安全条件满足时追加已有的 Linux execute 参数。不要把换 checkpoint 与首次真机运动合并成同一步测试。

## 4. 需要修改的一次性动作检查脚本

`inference/test_model_action_linux.py` 目前仍有两处 H=10 假设，使用 H=40 前必须修改：

1. `RuntimeConfig()` 改为接收命令行 `--horizon`、`--execution-steps`，本次传 40/20。
2. `target_index < 10` 改为 `target_index < config.horizon`，错误信息动态显示范围。

建议新增参数：

```text
--horizon 40
--execution-steps 20
```

一次性输出检查应由 `[10,7]` 改为 `[40,7]`。在完整 40 个动作保存和审查完成之前，不使用 `--execute`。

## 5. 无需修改的核心逻辑

以下生产代码已经从运行配置或模型动态读取 horizon：

- `piper_rtc_policy.py`：reference `[H,7]`、模型 noise `[1,H,32]`、输出 `[H,7]`。
- `rtc_controller.py`：延迟、padding、切换点及 `d+s<=H` 检查。
- `rtc_core.py`：bootstrap、RTC reference、executor 队列与历史容量。
- `rtc_transport.py`：服务 horizon 与 stats SHA256 契约检查。

当前输入 buffer 保留 2 秒，H=40 在 30 Hz 下覆盖约 1.33 秒，因此无需仅因 H=40 增大输入时间窗口。相机/CAN 数据格式、图像预处理、六关节 delta 变换、夹爪绝对 closedness 和输出反归一化也不改变。

## 6. 离线测试与文档中的旧固定形状

现有单元测试主要以 H=10/s=4 验证通用算法，可以继续作为回归基线；但应新增一组 H=40/s=20 的调度测试，至少覆盖：

- 模型响应形状 `[40,7]`；
- `d=9、15、18` 时能否进入 RTC；
- `d+s>H` 时必须退化；
- executor 每次只提交 20 个动作，队列不重复、不补发过期动作；
- metadata horizon/stats 不匹配时拒绝连接。

`docs/environment/LINUX_MODEL_LOAD_TEST.md` 与 `LINUX_INFERENCE_ENVIRONMENT.md` 中出现的 `[10,7]`、`(10,32)` 和 H=10 是旧 checkpoint 的历史验证记录，不应直接改写成 H=40。新模型结果应另行追加，保留二者的可追溯性。

## 7. 上线前验收顺序

1. 加载 H=40 checkpoint，核对新 stats SHA256。
2. 普通静态推理：输出 `[40,7]` 且全部有限。
3. RTC 静态推理：`num_steps=8`，记录首次 JIT、预热后 P50/P95 和显存峰值。
4. Linux 三路真机输入 + `NullCommandSink`，确认持续 RTC 而非同步 BOOTSTRAP。
5. 检查前 20 个动作的关节跳变、固件限位、夹爪 closedness/宽度转换。
6. 只有上述全部通过，才单独进行受限运动验证。

在 12 GB GPU 上，权重成功加载不等于 H=40 RTC JIT 一定成功。建议稳定峰值显存低于约 11 GB；若 OOM 或 P95 接近 20 个控制周期（约 667 ms），先优化推理，不直接进入连续控制。
