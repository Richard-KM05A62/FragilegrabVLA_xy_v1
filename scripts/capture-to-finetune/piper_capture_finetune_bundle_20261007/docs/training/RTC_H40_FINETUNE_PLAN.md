# Piper X RTC：H40 / s20 / 8-step 试验配置

更新：2026-09-19。

## 参数

```text
训练配置       pi05_piper_finetune_h40
action_horizon H = 40
execution_steps s = 20
flow num_steps    = 8
控制频率          = 30 Hz
```

`H` 写入训练配置；`s` 和 `num_steps` 只在部署时设置。保留原
`pi05_piper_finetune`（H=10）及其已验证 checkpoint，不覆盖、不混用 stats。

## 重新微调

H=40 会改变数据加载器构造的动作序列，必须用新配置重新计算 norm stats，随后启动一个新的实验名：

```bash
cd /path/to/piper_x_real/openpi
.venv/bin/python scripts/compute_norm_stats.py --config-name pi05_piper_finetune_h40
.venv/bin/python scripts/train.py pi05_piper_finetune_h40 \
  --exp-name piper_h40_rtc --overwrite
```

本轮已确认使用 batch 8、5,000 step、500 step 保存一次，并只保留一个滚动 checkpoint；若 H=40 在 24 GB RTX 4090 上显存不足，停止后再单独确认是否降低 batch，不自动修改训练参数。

## 2026-09-19 云端运行记录

- 实验名：`piper_v000_18ep_h40_5k`
- checkpoint：`/root/autodl-tmp/openpi/checkpoints/pi05_piper_finetune_h40/piper_v000_18ep_h40_5k`
- 日志：`/root/autodl-tmp/openpi/logs/pi05_piper_h40_5k/train.log`
- PID 文件：`/root/autodl-tmp/openpi/logs/pi05_piper_h40_5k/train.pid`
- H=40 norm stats SHA256：`20e6dc5df44cc0a08e5c86f048137612140bc280410aad98913234ec783e484d`
- 有效数据：排除 episode 6、11 后为 18 条轨迹、3,717 帧。
- 启动验证：step 0 `loss=0.0469`，step 100 `loss=0.0385`；约 1.3 秒/step。
- RTX 4090 训练显存约 22.1 GB，batch 8 已通过首轮编译和持续训练验证。
- 训练以 `nohup` 后台运行，已经过主动断开并重新连接 SSH 的存活验证。

## 部署参数

checkpoint 必须指向 H=40 新实验的根目录，且直接含 `params/` 和 `assets/`：

```bash
cd /path/to/piper_x_real
openpi/.venv/bin/python -m inference.serve_piper_rtc \
  --config pi05_piper_finetune_h40 \
  --checkpoint /path/to/h40/checkpoint \
  --host 127.0.0.1 --port 8000 \
  --num-steps 8 --max-guidance 5
```

另一个终端显式指定匹配的 runtime horizon：

```bash
openpi/.venv/bin/python -m inference.run_piper_rtc \
  --read-only --task "与训练一致的任务描述" \
  --host 127.0.0.1 --port 8000 \
  --horizon 40 --execution-steps 20 --latency-window 20 \
  --norm-stats /path/to/h40/checkpoint/assets/local/piper_x_recordings_v000/norm_stats.json \
  --summary logs/rtc_h40_s20_n8.json
```

服务与客户端会核对 horizon 和 stats SHA256；旧 H=10 stats 或 checkpoint 应直接被拒绝，不能绕过检查。

## 验收

先用零 guidance/普通推理检查 `[40,7]` 输出，再运行 RTC 静态测试，最后才接只读真机输入。记录：

- 第一次 JIT 时间与预热后的 P50/P95/最大推理延迟；
- `nvidia-smi` 峰值显存，12 GB 卡建议留出至少约 1 GB 余量；
- 是否持续处于 RTC，而不是反复进入同步 `BOOTSTRAP`；
- 动作有限值、关节跳变、夹爪语义及 RTC guidance 残差。

若 H=40 的预热后 P95 延迟仍接近 20 个控制周期（约 667 ms），实时余量不足；应先分析推理和显存，再决定降低 `num_steps` 或调整 H/s，而不是直接接入运动下发。
