# PiPER-X π0.5 LoRA 3k AutoDL 运行手册

## 1. 目的与当前边界

本手册用于在 AutoDL 单张 RTX 4090 上完成一次小型 π0.5 LoRA 链路验证，并在 Weights & Biases（W&B）查看训练 loss。固定实验参数如下：

- 初始化权重：官方 `pi05_base`，全程只读；
- 数据：20 条 PiPER-X 双 RGB LeRobot episode，仅作为 smoke / 链路验证数据；
- action contract：七维 `[q1..q6, gripper]`，原始数据是绝对目标；训练 transform 将前六维转换为相对当前 state 的 delta，gripper 保持绝对闭合度；
- 模型内部 action dimension：32（七维有效 action 加 padding）；
- action chunk horizon：20；
- batch size：8；
- 训练步数：3000；
- adapter 回滚点：500、1000、1500、2000、2500、3000；
- W&B：只上传指标，不上传 checkpoint artifact；
- 存储策略：多个 adapter-only 快照；保存官方 trainable filter 选中的 LoRA 与小型 projection/head 参数，不保存冻结 base、optimizer 状态或完整模型。

> **当前实现状态（2026-09-28）**：项目扩展脚本直接部署在云端 OpenPI 的 `scripts/policy/`。新 horizon-20 norm 已生成，SHA-256 为 `150a819ccc5e6183d822338009b43fb4c8f4025a6ad2c343de5f8d0f496f6b44`；真实 LeRobot 样本、absolute-to-delta、gripper、padding、PiPER-X metadata 和官方 trainable-filter alignment gate 已通过。下一步是 2-step GPU smoke；3000-step 和 `base + adapter` 重载等价性仍待验证。官方原始 `scripts/train.py` 会保存完整 checkpoint，不得用于本实验。

## 2. 固定路径

以下路径来自本次 AutoDL 实机检查：

```bash
export OPENPI_ROOT=/root/autodl-tmp/openpi
export PROJECT_ROOT=/root/autodl-tmp/openpi
export BASE_CKPT=/root/autodl-tmp/cache/openpi/openpi-assets/checkpoints/pi05_base
export PIPER_DATA=/root/autodl-tmp/piper_norm20_lerobot
export HF_LEROBOT_HOME=/root/autodl-tmp/lerobot
export DATASET_ID=local/piper_x_embodiment_norm20_v0_1
export RUN_ROOT=/root/autodl-tmp/piper_lora_runs
export RUN_NAME=piper_x_pi05_lora_h20_bs8_3k
export RUN_DIR="$RUN_ROOT/pi05_piper_x_lora_h20/$RUN_NAME"
export EVIDENCE_DIR="$RUN_ROOT/evidence/$RUN_NAME"
export WANDB_PROJECT=fragilegrab-piper-lora
```

每次重新连接服务器后只需粘贴这一组路径变量。不得把 API key 写入本文件、shell history、训练配置或 Git。

## 3. W&B 登录与连通性测试

确认 OpenPI 已把 W&B 永久锁定到支持新版 API key 的版本：

```bash
cd "$OPENPI_ROOT"
uv run python -c 'import wandb; print(wandb.__version__, wandb.__file__)'
uv run wandb --version
```

两处版本都必须是 `0.22.3` 或更高。登录时只在隐藏输入中粘贴新生成的 key：

```bash
cd "$OPENPI_ROOT"
unset WANDB_X_REQUIRE_LEGACY_SERVICE
read -rsp 'W&B API key: ' WANDB_API_KEY
echo
uv run wandb login --relogin "$WANDB_API_KEY"
unset WANDB_API_KEY
uv run wandb status
```

做一次最小上传测试：

```bash
cd "$OPENPI_ROOT"
WANDB_PROJECT="$WANDB_PROJECT" uv run python - <<'PY'
import wandb

run = wandb.init(project="fragilegrab-piper-lora", name="connection-test")
wandb.log({"connection_test": 1})
print(run.url)
run.finish()
PY
```

打开输出的 URL，确认 run 中存在 `connection_test=1` 后再继续。已经在聊天、终端回显或日志中暴露过的 key 必须撤销。

## 4. 磁盘、GPU、base 与数据预检

```bash
cd "$OPENPI_ROOT"
mkdir -p "$EVIDENCE_DIR" "$HF_LEROBOT_HOME/local"
test ! -e "$RUN_DIR"

df -h /root/autodl-tmp
du -sh "$BASE_CKPT" "$PIPER_DATA"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
  --format=csv

test -f "$BASE_CKPT/params/_METADATA"
test -f "$PIPER_DATA/meta/info.json"
test "$(find "$PIPER_DATA/data" -type f -name '*.parquet' | wc -l)" -eq 20
test "$(find "$PIPER_DATA/videos" -type f -name '*.mp4' | wc -l)" -eq 40
```

建立 LeRobot 只读软链接，不复制 136 MB 数据：

```bash
DATASET_LINK="$HF_LEROBOT_HOME/$DATASET_ID"
if [ -e "$DATASET_LINK" ] || [ -L "$DATASET_LINK" ]; then
  readlink -f "$DATASET_LINK"
else
  ln -s "$PIPER_DATA" "$DATASET_LINK"
fi
test "$(readlink -f "$DATASET_LINK")" = "$(readlink -f "$PIPER_DATA")"
```

要求：base 目录不可作为 `checkpoint_base_dir`、输出目录或临时目录；运行前数据盘至少保留 4 GiB，低于该值停止。

## 5. 记录实验身份

云端 OpenPI 目录当前不含 `.git`，因此用文件哈希记录实际代码身份：

```bash
cd "$OPENPI_ROOT"
sha256sum \
  src/openpi/training/config.py \
  scripts/train.py \
  src/openpi/training/checkpoints.py \
  uv.lock | tee "$EVIDENCE_DIR/openpi_code_sha256.txt"

sha256sum "$BASE_CKPT/params/_METADATA" \
  | tee "$EVIDENCE_DIR/base_metadata_sha256.txt"
sha256sum "$PIPER_DATA/meta/info.json" \
  | tee "$EVIDENCE_DIR/dataset_info_sha256.txt"
```

这不是完整 checkpoint hash，但可以防止把不同 base 元数据或不同数据集误认为同一次实验。

## 6. 训练配置的强制契约

项目入口 `scripts/policy/piper_openpi_lora.py` 定义 PiPER-X 专用配置 `pi05_piper_x_lora_h20`，且解析结果必须同时满足：

```text
model.pi05 = true
model.action_dim = 32
model.action_horizon = 20
model.max_token_len = 200
model.paligemma_variant = gemma_2b_lora
model.action_expert_variant = gemma_300m_lora
batch_size = 8
num_train_steps = 3000
wandb_enabled = true
weight_loader = <BASE_CKPT>/params
dataset repo_id = local/piper_x_embodiment_norm20_v0_1
checkpoint_base_dir = <RUN_ROOT>
```

数据配置必须使用 PiPER 专用 input/output adapter：

- 输入读取双 RGB、七维 state 和 task prompt；
- repack 后 state 顺序仍为 `[q1..q6 rad, gripper_closedness]`；
- raw action 顺序仍为 `[q1..q6 absolute target rad, gripper_target_closedness]`；
- data transform 只对前六维执行 `action[:6] - state[:6]`；
- gripper 第七维不做 delta；
- pad 到 32 维只发生在七维语义确认之后；
- inference output unnormalize 后，output adapter 恢复绝对六关节目标并只截取有效前七维。

不得用 `pi05_libero`、`pi05_droid`、旧 `pi05_piper_norm20` 或其他 embodiment 配置代替这个配置。

## 7. normalization

仅在第 6 节配置实际存在且 loader smoke 通过后计算。这个入口复用 OpenPI 官方 loader 与 `RunningStats`，统计顺序固定为 repack → `PiperXInputs` → 六关节 absolute-to-delta → stats：

```bash
cd "$OPENPI_ROOT"
PYTHONPATH="$PROJECT_ROOT:$OPENPI_ROOT/src" uv run \
  "$PROJECT_ROOT/scripts/policy/compute_piper_lora_norm_stats.py" \
  --base-checkpoint "$BASE_CKPT" \
  --assets-base-dir "$RUN_ROOT/assets" \
  --checkpoint-base-dir "$RUN_ROOT" \
  --dataset-id "$DATASET_ID"
```

计算完成后，找到该配置实际生成的 `norm_stats.json`，复制其路径与哈希到实验目录：

```bash
find "$RUN_ROOT/assets" -type f -name norm_stats.json -print
```

选定正确文件后执行（把右侧路径替换为上一步确认的唯一目标）：

```bash
export NORM_STATS="$RUN_ROOT/assets/pi05_piper_x_lora_h20/$DATASET_ID/norm_stats.json"
sha256sum "$NORM_STATS" | tee "$EVIDENCE_DIR/norm_stats_sha256.txt"
```

启动训练前必须离线打印 transformed sample 和统计量，确认：

- state/action 都是 32 维模型张量，其中仅前七维有 PiPER 物理语义；
- action 前六维量级是相邻目标相对当前 state 的 rad delta；
- action 第七维保持 `[0,1]` 附近的绝对 gripper 值；
- padding 维没有被误当成有效维统计；
- `q01/q99/mean/std` 不含 NaN/Inf；
- 第五关节等维度的极窄分布被明确记录为数据局限，而不是通过人工放大 stats 修复。

运行项目自带的离线对齐门禁：

```bash
cd "$OPENPI_ROOT"
PYTHONPATH="$PROJECT_ROOT:$OPENPI_ROOT/src" uv run \
  "$PROJECT_ROOT/scripts/policy/check_piper_lora_alignment.py" \
  --base-checkpoint "$BASE_CKPT" \
  --assets-base-dir "$RUN_ROOT/assets" \
  --run-root "$RUN_ROOT" \
  --dataset-id "$DATASET_ID" \
  --output "$EVIDENCE_DIR/alignment_report.json"
```

只有输出包含 `"passed": true` 才能继续。它首先要求 LeRobot metadata 的 `robot_type` 以 `piper_x` 开头，再用真实第一个样本验证六关节 delta 公式、gripper 不变、`(20,7)` pre-norm action、`(20,32)` model action，以及官方 LoRA trainable filter 没有放开 LLM base 权重。目录名中出现 `piper` 不作为机器人型号证据。本次云端检查已通过，报告保存在 `$EVIDENCE_DIR/alignment_report.json`。

## 8. 启动前硬门禁

以下各项必须全部回答“是”：

1. `pi05_piper_x_lora_h20` 能被当前云端 OpenPI 环境解析；
2. loader 实读一个 batch，图像/state/action/prompt 的 key、shape 和 dtype 符合第 6 节；
3. `compute_norm_stats` 使用的 transformed sample 与在线训练 sample 逐字段一致；
4. 可训练参数检查显示 LLM 内只有 LoRA leaves 可训练；官方 filter 允许少量非 LLM projection/head leaves，并将它们一并视为 adapter；
5. 训练循环已经接入 `scripts/policy/piper_lora_snapshot.py`，并在 500 步间隔保存 adapter-only 快照；
6. 快照目录中只有 `adapter_params.npz` 和 `manifest.json`，没有完整 `params/`；
7. 本轮明确不保存 optimizer 状态；中断恢复策略是保留已有 adapter、使用新实验名重新开始，不能宣称严格续训；
8. 保存前检查保证写完后仍至少保留 1 GiB；
9. 用 `base + adapter` 重新加载后的同输入输出与训练内存中的模型一致；
10. W&B 只记录 loss、grad norm、parameter norm、learning rate、step 和必要诊断图像，不上传模型 artifact。

本次第 1–4 项已通过离线 alignment；第 5、6、8、10 项由项目入口实现，须由 2-step smoke 运行验证；第 9 项在 3000-step 后验证。第 7 项的 optimizer 断点续训不启用，本轮选择“保留六个可回滚 adapter，但进程中断后开启新 run”。

## 9. 先跑 2-step smoke

先恢复第 2 节环境变量并执行硬检查：

```bash
cd "$OPENPI_ROOT"
test "$PROJECT_ROOT" = "$OPENPI_ROOT"
test -f "$NORM_STATS"
test "$(sha256sum "$NORM_STATS" | awk '{print $1}')" = \
  "150a819ccc5e6183d822338009b43fb4c8f4025a6ad2c343de5f8d0f496f6b44"
grep -q '"passed": true' "$EVIDENCE_DIR/alignment_report.json"
grep -q '"trainable_filter_matches_official_lora_contract": true' \
  "$EVIDENCE_DIR/alignment_report.json"
df -BG /root/autodl-tmp
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
```

2-step 使用独立 run 名；smoke 只执行两步，但沿用正式 3000-step 的完整学习率日程，因此前两步与正式训练一致。若该目录已存在或先前 smoke 失败，不覆盖，递增后缀（如 `_r2`）再运行：

```bash
export SMOKE_NAME="${RUN_NAME}_smoke2"
export SMOKE_DIR="$RUN_ROOT/pi05_piper_x_lora_h20/$SMOKE_NAME"
test ! -e "$SMOKE_DIR"
export OPENPI_IDENTITY="$(sha256sum "$OPENPI_ROOT/scripts/train.py" | awk '{print $1}')"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export WANDB_PROJECT=fragilegrab-piper-lora

set -o pipefail
PYTHONPATH="$PROJECT_ROOT:$OPENPI_ROOT/src" uv run \
  "$PROJECT_ROOT/scripts/policy/train_piper_lora.py" \
  --openpi-root "$OPENPI_ROOT" \
  --base-checkpoint "$BASE_CKPT" \
  --assets-base-dir "$RUN_ROOT/assets" \
  --run-root "$RUN_ROOT" \
  --exp-name "$SMOKE_NAME" \
  --dataset-id "$DATASET_ID" \
  --steps 2 \
  --openpi-identity "$OPENPI_IDENTITY" \
  2>&1 | tee "$EVIDENCE_DIR/smoke2.log"
```

运行完成后验收：

```bash
test -f "$SMOKE_DIR/2/adapter_params.npz"
test -f "$SMOKE_DIR/2/manifest.json"
test ! -d "$SMOKE_DIR/2/params"
PYTHONPATH="$PROJECT_ROOT:$OPENPI_ROOT/src" uv run python \
  "$PROJECT_ROOT/scripts/policy/piper_lora_snapshot.py" "$SMOKE_DIR/2"
grep -E 'loss=|grad_norm=|param_norm=|learning_rate=' "$EVIDENCE_DIR/smoke2.log"
du -sh "$SMOKE_DIR"
df -h /root/autodl-tmp
```

2-step 必须证明：base 初始化和真实 batch 成功、loss/grad norm/param norm/learning rate 有限、反向传播成功、batch 8 不 OOM、W&B 页面出现曲线、adapter manifest 的 `parameter_selection` 为 `openpi_train_config.trainable_filter`，且没有完整 checkpoint。如果 RTX 4090 OOM，记录为固定实验配置不成立；不得静默降低 batch size 后仍沿用相同实验名。

### 9.1 可选 500-step loss 诊断

2-step 通过后，可先运行独立的 500-step 诊断观察 W&B loss 曲线。它沿用正式 3000-step 学习率日程，并只在完成 step 500 时保存一份 adapter：

```bash
export PILOT500_NAME="${RUN_NAME}_pilot500"
export PILOT500_DIR="$RUN_ROOT/pi05_piper_x_lora_h20/$PILOT500_NAME"
test ! -e "$PILOT500_DIR"

cd "$OPENPI_ROOT"
set -o pipefail
PYTHONPATH="$PROJECT_ROOT:$OPENPI_ROOT/src" uv run \
  "$PROJECT_ROOT/scripts/policy/train_piper_lora.py" \
  --openpi-root "$OPENPI_ROOT" \
  --base-checkpoint "$BASE_CKPT" \
  --assets-base-dir "$RUN_ROOT/assets" \
  --run-root "$RUN_ROOT" \
  --exp-name "$PILOT500_NAME" \
  --dataset-id "$DATASET_ID" \
  --steps 500 \
  --openpi-identity "$OPENPI_IDENTITY" \
  2>&1 | tee "$EVIDENCE_DIR/pilot500.log"
```

该 run 不保存 optimizer state，不能从 step 500 严格续训到 3000；后续正式 3k 必须从 base 重新开始。当前官方 trainable filter 的单份 adapter 实测约 1.7 GB，因此启动前必须把现有 smoke adapter 和预期 500-step adapter 纳入磁盘预算，并保证保存后仍余至少 1 GiB。

## 10. 3000-step 正式启动

门禁和 2-step smoke 通过后，使用最终已经验证的训练入口启动。固定运行语义为：

```text
config: pi05_piper_x_lora_h20
exp name: piper_x_pi05_lora_h20_bs8_3k
steps: 3000
batch size: 8
adapter snapshot interval: 500
full checkpoint saving: disabled
W&B project: fragilegrab-piper-lora
W&B artifact upload: disabled
XLA memory fraction: 0.90
```

推荐在 `tmux` 中运行，避免 SSH 断开终止训练：

```bash
tmux new -s piper-lora
cd "$OPENPI_ROOT"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export WANDB_PROJECT=fragilegrab-piper-lora
export OPENPI_IDENTITY="$(sha256sum "$OPENPI_ROOT/scripts/train.py" | awk '{print $1}')"
```

然后执行项目自有训练入口；它调用当前 OpenPI 官方 `init_train_state` 和 `train_step`，但不创建官方 checkpoint manager：

```bash
test ! -e "$RUN_DIR"
set -o pipefail
PYTHONPATH="$PROJECT_ROOT:$OPENPI_ROOT/src" uv run \
  "$PROJECT_ROOT/scripts/policy/train_piper_lora.py" \
  --openpi-root "$OPENPI_ROOT" \
  --base-checkpoint "$BASE_CKPT" \
  --assets-base-dir "$RUN_ROOT/assets" \
  --run-root "$RUN_ROOT" \
  --exp-name "$RUN_NAME" \
  --dataset-id "$DATASET_ID" \
  --steps 3000 \
  --openpi-identity "$OPENPI_IDENTITY" \
  2>&1 | tee "$EVIDENCE_DIR/train_3k.log"
```

离开 tmux：按 `Ctrl-b`，再按 `d`。重新进入：

```bash
tmux attach -t piper-lora
```

### 10.1 低磁盘 1k/2k 阶段训练

需要先比较 1000 与 2000 step，且云端数据盘空间受限时，可以从 base 启动一次 2000-step run，只在 1000 和 2000 保存 adapter-only 快照：

```bash
export STAGED_NAME="${RUN_NAME}_staged2k"
export STAGED_DIR="$RUN_ROOT/pi05_piper_x_lora_h20/$STAGED_NAME"
test ! -e "$STAGED_DIR"

cd "$OPENPI_ROOT"
set -o pipefail
PYTHONPATH="$PROJECT_ROOT:$OPENPI_ROOT/src" uv run \
  "$PROJECT_ROOT/scripts/policy/train_piper_lora.py" \
  --openpi-root "$OPENPI_ROOT" \
  --base-checkpoint "$BASE_CKPT" \
  --assets-base-dir "$RUN_ROOT/assets" \
  --run-root "$RUN_ROOT" \
  --exp-name "$STAGED_NAME" \
  --dataset-id "$DATASET_ID" \
  --steps 2000 \
  --save-steps 1000 2000 \
  --openpi-identity "$OPENPI_IDENTITY" \
  2>&1 | tee "$EVIDENCE_DIR/train_staged2k.log"
```

该运行不写入完整模型 checkpoint、optimizer state 或合并后权重；仅保留 `1000/` 和 `2000/` 两份官方 trainable-filter adapter 快照。`resolved_config.json` 会记录实际 `save_steps`。因未保存 optimizer state，中断后仍不能严格续训。

## 11. 训练中监控

另开 SSH 会话：

```bash
watch -n 10 nvidia-smi
```

检查磁盘和快照：

```bash
watch -n 30 "df -h /root/autodl-tmp; du -sh '$RUN_DIR'"
find "$RUN_DIR" -maxdepth 3 -type f -printf '%p %s bytes\n' | sort
```

W&B 页面重点观察：

- `loss` 是否为有限值并呈总体下降趋势；
- `grad_norm` 是否持续爆炸或为 NaN；
- `learning_rate` 是否符合配置；
- step 是否连续增长；
- run config 是否准确记录 horizon 20、batch 8、3000 steps、dataset id 和 base checkpoint identity。

训练 loss 只能用于管线诊断，不能单独决定“最好权重”。最终选择需要离线固定样本对比和 MuJoCo rollout；所以保留六个 adapter 回滚点，而不是只保留最终一步。

## 12. 完成验收

训练结束后执行：

```bash
test -f "$RUN_DIR/500/manifest.json"
test -f "$RUN_DIR/1000/manifest.json"
test -f "$RUN_DIR/1500/manifest.json"
test -f "$RUN_DIR/2000/manifest.json"
test -f "$RUN_DIR/2500/manifest.json"
test -f "$RUN_DIR/3000/manifest.json"

find "$RUN_DIR" -type d -name params -print
du -sh "$RUN_DIR"
df -h /root/autodl-tmp
```

`find ... -name params` 应无输出。逐个验证 LoRA 快照可读：

```bash
for step in 500 1000 1500 2000 2500 3000; do
  PYTHONPATH="$PROJECT_ROOT" uv run python \
    "$PROJECT_ROOT/scripts/policy/piper_lora_snapshot.py" \
    "$RUN_ROOT/pi05_piper_x_lora_h20/$RUN_NAME/$step"
done
```

启动选定 adapter 的推理服务（以下以 3000 步为例）：

```bash
cd "$OPENPI_ROOT"
PYTHONPATH="$PROJECT_ROOT:$OPENPI_ROOT/src" uv run \
  "$PROJECT_ROOT/scripts/policy/serve_pi05_piper_lora.py" \
  --base-checkpoint "$BASE_CKPT" \
  --adapter-snapshot "$RUN_ROOT/pi05_piper_x_lora_h20/$RUN_NAME/3000" \
  --assets-base-dir "$RUN_ROOT/assets" \
  --run-root "$RUN_ROOT" \
  --dataset-id "$DATASET_ID" \
  --host 127.0.0.1 \
  --port 8000
```

最后确认：

- W&B run 状态为 Finished，loss 曲线可见；
- 六个 adapter-only 快照均通过 checksum/shape/dtype 校验；
- base checkpoint 的元数据哈希未变化；
- `base + 3000 adapter` 能完成固定输入 inference；
- 至少对 500/1000/1500/2000/2500/3000 做同一离线 batch 的 loss 对比；
- 进入 MuJoCo 前再次验证输出为 `(20, 7)`，前六维恢复为绝对关节目标，第七维为 gripper command。

### 12.1 500-step MuJoCo 诊断 rollout

先用 `piper_lora_snapshot.py` 验证 step 500 adapter，再以只读 base + adapter overlay 启动云端服务。云端服务必须发布 horizon 20、PiPER-X norm id、绝对关节输出语义及 `execution_permitted=true`；本地 runner 不会把 20 和 zero-shot 50 静默互换。

本地首先运行 `check_pi05_mujoco_inference.py`，它只生成一次观测和 action chunk，不执行动作。只有 metadata、`(20, 7)` shape、数值有限性和第一个动作前缀检查都通过后，才使用 `configs/pi05_mujoco_piper_x_lora_h20_pilot500.yaml` 做诊断 rollout。该配置每次推理只执行 1 个 action，最多 20 次查询，并显式记录范围/速率投影、接触力停止、视频和 timeline。这一轮只回答动作方向、夹爪时序和闭环是否改善，不把 500-step checkpoint 认定为完成训练的策略。

## 13. 中断、恢复与回滚

- SSH 断开：重新连接并 `tmux attach -t piper-lora`；
- 进程异常：保留已有 adapter 快照，不覆盖；本轮没有 optimizer 状态，必须使用新实验名重新开始，不能宣称严格续训；
- loss 变差：不要覆盖后续或旧快照，离线选择较早的 adapter；
- 磁盘低于 1 GiB：停止训练，不删除 base 和数据；先检查是否误写了完整 checkpoint 或 W&B artifact cache；
- 不允许把 adapter 合并回 `pi05_base` 目录。推理始终采用只读 base 加选定 adapter overlay。

## 14. 结论解释边界

这次 20-episode、3k-step LoRA 只能回答：PiPER 数据、transform、normalization、LoRA 更新、W&B 记录和推理 overlay 是否形成一致闭环。它不能证明模型已获得稳定抓放能力，也不能证明 20 条单场景数据具有足够泛化性。

若 LoRA 后固定训练样本 loss 明显下降、输出更接近示范，同时 `base + 官方 norm` 仍优于 `base + Piper norm` 的 zero-shot，则支持此前判断：Piper-specific norm 是与 Piper 微调共同使用的训练接口，不能期望在未微调 `pi05_base` 上单独替换 norm 就改善行为。若连训练 batch 都无法拟合，优先回到数据格式、transform、adapter 和 norm 对齐检查。
