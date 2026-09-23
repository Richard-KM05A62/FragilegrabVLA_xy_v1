# PiPER-X zero-shot grasp tendency 诊断

## 当前目标

本阶段不做 fine-tuning，也不把正式抓取成功率作为唯一判断。目标是判断未微调 `pi05_base` 在 PiPER-X MuJoCo 中达到 L0，还是已经出现 approach / pre-grasp / alignment / close / contact / lift 等结构化趋势。

## 实际数据流审计

```text
MuJoCo PiPER-X + scene
  -> ObservationAdapter
     - base_rgb: HWC uint8 RGB
     - wrist_rgb: HWC uint8 RGB
     - joint_position: joint1..6 rad + two symmetric finger positions m
  -> PiperPi05Contract.encode_observation
     - base_rgb -> observation/image
     - wrist_rgb -> observation/wrist_image
     - two fingers -> closure [0,1]
     - policy state = [q1..q6, closure]
  -> WebSocket -> locked OpenPI server
  -> PiperInputs
     - base_0_rgb present
     - left_wrist_0_rgb present
     - right_wrist_0_rgb zero + masked
  -> DeltaActions(mask first six dimensions)
  -> q01/q99 normalization
  -> pi05_base, internal action_dim=32, action_horizon=50
  -> inverse normalization
  -> AbsoluteActions(mask first six dimensions)
  -> PiperOutputs slices seven effective dimensions
  -> client proposal [q1..q6 absolute rad, closure 0..1]
  -> closure maps linearly: 0=open=0.05 m, 1=closed=0 m
  -> strict range and per-step delta validation
  -> ScriptedActionAdapter / MuJoCo position actuators
  -> hold target for 25 * 0.002 s = 0.05 s (20 Hz)
  -> next observation and synchronous replan
```

服务端使用的 stats 同时作用于七维 `state` 和经 `DeltaActions` 处理后的七维 `actions`。当前不能使用 Trossen/ALOHA stats 解释 Piper；`norm_stats_id`、provenance hash 与 server metadata 必须匹配。WebSocket response 已经过 inverse normalization 和 robot-specific absolute-action transform，客户端取前七维只是在这些变换之后移除模型 padding。

## 动作与控制语义

- dim0..5：joint1..joint6 的绝对位置目标，rad。
- dim6：策略侧 closure `[0,1]`；客户端线性映射为夹爪 position actuator 米数。
- actuator 顺序由编译模型强校验为 `joint1..joint6, gripper`。
- low-level controller：MuJoCo position actuator；增益来自 `configs/benchmark/robot.yaml`，属于 benchmark-defined，不是真机辨识值。
- physics rate：500 Hz。
- simulator control rate：20 Hz。
- model action horizon：50。
- execution horizon：rollout config 决定；诊断 sweep 为 5/7/10/15/20。
- policy query rate：同步 receding horizon，约为 `20 / execution_horizon` Hz，再叠加实际推理延迟；每个前缀执行完才重新查询。

## 可用诊断信号

当前环境实际提供：夹爪两指末端碰撞几何中心的均值（EE/grasp reference）、目标物体世界位置、腕部旋转、两指位置和 closure、关节位置/速度、逐物理步接触 body pair 与法向力、物体高度、requested/executed action、完整模型 action chunk 和终止原因。

## L0–L6 判定

阈值全部位于 `configs/benchmark/grasp_diagnostic.yaml`。

| Stage | 可执行判定 |
| --- | --- |
| L0 | 未满足以下任何阶段。 |
| L1 | EE–object 最小距离相对初始距离至少减少 `approach_progress_m`。 |
| L2 | EE–object 距离进入 `pregrasp_distance_m`。 |
| L3 | 已进入 pre-grasp，且工具 z 轴与期望方向误差不超过 `alignment_tolerance_deg`。 |
| L4 | 在目标附近触发 close，且至少一指与物体有效接触。 |
| L5 | 两指均与目标物体形成有效接触，但尚未形成稳定 lift。 |
| L6 | 双指接触下物体抬升达到 `stable_lift_m`，并连续保持 `stable_lift_hold_s`。 |

另行记录 close 开始时间与距离、是否出现 lift attempt、最大抬升、最小距离时刻和 20 Hz 稀疏 tendency trace。判定只解释仿真真值，不修改策略动作，不构成 Safety Layer。

## 场景与 sweep

- `configs/benchmark/object.project.yaml`：历史 robosuite table/bread 工程诊断场景，不是当前 canonical 或示范采集基线。
- `configs/benchmark/object.libero_spatial_bowl_to_plate.yaml`：锁定的完整 LIBERO-Spatial 黑碗到盘子任务，保留官方 BDDL、arena、furniture、camera、物体集和 region placement，机器人替换为 PiPER-X。
- `configs/pi05_horizon_sweep.example.yaml`：execution horizon 5/7/10/15/20，seed 7/17/27。

`scripts/sim/run_pi05_horizon_sweep.py` 会为每个 episode 保存完整目录，并在 `summary.json` 中按 horizon 汇总最高 stage、各 stage 达成数、最小距离、close 距离与最大抬升。

## 当前验证与运行状态

- 锁定 `agx_arm_sim@f8cd8b1` + `agx_arm_urdf@1ee5659` 的 PiPER-X+D435 资产已在本地可用，LIBERO scene 和 policy tunnel 已实际完成一次 10 s 闭环。
- seed 7、`execution_horizon=10` 的 rollout 显示 `approach_progress=0.03557 m`，但最小 EE–object 距离仍为 `0.11878 m`；夹爪在距物体 `0.14862 m` 时已开始闭合，无指尖接触或抬升。根据当前阈值其最高阶段仍为 L0，但连续距离信号显示弱接近趋势。
- 本次执行记录全部 `projection.changed=false`，所以没有投影或急停阻断后续行为。下一步只做相同条件下 `execution_horizon=5/7/10` 的小对照；对照不能替代示范采集和正式 normalization 重算。
