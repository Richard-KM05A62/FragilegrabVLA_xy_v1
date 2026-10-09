# π0.5–PiPER-X 仿真闭环审计与诊断计划

## 1. 当前数据流

```text
MuJoCo PiPER-X + D435 scene
  ├─ base_rgb: HWC uint8, 240×320×3
  ├─ wrist_rgb: HWC uint8, 240×320×3
  └─ qpos: joint1..joint6 rad + two fingers m
                │ PiperPi05Contract.encode_observation
                ▼
observation/image + observation/wrist_image + 7D state + prompt
                │ WebSocket / msgpack
                ▼
OpenPI pi05_base
  PiperXInputs → DeltaActions(first 6 dims) → normalization → model
  model → unnormalization → AbsoluteActions(first 6 dims) → PiperXOutputs(first 7 dims)
                │ 50×7 action chunk
                ▼
PiperPi05Contract
  ├─ six absolute joint targets, rad
  └─ gripper closure 0=open, 1=closed → left-finger actuator 0.05..0 m
                │ strict reject OR recorded diagnostic projection
                ▼
MuJoCo seven position actuators
  each target held for 25 × 0.002 s = 0.05 s (20 Hz)
                │ execute configured chunk prefix, cancel suffix, replan
                ▼
state/contact/two RGB streams/timeline/video/evaluation
```

真实代码路径：

- 仿真 raw observation：`src/fragilegrab/benchmark/environment.py`、`adapters.py::ObservationAdapter`。
- OpenPI server transform：`scripts/policy/serve_pi05_piper.py::PiperXInputs/PiperXOutputs/create_config`。
- 客户端字段与动作反变换：`src/fragilegrab/benchmark/pi05_contract.py::PiperPi05Contract`。
- chunk broker、执行与日志：`src/fragilegrab/benchmark/pi05_runner.py::run_pi05_episode`。
- 低层 position actuator：`environment.py::_configure_piper_x_mjcf/step_physics`。

## 2. Action adapter 的实际语义

`PiperXOutputs` 最后取 `[..., :7]` 本身是合理的有效维投影：模型内部使用 32 维 padding，而 PiPER-X 契约只有七个有效维度。只要有效维确实按约定放在前七维，这一步不是风险来源。

真正需要审计的是截取之前的语义链：训练数据 action convention → 前六维 delta transform → 与同一契约绑定的 normalization → 模型内部 padded action → inverse normalization → `AbsoluteActions` 依据当前 state 恢复绝对关节目标。第七维不参与 joint delta transform，保持绝对 gripper closure。完成这些 robot-specific transform 后，`PiperXOutputs` 才去除 padding；客户端再将 closure 线性映射到夹爪 position actuator。

| 有效维 | 服务响应语义 | MuJoCo 目标 | 单位/方向 |
| --- | --- | --- | --- |
| 0–5 | joint1–joint6 绝对位置 | 同名 position actuator | rad；正方向由官方 URDF joint axis 决定 |
| 6 | closure，0=open、1=closed | `0.05 + closure × (0−0.05)` | actuator 米数减小表示闭合；右指由 `joint8=-joint7` equality 驱动 |

模型内部仍是 32 维 padding 和 50-step chunk；外部接口是 `(50,7)`。关节范围、非有限值、gripper closure、单步变化和 metadata 不匹配均拒绝，不静默 clipping。`diagnostic_projection` 是独立实验模式，会记录投影，不能称作模型原生输出或 Safety Layer。

## 3. Normalization 状态

- 当前云端实际运行的 `piper_x_workflow_smoke_v0.1` 是合成 identity-like 统计，provenance 明确为 `transport_only`、`execution_permitted=false`；只能验证传输和推理。
- `piper_x_mujoco_scripted_calibration_v0.1` 已通过 metadata 门禁并实际用于 MuJoCo zero-shot 诊断闭环。它来自 scripted trajectory，只是 executable calibration stats，不是微调训练集或真机分布的证明。
- 正式 embodiment-adaptation normalization 仍必须从实际进入微调的全部 PiPER-X 示范计算。当前 rollout 中 joint1/5/6 提议变化极小，且夹爪远离物体时过早闭合；scripted stats 覆盖范围过窄与未微调 embodiment shift 都是当前主要可疑项。
- 复用 ALOHA/Trossen stats 与当前 PiPER joint-position 契约不兼容，是高风险项；不得用调 scale 或 clipping 掩盖。

锁定 OpenPI `15a9616...` 的实际顺序是：data input transform 先将前六维绝对 action 减 state，随后对 state/action 使用 q01/q99 分位数归一化；模型输出端先反归一化，再用同一条已反归一化 state 把前六维恢复为绝对关节角，最后才截取七个有效维。分位数公式是 `(x-q01)/(q99-q01+1e-6)*2-1`，因此非对称统计中不能用“正扰动的 normalized 值必须大于零”判断正负号；正确条件是负扰动、基线、正扰动三者保持严格单调，并在完整 inverse path 后恢复原绝对目标。

`scripts/policy/check_piper_transform_alignment.py` 已用六轴逐维 `-0.05/0/+0.05 rad`、三档夹爪 closure 和 32→7 padding path 运行离线审计。当前 `piper_x_workflow_smoke_v0.1` 的结果为 `transport_only_math_pass`：六轴正负方向与 roundtrip、夹爪 `closure↑ → actuator m↓`、padding 去除均通过，数值最大误差约 `1.1e-16`。这只排除了项目变换实现中的明显符号/顺序错误；由于 stats 是合成 transport-only 数据，不能证明真实 Piper 数据分布或 π0.5 训练 action convention 对齐。摘要见 `assets/manifests/piper_transform_alignment_smoke_v0.1.json`。

## 4. Horizon 与控制频率

- 模型 action horizon：50。
- 已完成的 10 s LIBERO 趋势诊断：`execution_horizon=10`，即每 0.5 仿真秒重规划。
- physics timestep：0.002 s，即 500 Hz。
- 每个 action target 保持 25 个 physics step，即 action/control rate 20 Hz。
- 当前诊断应对比 `execution_horizon=5/7/10`。`10` 不是“步幅太小”，而是 0.5 s 开环执行偏长，可能放大过早闭夹；降低它只能改善重规划频率，不能修复 normalization 或让 base 模型学会 PiPER-X。模型原生 action horizon 仍保持 50。

runner 没有跨 query chunk buffer；每次只执行前缀，剩余 suffix 记录为 cancelled，然后同步请求新 chunk。query 期间仿真暂停，所以 wall-clock inference latency 不推进 simulation time。

## 5. Observation adapter

- MuJoCo 输出两路 HWC `uint8` RGB，当前均为 `240×320×3`。
- 全局相机映射为 `base_0_rgb`；夹爪中部 D435 映射为 `left_wrist_0_rgb`；`right_wrist_0_rgb` 为同 shape 零图且 mask=false。
- state 为 joint1..joint6 rad 加由两指位置计算的 closure；两指必须满足对称约束。
- prompt 来自 rollout config。
- 本地 adapter 不 crop、不翻转、不转 CHW；OpenPI 的模型 transform 负责其官方 resize/tokenization。当前没有真机内参与仿真 fovy 的等价证据，也没有训练分布相机位姿等价证据。

## 6. Low-level controller

当前使用 MuJoCo position actuator：六轴 `kp=[80,80,80,40,10,10]`、夹爪 `kp=40`，并配置 `kv`、forcerange、armature、frictionloss 与 body gravcomp。数值为 benchmark-defined，不是真机辨识结果。policy runner 对一个目标直接 hold 25 physics step，不在两个 chunk action 之间额外插值；scripted oracle runner 在每个 20 Hz 控制块生成线性 phase target，但二者最终经过同一个 `ScriptedActionAdapter.decode → env.step_physics` 路径。

已有 scripted F1 证明当前控制路径可以完成工程抓取；逐维 sanity 也已核对六轴索引、正方向、跟踪误差和夹爪方向。

## 7. 风险分级

### Confirmed

- transport-only smoke stats 仍不能执行；scripted calibration stats 已可执行，但只用于 zero-shot 诊断，不得当作正式微调 stats。
- 当前已记录 `execution_horizon=10`；下一诊断对照为 5/7/10，模型 action horizon 始终是 50。
- 当前 action contract 是六轴绝对 joint position + closure，不是 EE delta/velocity。
- 两路 RGB/state/gripper 映射、20 Hz target hold、500 Hz physics 和完整 chunk logging 已在代码中实现。
- scripted controller path 能完成 F1 工程抓取；这不等于 π0.5 抓取成功。

### Likely

- scripted calibration normalization 的训练轨迹覆盖不足与未微调 `pi05_base` 的 embodiment shift 是首要失败来源。
- 0.5 s 的 10-step 开环前缀可能放大时序错误，需用 5/7/10 对照量化。
- 未微调 `pi05_base` 对 PiPER-X morphology、相机和任务分布存在明显 embodiment shift。

### Possible

- 关节 zero/sign 与模型隐含训练 embodiment 不同，即使 MuJoCo 内部索引正确也可能让 zero-shot 输出无意义。
- 全局/腕部相机几何和当前简单桌面布局偏离基础模型训练分布。
- 低层增益、直接 hold 和无插值策略可能放大不连续提议。

### Not yet checked

- 5/7/10 在相同 seed、场景、stats 和 prompt 下的 horizon 对照。
- 示范训练集产生的 normalization 是否解决 joint1/5/6 输出近似锁死和过早闭夹。
- 真机 Piper 的零位、方向、夹爪非线性、控制周期与仿真契约是否一致。

## 8. 三个 sanity test

1. `scripts/policy/check_piper_transform_alignment.py`：离线逐维注入正负 probe，完整复现锁定 OpenPI 的 quantile normalize/unnormalize、前六维 delta/absolute 和 32→7 padding path，并核对夹爪方向。
2. `scripts/sim/check_robot_adapter.py`：不调用 VLA，逐维通过现有 action adapter 和 position actuator 发小幅绝对目标，输出 mapping、range、方向、串扰、跟踪误差和夹爪开闭结果。完整 scripted F1 继续由 `run_mujoco_benchmark.py` 验证。
3. `scripts/sim/analyze_pi05_trace.py`：离线读取现有 closed-loop timeline，记录 query state、robot-specific output transform 后的七维 action、decoded actuator action、执行前缀、范围违规、gripper closure、chunk 内最大步差和边界跳变。它审计 normalization/transform 的最终效果，不把 `[:7]` padding 去除当成风险。官方 WebSocket 不暴露 transform 前的内部 normalized tensor，因此明确记录为 unavailable，不伪造。
4. `scripts/sim/run_pi05_horizon_sweep.py`：读取 `configs/pi05_horizon_sweep.example.yaml`，只在 server metadata 的 `execution_permitted=true` 时运行 5/7/10/15/20/25 少量固定 seed episode，并汇总 reach/pick/lift、slip/drop/collision、query/action 数和时长。

## 9. LIBERO scene infrastructure 复用评估

结论：**可以作为 Candidate 复用场景基础设施，但不能把现有 LIBERO runtime 直接换一个 robot 名称就认为兼容。**

官方 LIBERO 将任务组织为 BDDL、problem class、scene XML、placement initializer、predicate 和 init states；其 `BDDLBaseDomain` 继承 robosuite `SingleArmEnv`，默认 20 Hz，依赖已注册 robot model、base offset、gripper 和 controller。官方 requirements 固定 robosuite 1.4.0，而本项目锁定 1.5.2；当前 robosuite 1.5.2 也没有 PiPER-X 注册。LIBERO 官方 camera、placement 和 predicate 因而可借鉴/迁移，但直接导入 runtime 会同时引入版本与 embodiment 集成工作。

官方依据：

- LIBERO repository、task suites 与 MIT license：https://github.com/Lifelong-Robot-Learning/LIBERO
- LIBERO BDDL benchmark mapping：https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/benchmark/__init__.py
- LIBERO problem template：https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/templates/problem_class_template.py
- robosuite Task/Arena/RobotModel/ObjectModel 结构：https://github.com/ARISE-Initiative/robosuite/blob/master/docs/modules/environments.md

## 10. LIBERO-derived 两层 benchmark

1. 锁定具体 LIBERO commit、license、scene XML、BDDL 和 init-state 来源；不引入原 policy/checkpoint。
2. Canonical benchmark 保留 LIBERO arena、furniture、camera 和 placement logic；将 robot 替换为 PiPER-X，target object 替换为 fragile / fragile-like objects，并重新定义 task predicate。
3. Safety stress-test benchmark 仅从 canonical config 派生，覆盖更窄空间、clutter、障碍物邻近、更低摩擦、更脆弱参数、不同质量和不同接触风险。每个变体保存 canonical config id 与差异配置，支持 matched comparison。
4. 优先在现有原生 MuJoCo composition 中恢复上述 LIBERO 语义；若不完整接入 BDDL/runtime，必须逐项保留 placement 和 predicate 的来源及对照证据，不能自行搭建代替场景。
5. 因 robot、target object 和 predicate 已改变，结果命名为 LIBERO-derived PiPER-X benchmark，不宣称为未修改的原版 LIBERO benchmark。

当前默认 canonical scene 是 LIBERO-Spatial 官方任务 `Pick the akita black bowl from table center and place it on the plate`，锁定 commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`。组合器直接读取官方 BDDL、arena 和对象 XML，保留家具/干扰物、agentview/frontview 和 BDDL region 采样，仅将机器人换为 PiPER-X。上游 Python runtime 因锁定 robosuite 1.4.0 而不直接导入；项目在 robosuite 1.5.2 下用 `MujocoXMLObject` 完成等价的 MJCF 组合。资产与实测证据见 `assets/manifests/libero_spatial_bowl_to_plate_v0.1.yaml`。

## 11. 诊断工具已实现范围

- 新增 `src/fragilegrab/benchmark/diagnostics.py`。
- 新增 `scripts/sim/check_robot_adapter.py`。
- 新增 `scripts/sim/analyze_pi05_trace.py`。
- 新增 `scripts/sim/run_pi05_horizon_sweep.py`。
- 新增 `configs/pi05_horizon_sweep.example.yaml`。
- 更新 `docs/architecture.md`、`docs/status.md` 和运行手册。

这组诊断工具不修改 OpenPI 推理核心、不切换 checkpoint，也不导入 LIBERO Python runtime。fine-tuning 是后续独立阶段，不属于本诊断文档的运行入口。
