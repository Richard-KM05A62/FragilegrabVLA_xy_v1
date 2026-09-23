# FragileManip-Bench v0.1 实施与复现

## 最小改动方案与范围

当前实现位于 `src/fragilegrab/benchmark/`，运行入口位于 `scripts/sim/` 和 `scripts/policy/`。已有 MuJoCo PiPER-X/LIBERO 组合、双 RGB observation、π0.5 WebSocket 接入、receding-horizon 执行、timeline/视频记录和评测；Isaac runner 保留为独立支线。

已确认主线：π0.5 **base** 权重、MuJoCo、AgileX 官方 `agx_arm_sim` PiPER-X + 夹爪 + D435 锁定资产，以 LIBERO 场景基础设施建立 canonical 和 safety stress-test 两层 benchmark。禁止将脚本 oracle 物体位置接入 π0.5 observation。

## 两层 benchmark 定义

### A. Canonical benchmark

- 保留锁定 LIBERO 版本的 arena、furniture、camera 和 placement logic；
- robot 替换为 PiPER-X，但不改变项目已定义的七维 action contract；
- target object 替换为有来源、有版本的 fragile / fragile-like objects；
- task predicate 针对新 object 和 PiPER-X 工作区重新定义，但不自行搭建另一套 arena 或 placement 逻辑。

### B. Safety stress-test benchmark

以 canonical config 为唯一基线，只通过版本化配置改变：可操作空间宽度、clutter、障碍物邻近程度、摩擦、脆弱参数、质量和接触风险。每次 rollout 必须记录 canonical config id、变体轴、取值、seed 和资产 hash，以便生成 matched comparison。在实现可执行损伤定义前，“脆弱”和“接触风险”保持 Candidate，不构成 Safety Layer。

因 robot、target object 和 predicate 已改动，两层结果统称 LIBERO-derived PiPER-X benchmark，不与原版 LIBERO benchmark 成绩混同。

## 微调数据与 normalization

当前先做 PiPER-X embodiment adaptation，目标是让 π0.5 学会双相机 + 关节状态到七维 PiPER-X action chunk 的控制映射。示范数据从单一稳定 canonical 抓放任务采集，fragile-like object 在本阶段只是稳定抓取载体，不进行易碎评判。

本阶段 normalization 使用实际进入该次微调的全部训练示范，统计 OpenPI Piper data transform 后的 `state` 和 `actions`。对当前契约，前六维 action 在统计前已变换为相对当前 state 的 delta，第七维保持绝对 gripper closure。统计量必须与 dataset version、transform id 和微调 checkpoint 绑定。正式 validation/test split 在控制闭环跑通后再建立。

当前 `compute_piper_norm_stats.py` 从 scripted MuJoCo episode 得到的统计只能用于接口校准或 zero-shot 调试，不能作为正式微调 normalization。

Pilot 训练使用 LeRobot、20 Hz、front RGB（可选 wrist RGB，但必须与最终 inference 一致）、七维 state 和七维 absolute joint target。模型从 `pi05_base` 初始化，保留默认内部 action dimension，首轮 fine-tuning action horizon 默认为 16，execution horizon 独立 sweep 5/7/10。Pilot 训练限于 10k–20k steps；每 1k 保存、每 5k 永久保留，并用 MuJoCo rollout 选择最终 checkpoint，不仅依赖 loss。

## 官方依据与后备路径

- [AgileX agx_arm_sim](https://github.com/agilexrobotics/agx_arm_sim)：锁定 PiPER-X + gripper + D435 URDF/USD、相机支架与官方 URDF→MJCF 工具；`agx_arm_urdf` 子模块也锁定到具体 commit。官方资产仍不等于已对用户真机做计量与动力学标定。
- [robosuite 1.5.2 机器人注册](https://github.com/ARISE-Initiative/robosuite/blob/v1.5.2/robosuite/robots/__init__.py)：本版无 PiPER。机器人模型、独立夹爪和控制器需要专用集成；本次静态核对该注册表与 `ManipulatorModel`/`Panda` 示例后采用用户授权的原生 MuJoCo 后备，不声称 robosuite 集成失败或不可能。
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) commit `8f1084e...`：当前 canonical 首任务直接读取其 BDDL、tabletop arena、furniture、objects、camera 和 region placement 资产。黑碗仅是当前 embodiment-adaptation 刚体载体，不代表已完成易碎材料建模。
- [MuJoCo 3.3.7 API](https://mujoco.readthedocs.io/en/3.3.7/APIreference/APIfunctions.html)：使用 `mj_forward`、`mj_step`、`mj_contactForce` 和 Python `Renderer`。

组合时只解析上游 XML、规范化资产路径、设置机器人世界安装位姿、加入物体自由关节、扩展 home keyframe 的物体状态、设置 benchmark 物理步长。关节范围、惯性、质量、碰撞几何、摩擦、驱动增益/限幅、mimic equality 和 gravcomp 均保留上游。上游有简化惯性和重力补偿，source-derived 只表示可追溯，不表示经过真机物理辨识。

## 配置来源

`configs/benchmark/` 中每份 YAML 的 `provenance` 作用于 `values` 的所有叶子；`provenance_overrides` 逐字段覆写。类别限定为 `source-derived`、`benchmark-defined`、`randomized`。当前无启用的随机化参数，`randomization_enabled=true` 明确拒绝运行；固定 seed 控制 reset，重复性不等于已验证随机化。

- `benchmark.yaml`：步长、频率、时长、脚本抓取位置偏移、判据和阈值扫描均为 benchmark-defined；相机来自上游。
- `robot.yaml`：版本、命名与模型约定来自 source-derived。joint/actuator 数值读取编译模型并写入审计，不在配置中重复猜测。
- `object.libero_spatial_bowl_to_plate.yaml`：锁定首个 LIBERO-Spatial task、BDDL/scene/object hash、policy camera 和 PiPER-X base pose。真实易碎物体与 fragility 校准 TBD。
- `tasks.yaml`：Qualification 的 reach/pick/transport/place 与 F1 可运行；F2–F5 只定义范围，拒绝执行。

所有长度为米，六轴为 rad，两指为米。脚本 action 是上游七个 position actuator 的绝对目标：joint1–joint6、joint7；joint8 由上游 equality 约束。禁止将此语义等同于 π0.5 action。

## 最小可执行命令

在仓库根目录执行（Linux，Python 3.12，Git，EGL 图形运行库）：

```bash
python3 -m venv .venv
.venv/bin/pip install -r scripts/sim/requirements-mujoco.txt
.venv/bin/python scripts/sim/fetch_piper_x.py
MUJOCO_GL=egl .venv/bin/python scripts/sim/run_mujoco_benchmark.py \
  --robot-assets assets/vendor/agx_arm_sim --task F1 \
  --output runs/f1_smoke --seed 7 --smoke
MUJOCO_GL=egl .venv/bin/python scripts/sim/run_mujoco_benchmark.py \
  --robot-assets assets/vendor/agx_arm_sim --task qualification \
  --output runs/qualification --seed 7
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests
```

通用 benchmark runner 默认要求输出目录不存在，以保护旧证据；专用 10 s 趋势脚本则只覆盖其固定的 `/tmp` 目录。没有 EGL 时需要另行安装和验证 OSMesa/显示服务，不静默省略 render/video。本机 shell 自动加载 ROS pytest 插件会缺失 `lark`，上述测试命令隔离插件，不改 ROS 环境。本地 MuJoCo 环境不安装 OpenPI 权重；推理由已锁定的远端 OpenPI 环境通过 loopback SSH tunnel 提供。

## Episode 与重复性

输出继承 `fragilegrab.episode/0.1` 逻辑契约，新增 `profile=mujoco_qualification/0.1`。包含有效配置及 hash、源码 hash/Git 状态、资产树 hash、依赖版本、task/seed、零位/home/夹爪/控制审计、仿真时间、逐物理步接触与状态、逐控制步 requested/executed action、RGB 原始数组、原始 MP4 和 VS Code 可预览的 VP8/WebM。`scene.xml` 含该运行机的资产绝对路径，只是本地复核快照；跨机复现从版本化配置重新组合资产。日志和视频由原有 `.gitignore` 中 `runs/` 等规则排除。

当前 LIBERO-derived policy 全局 RGB 使用锁定的 `agentview`；腕部 RGB 外参从官方 PiPER-X+D435 固定关节链和 D435 xacro 的 `camera_color_optical_frame` 解析，不使用 targetbody 或目标跟踪。D435 fovy 仍为 Candidate，待真机内参绑定。observation 保存八个 joint state；action 七维。时间为 episode 仿真时钟；接触在积分前用 `mj_forward` + `mj_contactForce` 取当前状态，逐 dt 积分，视频/观测在执行步后记录各自实际时间。reset 静置段不计入评测。失败保留 `.incomplete` 和 `failure.json`，不会生成成功完成标志。

`--smoke` 执行两次相同 seed，比较 qpos/qvel、原始 RGB、timeline 的 SHA-256 和指标；每次解码 MP4 与 WebM 检查全部帧数及尺寸，检查模型审计。视频编码字节不作为物理确定性条件。它验证当前机器与依赖组合内的重复性，不保证跨 MuJoCo/GPU/驱动逐位一致。工程 smoke 的通过和任务成功独立报告。

## 指标的可执行定义与限制

- task success：F1 在 episode 结束时仍满足双指接触、相对 reset 后静置物体初始高度升高达阈值，并连续保持至少 `hold_s`。Qualification 顺序检查接近、相同 lift 条件、搬运到目标 XY、释放并后撤后在目标附近连续低高度保持。尚无独立姿态/角速度稳定判据。
- peak contact force：每步**目标物体所有接触法向力标量之和**的最大值，单位 N，包含桌面支撑；不是六维合力，也不是指尖压力。
- contact impulse：上述法向力之和乘每个物理 dt 后累加，单位 N·s；包含持续支撑/夹持，**不能命名为瞬时 impact impulse**。另报 body pair 连续接触事件的最大累计冲量；同 pair 多个 contact 合并，超过 epsilon 才计入事件，断开即结束。阈值扫描使用 peak 和 max event impulse。
- slip：首次双指离桌后，物体与抓取参考点在腕部旋转坐标系的相对位移超过配置阈值。属于姿态真值代理，不能替代接触切向速度/真实触觉；允许释放阶段，以及 place 阶段物体在目标区域获得环境支撑时，不判 slip/drop。
- drop：曾双指抬升，且非允许释放/已支撑的目标放置阶段下降到接近初始高度并失去双指接触。中途失去接触但尚未落下时不立即报告 drop。
- unintended collision：涉及机器人、但不是指尖–目标接触，也不是安装基座–桌面的 body pair 接触事件；按接触开始次数统计，持续接触不每帧重复计数。self contact 也适用。目标–桌面为预期支撑。
- safe success：task success 且 peak/event impulse 未越过给定实验阈值，且无 slip/drop/unintended collision。当前阈值网格仅是示范性的 benchmark variable，绝非真实材料断裂值；不添加 Safety Layer，不干预控制。

## π0.5 接口当前状态

`piper_x_joint_position_dual_rgb_v0.3` 已实现并实际跑通：双路 HWC `uint8` RGB、七维 state、六轴绝对关节目标 + closure、32 维内部 padding、50-step model chunk、前缀执行/取消后缀和同步重规划均会记录。当前 scripted calibration stats 只能支撑 zero-shot 失败诊断，不是正式微调 normalization。

当前下一阶段是在已锁定的 LIBERO-derived 单任务采集 PiPER-X 完整抓放示范，从实际训练集重算 normalization，再进行 10k–20k step pilot。robosuite 机器人注册、RoboCasa、正式 fragile predicate 和 F2–F5 均不是这一步的前置。

实际验证结果见 [PiPER-X 验证记录](benchmark_validation.md)。

## 闭环诊断入口

先运行不调用 VLA 的逐维 controller/adapter sanity：

```bash
MUJOCO_GL=egl .venv/bin/python scripts/sim/check_robot_adapter.py \
  --robot-assets assets/vendor/agx_arm_sim \
  --output outputs/robot_adapter_sanity.json
```

对已有 π0.5 closed-loop episode 做离线 trace inspection：

```bash
.venv/bin/python scripts/sim/analyze_pi05_trace.py runs/<episode> \
  --output outputs/<episode>_trace_inspection.json
```

只有 executable PiPER-X normalization、adapter 和 trace 检查通过后，才按 `configs/pi05_horizon_sweep.example.yaml` 运行少量 horizon sweep。transport-only server 会被硬性拒绝。设计、语义和 LIBERO-derived scene 评估见 [仿真闭环诊断](simulation_diagnostics.md)。

### VS Code 视频预览

`base_rgb.mp4` 使用 OpenCV 的 `mp4v`，部分编辑器媒体预览不支持该编码。运行会同时生成 `base_rgb.webm`（VP8）、`preview_manifest.json`，并在 episode manifest 中登记预览 hash；原始 RGB、MP4、物理日志与指标不变。需要重新生成预览时运行：

```bash
.venv/bin/python scripts/sim/export_compatible_video.py runs/<episode>
```

转换只读取原始 RGB 帧，不覆盖已存在的证据；预览文件已存在时会拒绝覆盖。若本机 OpenCV 缺少 VP8 编码器，会给出明确错误。
