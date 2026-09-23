# FragileGrabVLA 项目状态

本文件按证据强度记录状态。`Confirmed` 表示项目决定；`Verified` 表示仓库中已有可检查的运行或静态验证证据；`In Progress` 表示正在实现或验证；未知内容标记为 `TBD`。

## Confirmed

- 项目定位为 Piper X + π0.5 易碎物体抓取研究工作区，覆盖真机数据、仿真、策略和评测。
- 当前只计划使用 π0.5，并保留原生 action chunk。
- 在 embodiment-adaptation 微调前，保留 OpenPI 官方 `pi05_base` 的 zero-shot 诊断基线，用于验证数据链路、动作语义和失败模式；该基线不取代后续 PiPER-X 示范采集、normalization 重算和 10k–20k step pilot。
- 零微调基线使用锁定 OpenPI commit 中 `Pi0Config(pi05=True)` 和 `sample_actions` 的默认设置，不添加基于结果调出的模型采样参数。Piper 字段映射、padding、归一化和仿真控制映射仍是运行所必需的显式接口配置。
- 外部 `agx_ws` 属于模型切换前的 OpenVLA 历史实现；当前项目不继续采用 OpenVLA，只将其中的 Piper X 数据与执行接口作为核对依据。
- 外部 `agx_datacollect_ws` 属于同一代 OpenVLA 历史数采管线；它只作为旧数据来源和设备语义参考，不替代当前双相机 LeRobot 采集器。
- 近期仿真主线更新为 MuJoCo + AgileX 官方 `agx_arm_sim` PiPER-X + 夹爪 + D435 锁定资产，优先 robosuite；复杂/OOD 场景后续选择性使用 RoboCasa。Isaac Sim 6.1.0 作为保留支线。
- Canonical benchmark 基于 LIBERO 场景：保留 arena / furniture / camera / placement logic，robot 替换为 PiPER-X，target object 替换为 fragile / fragile-like objects，task predicate 重新定义。不自行搭建替代场景。
- Safety stress-test benchmark 从同一 canonical scene 派生，覆盖更窄空间、clutter、障碍物邻近、更低摩擦、更脆弱参数、不同质量和不同接触风险；所有变体都必须保存相对 canonical config 的差异与来源。
- 当前 embodiment-adaptation normalization 从实际进入本轮微调的全部训练示范计算，且统计发生在锁定 OpenPI Piper data transform 之后；正式 validation/test 与 stress-test 数据设计后移。
- 当前最高优先级是 **PiPER-X embodiment adaptation**：先让 π0.5 学会 PiPER-X 的视觉–状态–动作映射和完整抓放控制。fragile-like object 在此阶段只是稳定的抓取载体；易碎评判、损伤阈值、正式 validation 和 safety stress-test 均后移，不作为当前闭环的阻塞项。
- PiPER-X embodiment pilot 使用 LeRobot、20 Hz、七维 state `[q1..q6, gripper]`、七维 absolute joint target action 与每 episode 一条真实 manipulation instruction。使用front RGB和wrist RGB。
- Pilot 从 `pi05_base` 初始化，保留模型默认内部 action dimension，dataset 有效 action 维度为 7。Fine-tuning action horizon 首轮候选 10/16，默认选 16；execution horizon 独立 sweep 5/7/10。Zero-shot `pi05_base` 的50步 horizon 作为另一基线保留。
- Pilot 训练 10k–20k steps，每 1k 保存 checkpoint，每 5k 永久保留；最终使用仿真 rollout 而不是仅凭 training loss 选择约 4–6 个 checkpoint。Piper-X normalization 为主实验，Trossen stats 仅作独立 ablation。
- 真实数据采集使用一个外部相机和一个腕部相机；外部相机具备深度能力。
- 项目成员确认主臂关节和夹爪控制目标可以取得；当前采集器尚未接入或验证该数据流。
- 当前 Isaac Sim 目标版本为 NVIDIA Isaac Sim 6.1.0，仿真 instrumentation 适配优先于 π0.5 推理接入。
- 仿真场景不从空白 USD 自行设计；优先迁移 AgileX College 的 Piper 方块堆叠任务并使用 AgileX 官方 Piper X USD，DynamicVLA 的 Piper pick/place 与 DOM scene/object 包作为第二候选。上游版本均低于 6.1.0，采用前必须记录来源并完成 6.1.0 迁移验证。
- 保留的 Isaac 支线原计划在队友同一台电脑上运行 Isaac Sim 6.1.0 和 π0.5 推理，使用独立进程与依赖环境并通过本机连接交换 observation/action chunk。
- 仿真实验在微调成功后需要设置物体的易碎相关属性，生成抓取视频，并通过项目评测体系计算指标。
- Episode 数据契约需要为不可变原始数据、版本化数据处理、多个时钟域与对齐证据、π0.5 原生 action chunk 的执行/重规划记录，以及未来新增的感知、控制和安全评测信号保留向后兼容的扩展位置；未知信号不以占位数值伪造。
- 易碎物体安全机制整体为 **Open Research Question / TBD**；项目没有已确认或已实现的 Safety Layer。
- 代码、配置模板和项目事实进入 Git；数据、视频、权重、checkpoint、日志和生成产物存放在外部或本地。

## Verified in Repository

- `scripts/data_collect/aligned_capture.py` 和 `run_aligned_capture.py` 可被 Python 语法解析。
- 静态检查显示，当前数采代码只读取 D435i RGB、DaBai DC1 RGB 和 Piper X 关节/夹爪反馈，不包含机械臂使能、运动、夹爪命令、复位或标定调用。
- 当前代码以主机 `time.monotonic_ns()` 为三路数据打时间戳，并使用近邻选择形成对齐样本。
- 当前 writer 定义了双 RGB、七维机器人 state、七维 `action` 代理标签和 task，并写入 LeRobot 数据集及 sidecar 诊断文件。
- 已提供的 `openpi.tar` 对应 OpenPI commit `15a9616a00943ada6c20a0f158e3adb39df2ccac`；该版本作为当前 π0.5 接入分析基线。
- AutoDL 实际已跑通的 OpenPI HEAD 为 `215abfb217dbac7d5f1273282331b9b1866c0479`；它是上述分析基线的直接子提交，上游变更仅为 DROID 文档拼写。当前部署 manifest 记录实际 HEAD `215abfb...`。
- `scripts/sim/` 已提供 Isaac Sim 6.1.0 policy-free episode 入口、公开 URDF importer 调用和 episode 完整性检查；普通 Python 配置/episode 契约测试共 3 项通过。
- 已实现锁定 OpenPI commit 的 PiPER π0.5 server transform、客户端协议、七维 action contract、不执行动作的 inference check 和 MuJoCo receding-horizon runner；相关纯 Python 测试已通过。
- 资产无关的 π0.5 workflow smoke 已实际通过：本地发送两路 `240×320` RGB、七维 state 和 prompt，经 WebSocket/SSH 隧道调用云端官方 `pi05_base`，收到有限的 `(50,7)` action chunk；`execution_permitted=false` 且未尝试执行动作。首次冷启动往返为 `32425.782 ms`；随后三次为 `553.866 / 364.601 / 322.799 ms`，平均 `413.755 ms`。这只是传输加推理延迟，不是控制周期验证；证据摘要见 `assets/manifests/pi05_workflow_smoke_v0.1.json`。
- AgileX 官方 `agx_arm_sim` commit `f8cd8b1...` 及 `agx_arm_urdf` 子模块 `1ee5659...` 已绑定；PiPER-X+D435 URDF、USD、转换脚本和相机支架/网格 hash 已记录。
- 派生 MuJoCo 模型已实测 `nq=15`/`nu=7`，六轴 + 两指顺序、±0.05 m 夹爪、mimic equality、home 静止、joint1 tracking 及全局/夹爪中部 D435 非常量渲染通过。
- PiPER-X controller/adapter sanity 已逐维实测通过：六轴 action index 与同名关节一一对应，正向 `0.1 rad` 探针产生同向响应，夹爪 `0.05 m=open / 0 m=closed` 且两指对称；相同低层路径的 scripted F1 再次完成 reach、pick 和约 `0.08005 m` lift，无 slip/drop/非预期碰撞。该证据只排除当前 MuJoCo 低层路径无法抓取，不验证 π0.5；摘要见 `assets/manifests/piper_x_controller_sanity_v0.1.json`。
- 锁定 OpenPI `15a9616...` 的 Piper transform 离线数值审计已通过：joint1–6 的负/零/正 probe 经 delta、q01/q99 normalization、32 维 padding、反归一化和 absolute restore 后保持方向并恢复原目标；第七维不参与 delta，closure 增大对应夹爪 actuator 米数减小。当前只使用 transport-only smoke stats，因此状态为 `transport_only_math_pass`，不证明真实数据分布或策略 action convention；摘要见 `assets/manifests/piper_transform_alignment_smoke_v0.1.json`。

上述每项仅支持其明示的证据边界。MuJoCo 脚本抓取、控制审计、推理 smoke 和 LIBERO zero-shot rollout 已有运行证据；真机相机/CAN/LeRobot 完整写盘、Isaac Sim 启动和 Piper USD 导入仍未在本工作区现场验证。

## Reported Evidence Pending Repository Record

项目成员报告 AutoDL 上连续 20 次 `pi05_base` inference 成功，`policy_infer_ms≈57.7`、client mean `≈91.9 ms`、P99 `≈100 ms`。该 20 次的原始时延日志尚未入库，因此仍只作报告证据；闭环行为则以下方已保存的实际 rollout 为准。

数采设计说明记录了 Piper CAN/固件、SDK 加载和 LeRobot smoke test 等队友测试结果。当前仓库尚缺少对应命令、环境清单、日志摘要或数据 manifest，因此这些结果暂不归入 `Verified in Repository`。

外部 `agx_datacollect_ws` 的静态代码显示了单相机 MCAP、TCP/夹爪最近邻对齐和七维末端增量标签管线，但其顶层缺少 Git 版本记录，当前目录中也没有原始 bag、统计量或转换产物，因此只作为历史参考证据。

## In Progress

- 真实 PiPER-X 作为只读遥操手柄的本地 MuJoCo 示范采集 MVP 已实现：复用 `pyAgxArm` 反馈、LIBERO-derived PiPER-X environment、双 RGB 和七维绝对 target；提供配置化 mapping/限位/低通、独立 read/control/render 节拍、热键、LeRobot 单 episode 写盘和 read/apply/render mean/P95。纯 Python 回归、锁定 MuJoCo 双相机/七维 action smoke，以及 LeRobot 0.3.3 + CPU-only PyTorch 2.7.1 的两帧双 RGB/Parquet/video/manifest 写盘 smoke 已通过；`pyAgxArm` 仍仅在系统 Python，且本机尚未发现 CAN 适配器/真机，因此真机跟随、一条真机遥操 episode 现场写盘及 50–100 ms 视觉反馈仍为 In Progress。

- Zero-shot grasp tendency harness 已接入 policy runner，并完成 seed 7、10 s、`execution_horizon=10` 的 LIBERO-Spatial 实际 rollout。EE–object 距离从 `0.15435 m` 降到 `0.11878 m`，但夹爪在距物体 `0.14862 m` 时已开始闭合，最终未进入 pre-grasp、未指尖接触、未抬升。结论是“弱接近趋势但时序/控制不正确”，不是 grasp success。
- π0.5–PiPER-X 闭环失败诊断：controller/adapter sanity 已通过，本次 rollout 中所有执行记录均为 `projection.changed=false`，可排除投影/急停阻止抓取。当前主要可疑项是 scripted calibration normalization 分布太窄及 `pi05_base` 未学习 PiPER-X embodiment；`execution_horizon=10` 的 0.5 s 开环前缀会放大过早闭夹，但不是根因。
- 首个完整 LIBERO-Spatial task 已接入 PiPER-X MuJoCo composition：`Pick the akita black bowl from table center and place it on the plate`。使用官方 commit `8f1084e...` 的 BDDL、arena、木柜、炉灶、双黑碗、盘子、cookies、ramekin、agentview/frontview 和 region ranges；机器人替换为 PiPER-X。已修正并回归锁定 LIBERO 0.1.0 的桌腿位置、地板/墙面材质、桌面摩擦、legacy NumPy placement 采样顺序、物体边界与 z-offset；D435 腕部 RGB 也由 `camera_link` 修正为官方 xacro 中的 `camera_color_optical_frame`。本地 21 项相关回归测试通过，并已完成上述 policy rollout。
- FragileManip-Bench v0.1：PiPER-X 模型审计与单次 scripted F1 smoke 已完成；下一步是用 PiPER-X episode 重算 normalization。

- 将已有真机数采代码纳入可维护的数据规范和验证流程。
- 完成 LIBERO-derived PiPER-X 单任务控制闭环：保留 arena/furniture/camera/placement，接入 fragile-like target，并采集第一批 embodiment-adaptation 示范。
- LIBERO-derived cookies-box-to-plate 首任务已配置 12 条纯几何 variation，先用前 3 条做人工双视角验收。工作区收缩后，第 1、2 条已分别完成抓取、约 0.106 m 抬升和稳定放置；第 3 条完成抓取与抬升但放置后侧翻，已将未验证的正向腕部 yaw 收敛为已验证的姿态，待项目成员只复测第 3 条。旧的 10-episode bowl diagnostic pilot 为 0/10，旧的 contact/force 分支版本为 2/3，均拒绝进入训练和 normalization。当前版本的 force/contact 只做被动记录，不干预动作；不得表述为训练数据集已验证。
- 从实际用于微调的全部 embodiment-adaptation 训练数据计算 normalization，先跑通微调与 MuJoCo 闭环；正式 split、validation 和 safety stress-test 后移。
- 以 `configs/experiment_v0.1.example.yaml`、Episode Schema v0.1 和 Evaluation Schema v0.1 建立仿真实验记录契约；runner/validator 已实现，尚未在 Isaac Sim 中运行验证。
- 在 Isaac Sim 6.1.0 目标机准备并验证 Piper 场景，生成第一份 policy-free instrumented episode；π0.5 inference 暂不在该步骤实现。
- 用真实 Piper 数据对当前 MuJoCo π0.5 observation/action transform 做跨域核对。
- 从锁定的公开 Piper 仿真任务迁移并验证可由队友直接复现的 Isaac Sim 6.1.0 Piper X 简单抓取场景包。
- 建立实验 manifest、视频记录与第一版评测输入输出约定。
- 清理编辑器缓存、机器路径和生成产物的 Git 边界。

## TBD

- 官方 PiPER-X 资产与用户真机的计量级几何/动力学差异；benchmark-defined 执行器、gravcomp 与工作位没有真机辨识证据。
- D435 仿真 fovy 与真机内参等价性、F2–F5 实现、RoboCasa 资产选择、matched pair 生成器与物理/视觉随机化。
- Canonical 首任务已固定为 LIBERO-Spatial 黑碗到盘子；尚待确定后续 fragile / fragile-like object 列表、重定义 predicate 阈值、示范来源和每个 split 的 episode 数量。
- Stress-test 各维度的等级、取值范围、单因素/联合组合规则，以及“脆弱参数”和“接触风险”的可执行定义。
- 真实易碎对象资产及材料阈值校准；当前 LIBERO 黑碗只是刚体 embodiment-adaptation 载体，不代表已实现易碎物理。

- 主臂 joint/gripper control target 通过当前 `pyAgxArm` 采集链路接入的具体接口、SDK/固件版本、时间对齐、写盘兼容性，以及真实 action 的最终定义。
- 当前 zero-shot 结论在更多 seed、物体初始位姿和相机扰动下的重复性；真机映射仍未确认。
- 用真实 Piper episode stats 重跑七维 transform 审计，并以真实控制命令/反馈确认训练 action convention；当前仅完成 transport-only 数学路径验证，`[:7]` 本身只负责去除内部 padding，不作为风险项。
- 外部相机深度数据是否进入训练 observation，及其标定、单位和有效区域。
- 两相机内外参、时间同步精度、掉帧处理和 episode 时间语义。
- 夹爪闭合量标定、力反馈语义以及固件和 SDK 的最终兼容组合。
- Isaac Sim 6.1.0 的实际 build 与安装证据、physics backend、最终 Piper X 资产 commit/import 结果、关节/夹爪驱动、控制接口和传感器配置。
- action chunk 的执行长度、控制频率、重推理条件和异常处理。
- 易碎物体属性的物理建模、损伤判定、指标定义和阈值。
- 同机运行 Isaac Sim 与 π0.5 时的 GPU 分配、显存余量、推理端口、进程启动顺序和可持续控制频率；未来真机与推理分布在不同机器时的通信方式和延迟约束。

## Known Risks

- 当前 `action[t]` 是后一条对齐从臂状态代理，不是已观测到的控制命令；两者可能因控制和机械延迟而不一致。
- 历史 `agx_ws` 的七维末端增量 action、当前采集器的下一帧关节状态代理和未来主臂控制目标语义不同；混用其 checkpoint、统计量或执行映射会造成训练与推理契约错误。
- 跳过的逻辑采样 tick 会在当前 LeRobot 帧序号时间轴中被压缩，可能改变动作时序语义。
- 当前采集只保存两路 RGB；D435i 的深度能力尚未进入数据格式。
- 相机 SDK、Piper 固件和 `pyAgxArm` 选项需要按实际版本重新核对。
- 中断或异常可能留下不完整数据集，尚无明确的完成标记与恢复约定。
- 当前本地提交曾纳入大型 `.vscode` 数据库；即使工作区停止跟踪，该对象仍可能存在于尚未推送的提交历史中。
- 历史 AGX workspace 与 AgileX 当前公开 Piper X Xacro 的夹爪 joint/mimic 结构不同；若资产来源和实际 USD DOF 顺序没有绑定到 config，夹爪 target 会映射错误。

## Next Milestones

1. **PiPER-X 单任务数据闭环**：在锁定 LIBERO 场景中跑通 PiPER-X、双 RGB、七维 action、稳定目标物与完整 pick-and-place 示范录制。
2. **Embodiment normalization**：使用实际微调训练数据，经锁定 Piper transform 计算 state/action normalization，并完成 transform round-trip 与 batch smoke test。
3. **π0.5 PiPER-X 控制微调**：绑定 dataset/transform/normalization/checkpoint，先 inference-only 校验，再运行 MuJoCo closed loop，目标是证明模型已学会 PiPER-X 抓放控制。
4. **Benchmark 与易碎研究（后续）**：在控制闭环稳定后，再固定正式 split/validation，实现 fragile predicate、损伤指标和 safety stress-test。
