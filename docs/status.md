# FragileGrabVLA 项目状态

本文件按证据强度记录状态。`Confirmed` 表示项目决定；`Verified` 表示仓库中已有可检查的运行或静态验证证据；`In Progress` 表示正在实现或验证；未知内容标记为 `TBD`。

## Confirmed

- 项目定位为 Piper X + π0.5 易碎物体抓取研究工作区，覆盖真机数据、仿真、策略和评测。
- 当前只计划使用 π0.5，并保留原生 action chunk。
- π0.5 第一阶段策略实验使用 OpenPI 官方 `pi05_base` 基础 checkpoint，不做微调，目的是测量基础模型在 Piper X Isaac Sim 场景中的零微调表现；后续微调必须作为独立实验记录。
- 零微调基线使用锁定 OpenPI commit 中 `Pi0Config(pi05=True)` 和 `sample_actions` 的默认设置，不添加基于结果调出的模型采样参数。Piper 字段映射、padding、归一化和仿真控制映射仍是运行所必需的显式接口配置。
- 外部 `agx_ws` 属于模型切换前的 OpenVLA 历史实现；当前项目不继续采用 OpenVLA，只将其中的 Piper X 数据与执行接口作为核对依据。
- 外部 `agx_datacollect_ws` 属于同一代 OpenVLA 历史数采管线；它只作为旧数据来源和设备语义参考，不替代当前双相机 LeRobot 采集器。
- Isaac Sim 是近期仿真环境，第一项任务是简单抓取。
- 真实数据采集使用一个外部相机和一个腕部相机；外部相机具备深度能力。
- 项目成员确认主臂关节和夹爪控制目标可以取得；当前采集器尚未接入或验证该数据流。
- 当前 Isaac Sim 目标版本为 NVIDIA Isaac Sim 6.1.0，仿真 instrumentation 适配优先于 π0.5 推理接入。
- 仿真场景不从空白 USD 自行设计；优先迁移 AgileX College 的 Piper 方块堆叠任务并使用 AgileX 官方 Piper X USD，DynamicVLA 的 Piper pick/place 与 DOM scene/object 包作为第二候选。上游版本均低于 6.1.0，采用前必须记录来源并完成 6.1.0 迁移验证。
- 第一阶段在队友同一台电脑上运行 Isaac Sim 6.1.0 和 π0.5 推理，使用独立进程与依赖环境并通过本机连接交换 observation/action chunk。
- 仿真实验需要设置物体的易碎相关属性，生成抓取视频，并通过项目评测体系计算指标。
- Episode 数据契约需要为不可变原始数据、版本化数据处理、多个时钟域与对齐证据、π0.5 原生 action chunk 的执行/重规划记录，以及未来新增的感知、控制和安全评测信号保留向后兼容的扩展位置；未知信号不以占位数值伪造。
- 易碎物体安全机制整体为 **Open Research Question / TBD**；项目没有已确认或已实现的 Safety Layer。
- 代码、配置模板和项目事实进入 Git；数据、视频、权重、checkpoint、日志和生成产物存放在外部或本地。

## Verified in Repository

- `scripts/data_collect/aligned_capture.py` 和 `run_aligned_capture.py` 可被 Python 语法解析。
- 静态检查显示，当前数采代码只读取 D435i RGB、DaBai DC1 RGB 和 Piper X 关节/夹爪反馈，不包含机械臂使能、运动、夹爪命令、复位或标定调用。
- 当前代码以主机 `time.monotonic_ns()` 为三路数据打时间戳，并使用近邻选择形成对齐样本。
- 当前 writer 定义了双 RGB、七维机器人 state、七维 `action` 代理标签和 task，并写入 LeRobot 数据集及 sidecar 诊断文件。
- 已提供的 `openpi.tar` 对应 OpenPI commit `15a9616a00943ada6c20a0f158e3adb39df2ccac`；该版本作为当前 π0.5 接入分析基线。
- `scripts/sim/` 已提供 Isaac Sim 6.1.0 policy-free episode 入口、公开 URDF importer 调用和 episode 完整性检查；普通 Python 配置/episode 契约测试共 3 项通过。
- `assets/manifests/piper_x_agx_history_candidate_v0.1.yaml` 已记录历史 `agx_datacollect_ws` Piper X URDF/Xacro 的 source commit、文件校验值、许可证和观察到的 DOF 名称。

以上是代码和归档的静态证据，不代表相机、CAN、LeRobot 写盘、Isaac Sim 启动、Piper USD 导入或完整仿真 episode 已在本工作区现场运行验证。

## Reported Evidence Pending Repository Record

数采设计说明记录了 Piper CAN/固件、SDK 加载和 LeRobot smoke test 等队友测试结果。当前仓库尚缺少对应命令、环境清单、日志摘要或数据 manifest，因此这些结果暂不归入 `Verified in Repository`。

外部 `agx_datacollect_ws` 的静态代码显示了单相机 MCAP、TCP/夹爪最近邻对齐和七维末端增量标签管线，但其顶层缺少 Git 版本记录，当前目录中也没有原始 bag、统计量或转换产物，因此只作为历史参考证据。

## In Progress

- 将已有真机数采代码纳入可维护的数据规范和验证流程。
- 以 `configs/experiment_v0.1.example.yaml`、Episode Schema v0.1 和 Evaluation Schema v0.1 建立仿真实验记录契约；runner/validator 已实现，尚未在 Isaac Sim 中运行验证。
- 在 Isaac Sim 6.1.0 目标机准备并验证 Piper 场景，生成第一份 policy-free instrumented episode；π0.5 inference 暂不在该步骤实现。
- 明确真实数据到 OpenPI π0.5 的 Piper observation/action transform。
- 从锁定的公开 Piper 仿真任务迁移并验证可由队友直接复现的 Isaac Sim 6.1.0 Piper X 简单抓取场景包。
- 建立实验 manifest、视频记录与第一版评测输入输出约定。
- 清理编辑器缓存、机器路径和生成产物的 Git 边界。

## TBD

- 主臂 joint/gripper control target 通过当前 `pyAgxArm` 采集链路接入的具体接口、SDK/固件版本、时间对齐、写盘兼容性，以及真实 action 的最终定义。
- 七维 state/action 与 OpenPI π0.5 的精确映射、归一化统计和推理后处理。
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

1. **仿真 instrumentation 基线**：把已实现 runner 放到 Isaac Sim 6.1.0 目标机，锁定 Piper X 资产与 physics backend，生成并验证第一份不依赖 π0.5 的简单抓取 episode，包含配置、robot/object state、动作、接触、终止和视频。
2. **数据基线**：记录一次真实完整 episode 的环境、固件、SDK、配置、manifest 和质量检查结果，并接入可取得的主臂控制目标以确认 action 与时间语义。
3. **π0.5 最小闭环**：依据锁定 OpenPI 版本实现并验证 Piper transform，使 observation → inference → action chunk → execution → next observation 可重复运行。
4. **第一版评测**：对若干固定场景生成 episode 与视频，审核并实现第一组基础指标；易碎损伤指标先以 Candidate 实验定义管理。
