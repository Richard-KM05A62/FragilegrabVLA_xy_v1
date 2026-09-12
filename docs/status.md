# FragileGrabVLA 项目状态

本文件按证据强度记录状态。`Confirmed` 表示项目决定；`Verified` 表示仓库中已有可检查的运行或静态验证证据；`In Progress` 表示正在实现或验证；未知内容标记为 `TBD`。

## Confirmed

- 项目定位为 Piper X + π0.5 易碎物体抓取研究工作区，覆盖真机数据、仿真、策略和评测。
- 当前只计划使用 π0.5，并保留原生 action chunk。
- Isaac Sim 是近期仿真环境，第一项任务是简单抓取。
- 真实数据采集使用一个外部相机和一个腕部相机；外部相机具备深度能力。
- 仿真实验需要设置物体的易碎相关属性，生成抓取视频，并通过项目评测体系计算指标。
- 易碎物体安全机制整体为 **Open Research Question / TBD**；项目没有已确认或已实现的 Safety Layer。
- 代码、配置模板和项目事实进入 Git；数据、视频、权重、checkpoint、日志和生成产物存放在外部或本地。

## Verified in Repository

- `scripts/data_collect/aligned_capture.py` 和 `run_aligned_capture.py` 可被 Python 语法解析。
- 静态检查显示，当前数采代码只读取 D435i RGB、DaBai DC1 RGB 和 Piper X 关节/夹爪反馈，不包含机械臂使能、运动、夹爪命令、复位或标定调用。
- 当前代码以主机 `time.monotonic_ns()` 为三路数据打时间戳，并使用近邻选择形成对齐样本。
- 当前 writer 定义了双 RGB、七维机器人 state、七维 `action` 代理标签和 task，并写入 LeRobot 数据集及 sidecar 诊断文件。
- 已提供的 `openpi.tar` 对应 OpenPI commit `15a9616a00943ada6c20a0f158e3adb39df2ccac`；该版本作为当前 π0.5 接入分析基线。

以上是代码和归档的静态证据，不代表相机、CAN、LeRobot 写盘或完整 episode 已在本工作区现场运行验证。

## Reported Evidence Pending Repository Record

数采设计说明记录了 Piper CAN/固件、SDK 加载和 LeRobot smoke test 等队友测试结果。当前仓库尚缺少对应命令、环境清单、日志摘要或数据 manifest，因此这些结果暂不归入 `Verified in Repository`。

## In Progress

- 将已有真机数采代码纳入可维护的数据规范和验证流程。
- 明确真实数据到 OpenPI π0.5 的 Piper observation/action transform。
- 定义可由队友直接复现的 Isaac Sim Piper X 简单抓取场景包。
- 建立实验 manifest、视频记录与第一版评测输入输出约定。
- 清理编辑器缓存、机器路径和生成产物的 Git 边界。

## TBD

- Piper X 主臂命令能否被可靠记录，以及真实 action 的最终定义。
- 七维 state/action 与 OpenPI π0.5 的精确映射、归一化统计和推理后处理。
- 外部相机深度数据是否进入训练 observation，及其标定、单位和有效区域。
- 两相机内外参、时间同步精度、掉帧处理和 episode 时间语义。
- 夹爪闭合量标定、力反馈语义以及固件和 SDK 的最终兼容组合。
- Isaac Sim 版本、Piper X 资产来源、关节/夹爪驱动、控制接口和传感器配置。
- action chunk 的执行长度、控制频率、重推理条件和异常处理。
- 易碎物体属性的物理建模、损伤判定、指标定义和阈值。
- 真机、Isaac Sim 与 π0.5 推理分布在不同机器时的通信方式和延迟约束。

## Known Risks

- 当前 `action[t]` 是后一条对齐从臂状态代理，不是已观测到的控制命令；两者可能因控制和机械延迟而不一致。
- 跳过的逻辑采样 tick 会在当前 LeRobot 帧序号时间轴中被压缩，可能改变动作时序语义。
- 当前采集只保存两路 RGB；D435i 的深度能力尚未进入数据格式。
- 相机 SDK、Piper 固件和 `pyAgxArm` 选项需要按实际版本重新核对。
- 中断或异常可能留下不完整数据集，尚无明确的完成标记与恢复约定。
- 当前本地提交曾纳入大型 `.vscode` 数据库；即使工作区停止跟踪，该对象仍可能存在于尚未推送的提交历史中。

## Next Milestones

1. **数据基线**：记录一次真实完整 episode 的环境、固件、SDK、配置、manifest 和质量检查结果，确认 action 与时间语义。
2. **仿真移交包**：锁定 Isaac Sim 与 Piper X 资产版本，提供简单抓取场景、配置、启动说明、验收记录和视频输出约定。
3. **π0.5 最小闭环**：依据锁定 OpenPI 版本实现并验证 Piper transform，使 observation → inference → action chunk → execution → next observation 可重复运行。
4. **第一版评测**：对若干固定场景生成 episode 与视频，审核并实现第一组基础指标；易碎损伤指标先以 Candidate 实验定义管理。
