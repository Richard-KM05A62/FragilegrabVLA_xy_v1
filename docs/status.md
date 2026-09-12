# Project Status

## Current Goal

将 π0.5 部署到 Isaac Sim，建立从 observation、推理、action chunk 执行到 next observation 的最小闭环操作链路。

## Confirmed

- 研究课题是易碎物品抓取。
- 当前仅计划使用 π0.5，不引入 OpenVLA 或多模型抽象。
- π0.5 原生输出 action chunk，第一版应保留该范式。
- Isaac Sim 是当前仿真与 robot 环境。
- GitHub 是代码、文档和长期项目事实来源。
- 数据、视频、checkpoint、模型权重、日志和 `runs/` 等不进入 Git，保存在本地或外部存储。
- 用户主要负责架构与设计；队友主要负责 Isaac Sim/robot 环境实现。

## In Progress

- 明确 π0.5 与 Isaac Sim 之间的 observation 和 action 映射。
- 确认最小闭环所需的机器人、传感器、动作空间与运行配置。
- 设计 action chunk 的执行节奏、重新观测和重新推理条件。

本节表示当前工作方向，不表示相关实现或验证已经完成。

## Open Research Questions

- 易碎物品的“安全”应由哪些物理量和任务结果定义？
- 是否需要力、触觉、滑移或视觉形变等反馈？
- 安全机制应在 action chunk 执行前、执行中，还是两者结合地介入？
- 如何构建可重复的损伤判定、风险指标和实验基线？
- 安全目标与任务成功率、速度之间如何权衡？

当前安全性研究方案整体标记为 **Open Research Question / TBD**。项目尚无已确认或已实现的 Safety Layer。

## Candidate Ideas

以下仅是待研究候选，不是架构承诺：

- 接触力或夹持力约束
- 速度、加速度或 jerk 限制
- action chunk 执行前检查或在线修正
- 触觉反馈与滑移检测
- 物体损伤或风险预测
- safety critic 或约束策略

## Next Milestone

在 Isaac Sim 中完成并记录一次可复现的 π0.5 最小闭环运行：获得 observation，完成 input transform 与 inference，按既定策略执行 action chunk，并取得 next observation。里程碑记录应包含环境与模型版本、配置、动作/观测约定、运行步骤和验证结果。
