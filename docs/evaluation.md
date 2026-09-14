# 评测设计

## 目标

评测用于比较不同数据、checkpoint、场景参数和执行设置下的抓取表现。评测模块读取完整 episode 和仿真事件，不参与实时动作控制。

## 第一版输入与输出

输入至少包含：

- 实验和 episode manifest；
- task、初始条件和终止原因；
- observation/action 时间序列；
- Isaac Sim 任务状态和接触事件；
- 抓取视频或视频引用。

输出应为机器可读的逐 episode 指标、聚合统计和可追踪到原始运行的报告。具体文件格式在第一份仿真 episode 产生后确认。

当前 `scripts/sim/validate_episode.py` 只做 v0.1 结构和引用完整性检查：要求 complete termination、双相机数组和视频、有效 robot/contact reading、object state、至少一条 execution，以及单调时间线和 config 校验值。它不使用任务或易碎阈值，因此不是正式 evaluation，也不生成成功或安全结论。

## Evaluation Schema v0.1（In Progress）

逐 episode 输出至少保留以下独立语义：

```text
evaluation_schema_version: fragilegrab.evaluation/0.1
episode_id
protocol_version
validity.status / validity.reasons
task_success
task_subresults
safety_evaluation_status
safety_violations
severity_measurements
execution_measurements
failure_stage
successful_but_unsafe
```

`task_success` 使用 `true`、`false` 或 `null`；`null` 表示没有按照已定义协议评定。`safety_violations` 使用 list 或 `null`：空 list 只表示已经执行有效 safety evaluation 且未发现事件，`null` 表示未评定或所需信号不可用。

`successful_but_unsafe` 是派生值：

```text
task_success is true
and safety_evaluation_status == evaluated
and safety_violations is non-empty
```

任一输入未评定时，结果为 `null`，不得默认为 `false`。

`contact_to_replan_latency` 的 v0.1 语义为：

```text
next_policy_query_started_ns - first_contact_start_ns
```

只有两个事件处于同一时钟域时才计算。为区分查询调度、推理和新 chunk 生效时间，还应保留 query completed 与新 chunk 首次 dispatch 的 raw timestamp；这些是独立测量，不折叠成一个综合延迟。

## 指标优先级

### Candidate baseline metrics

- 任务成功率；
- episode 时长或完成时间；
- 是否产生有效视频与完整运行记录；
- 多次运行的均值、分布和失败原因计数。

项目已经确认需要通过评测体系产出指标，但以上具体指标和成功判据仍需结合第一项仿真任务审核后确认。

### Candidate motion/contact metrics

- 抓取过程中目标物体的位姿变化、滑移或掉落；
- 末端、关节或夹爪轨迹的长度和峰值变化；
- 接触次数、接触持续时间、冲量或力相关统计；
- action chunk 执行与重推理的次数和时间。

这些候选项只有在 Isaac Sim 对应信号的单位、采样方式和可靠性验证后才能升级为正式指标。

Isaac Sim 6.1.0 仿真应优先保留官方 physics sensor 返回的 simulation time、physics step、validity、contact force/raw impulse 和 joint effort。具体 signal、physics backend 和采样配置确认前，这些仍是 Candidate。`maximum_violation` 暂不进入 v0.1，因为它依赖尚未定义的 threshold 或归一化规则。

### Open research: fragility/damage

易碎损伤的定义、可观测信号、物理模型、阈值和与真机的对应关系均为 TBD。损伤率、最大容许接触或风险分数只能作为经过说明的实验指标，不能先验写成安全保证。

## 可重复性要求

每个聚合结果必须保留 episode 集合、配置版本、场景版本、checkpoint、随机种子、指标实现版本和无效运行处理规则。视频用于人工复核，但不能代替机器可读的终止状态和测量记录。
