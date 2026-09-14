# 数据格式

## 目标

数据层需要同时承载真实机器人示范、Isaac Sim rollout、π0.5 输入准备和评测证据。真实与仿真数据不必共用同一采集实现，但应共享可以比较的 episode 语义。

## 当前采集 schema

当前真机采集代码定义：

| 字段 | 当前语义 | 状态 |
| --- | --- | --- |
| `observation.images.base` | D435i 外部视角 RGB | Implemented，现场验证待补 |
| `observation.images.wrist` | DaBai DC1 腕部视角 RGB | Implemented，现场验证待补 |
| `observation.state` | 六个关节角（rad）与归一化夹爪闭合量 | Implemented，夹爪标定 TBD |
| `action` | 下一条对齐从臂 state 的代理 | Implemented，最终 action 定义 TBD |
| `task` | episode 自然语言任务 | Implemented |

当前代码没有保存 depth 训练 feature。主机逻辑时间、各数据源时间偏差、原始夹爪反馈和部分 SDK 时间戳保存在 sidecar，而 LeRobot 训练时间轴由帧序号构造。

## Episode 逻辑契约

每个可发布 episode 应能由 manifest 回答以下问题：

```text
identity      dataset/schema/episode id and version
provenance    real or sim, collector commit, source versions
task          task id, instruction, initial condition, success definition
observation   feature names, shapes, dtypes, units, frames, camera roles
action        feature names, units, frames, absolute/delta semantics, source
time          clock domains, nominal rate, observed timestamps, dropped samples
termination   completed/interrupted/failed and termination reason
artifacts     data, sidecars, videos, logs, configs and evaluation references
```

字段名称、格式版本和存储布局在实现 schema validator 前为 Candidate。不得在缺少迁移说明的情况下改变已有字段含义。

## Episode Schema v0.1（In Progress）

v0.1 定义逻辑语义，不要求迁移或替换当前 LeRobot 数据。一次 episode 可以继续由 LeRobot feature、`piper_capture_manifest.json` 和 alignment sidecar 组成；manifest 通过 storage reference 指向实际数据，未来 Isaac Sim recorder 也使用相同逻辑字段。

### Core manifest

| 字段 | 要求 | 说明 |
| --- | --- | --- |
| `episode_schema_version` | Required | 固定为 `fragilegrab.episode/0.1`，与当前 capture manifest 的数字版本分开命名。 |
| `episode_id` | Required | 稳定标识；生成规则 TBD。 |
| `experiment_config` | Required | 有效配置的引用和校验值。 |
| `source.kind` | Required | `real` 或 `simulation`。 |
| `robot.name` | Required | 当前为 `piper_x`。 |
| `task.id` / `task.instruction` | Required | 成功判据通过独立 reference 记录；未知时为 TBD。 |
| `object` | Required for evaluable runs | 真实物体 specimen 或仿真 asset 的稳定引用。 |
| `provenance` | Required | 仓库 commit、dirty 状态、软件版本以及设备或资产版本。 |
| `clocks` | Required | 时钟名称、scope、单位及可用的时钟映射。 |
| `streams` | Required | 只登记实际存在的数据流及 storage reference。 |
| `termination` | Required | `complete`、`interrupted` 或 `failed`，并记录原因。 |
| `result` | Required | task 与 safety 独立；未评定值使用 `null`。 |
| `artifacts` | Optional | 视频、日志、配置和评测结果引用。 |

每个 stream descriptor 至少记录 semantic name、storage reference、dtype/shape、单位、坐标系和 clock。真实机器人没有 exact object pose 或 contact impulse 时直接省略相应 stream，不能生成占位测量。

### Sparse timeline

observation、策略查询、chunk 执行和物理事件可能处于不同频率。v0.1 使用稀疏 timeline record，不要求给每个 LeRobot frame 增加大量空字段：

| record type | 最小语义 |
| --- | --- |
| `observation` | timestamp、frame reference、各来源样本时间或 skew。 |
| `policy_query` | query id、实际使用的 observation reference、请求开始/完成时间、完整 `predicted_action_chunk` 和 action semantics reference。 |
| `execution` | timestamp、query id、chunk index、requested action、executed action 和 dispatch status。 |
| `contact` | event id、开始/结束/峰值时间、参与物体、来源以及可用 raw measurement。 |
| `phase_event` | phase、timestamp、online/manual/postprocess 来源、方法版本和 evidence reference。 |

同一条 latency 计算中的时间戳必须处于同一时钟域。推理服务位于另一台机器时，策略 request send 和 response receive 使用执行端时钟记录；未建立时钟映射前，不直接相减服务端时间。

### Action identity

| 字段 | 语义 |
| --- | --- |
| `training_action` | 数据集监督标签；当前采集值必须保留 `source=next_observation_proxy`。 |
| `predicted_action_chunk` | π0.5 output transform 后、处于环境控制语义中的完整 action chunk。 |
| `requested_action` | 执行侧从 chunk 中选择的某一项。 |
| `executed_action` | 经执行侧修改后实际提交给低层控制接口的命令。 |
| `robot_state` | 反馈测量；用于描述实际物理响应，不能由 `executed_action` 代替。 |

项目成员已确认主臂控制目标可取得，但当前采集脚本尚未记录。接入后它应成为带独立时间戳和 `source=leader_command`（最终命名 TBD）的 training action 候选；在现场验证和兼容方案完成前，不覆盖现有代理 `action`。

## π0.5 转换边界

原始 episode 应保存机器人和传感器的可解释物理量。OpenPI 所需的重命名、动作转换、状态和动作 padding、图像预处理、归一化与反归一化属于版本化 transform，不应回写原始数据。

Piper X 的最终 transform 需要明确：

- π0.5 看到的图像角色及缺失图像处理；
- proprioception 顺序、单位和夹爪定义；
- action 是绝对关节目标、关节增量、末端增量或其他语义；
- state/action 的有效维度与模型内部 padding；
- 训练统计量的来源及其与 checkpoint 的绑定方式；
- 推理输出到 Isaac Sim/Piper 控制命令的逆变换。

以上均须通过锁定 OpenPI 版本的代码和实际数据验证；未验证内容为 TBD。

## 仿真与评测扩展

Isaac Sim episode 还应记录场景、机器人资产、随机种子、相机参数、物体物理属性、接触事件和 reset/termination 结果。易碎属性与损伤标签属于实验 schema，定义和阈值目前为 Candidate，不构成安全控制接口。

Isaac Sim 6.1.0 的优先 instrumentation profile 包含双相机角色、Piper articulation state、实际提交的控制命令、目标物体状态，以及官方接口可获得的 contact/joint effort 原始读数。physics backend、传感器更新周期、contact threshold、Piper joint mapping、物体属性和成功判据仍为 TBD；配置不得补写默认实验值。

当前 `scripts/sim/run_instrumented_episode.py` 已按该 profile 实现一种存储：`episode_manifest.json` 保存 core manifest，`timeline.jsonl` 交错保存 execution/observation record，两路 `streams/<camera-role>/*.npy` 保存原始 HWC `uint8` RGB/RGBA，`videos/` 保存人工复核视频。它是 Episode Schema v0.1 的一个具体 writer，不要求现有 LeRobot 真机数据迁移到相同目录。运行器把 JointStateSensor 返回的 revolute/prismatic position 分别标为 rad/m，并另外记录 `stage_meters_per_unit`；object pose 和 linear velocity 保留 stage length unit 及换算因子。

[Isaac Sim 6.1.0 Contact Sensor schema](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/omniverse_usd/sensor_schema.html) 将 threshold/force 的线性量纲定义为 `kg * stage_length_unit / s^2`；因此 runner 不在非米制 stage 上把 raw contact value 直接标成 N，也不自行推断 raw impulse 单位。后续评测转换必须结合 episode 中实际 `stage_meters_per_unit` 和锁定 backend 的官方定义。
