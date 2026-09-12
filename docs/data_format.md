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
