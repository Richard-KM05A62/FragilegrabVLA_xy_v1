# 真实数据采集

## 目的与边界

真实数据采集为 Piper X 示范学习提供可追踪的 episode。采集端负责传感器和机器人状态读取、时间记录、样本组织、写盘与诊断，不负责 π0.5 训练，也不在未审核的情况下控制机械臂。

当前实现位于：

- `scripts/data_collect/aligned_capture.py`：读取器、时间缓存、对齐和 LeRobot writer；
- `scripts/data_collect/run_aligned_capture.py`：单次采集入口；
- `scripts/data_collect/DATA_COLLECTION_DESIGN.md`：实现级说明。

## 当前实现快照

静态检查确认当前代码读取：

- D435i 外部相机 RGB；
- DaBai DC1 腕部相机 RGB；
- Piper X 六个关节角和夹爪反馈；
- episode 的自然语言 task。

三路生产线程用主机单调时钟记录到达时间，采集循环围绕逻辑 tick 选择最近样本。代码中的采样率、允许偏差、启动延迟、缓存窗口、图像 profile 和夹爪开口值都是**当前实现参数**，不是项目全局标准；现场验证后应迁移到配置并记录在每次采集 manifest 中。

当前 `action` 使用后一条对齐从臂状态作为代理标签。它不是已读取的主臂命令，训练前必须保留并检查 `action_source`，不能把两者视为等价。

项目成员已确认主臂关节和夹爪控制目标可以取得。AgileX 官方 [`piper_sdk` 接口文档](https://github.com/agilexrobotics/piper_sdk/blob/master/asserts/V2/INTERFACE_V2.MD)提供了读取主臂 joint/gripper control message 的接口与原始单位；当前采集器使用的是 `pyAgxArm`，尚未确认实际运行版本中应通过哪个公开接口接入、如何与三路 host monotonic 时间对齐。因此该能力为 **Confirmed, integration TBD**，当前 `action` 语义保持不变。

## 历史 `agx_datacollect_ws` 参考

项目成员确认，外部 `agx_datacollect_ws` 是模型切换前的 OpenVLA 数据采集工作区。静态检查显示，它录制单路 D435i RGB、Piper TCP 位姿和夹爪开度到 ROS 2 MCAP，再以图像时间为锚点做最近邻配对，输出中心裁剪 RGB、绝对 TCP 位姿和七维 `[delta_xyz, delta_rpy, gripper_binary]` 标签。

该工作区的采集、转换、统计和 CAN listener 文件与历史 `agx_ws` 中的对应文件逐文件一致。其 action 是相邻已观测 TCP 位姿的差分，不是主臂或从臂控制命令；录制列表也不包含 `/control/*` 或关节反馈。它可用于核对历史数据来源、Piper 单位和 ROS/CAN 行为，但不替代当前双相机 LeRobot 采集器，也不直接定义 π0.5 action。

外部工作区顶层没有 Git 版本记录，当前目录中也未发现 bag、`dataset_statistics.json` 或转换后的数组产物。将其结论用于正式数据迁移前，仍需取得原始数据、配置和实际依赖版本，并验证旧 action 与同索引图像的因果对齐方向。

## 尚未完成的能力

- 外部相机 depth 的采集、对齐、标定和数据字段；
- 双相机与 Piper CAN 的整条现场验证记录；
- 主臂实际命令或从臂目标指令与当前采集器的接入、时间对齐和写盘验证；
- 相机内外参、夹爪量程和设备版本的结构化记录；
- 中断 episode 的完成标记、隔离和恢复流程；
- 多 episode 会话、数据检查报告和数据版本发布流程。

## 一次采集应留下的证据

在把采集链路标记为 Verified 前，至少保留：

1. 采集器 Git commit 与未提交改动状态；
2. 操作系统、Python、LeRobot、相机 SDK、Piper SDK 和固件版本；
3. 使用的配置副本，但不包含设备密钥或个人路径；
4. episode manifest、数据 schema、帧数、时长和终止原因；
5. 三路时间偏差、掉帧和错误摘要；
6. 可人工检查的低分辨率预览或视频引用；
7. 数据质量检查结果和已知异常。

## 修改规则

修改数采逻辑前，应先核对实际 SDK 版本的官方接口，尤其是资源所有权、固件选项、设备生命周期和 LeRobot 数据约定。相机、CAN、同步、action、episode 和序列化变更应分别说明影响，并以小范围现场测试验证。机器路径和设备标识进入 `configs/local/` 或环境变量，不进入提交。
