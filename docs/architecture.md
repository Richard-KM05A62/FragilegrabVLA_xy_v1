# FragileGrabVLA 系统架构

## 1. 范围

FragileGrabVLA 是围绕 Piper X 与 π0.5 建设的易碎物体抓取研究工作区。项目覆盖真实机器人数据采集、数据检查与转换、Isaac Sim 仿真、π0.5 训练和推理、动作执行、实验记录与评测。

当前只接入 π0.5，并保留其原生 action chunk 范式。项目不建设多策略注册表或通用 VLA backend。易碎物体的安全信号、约束方法和干预方式仍为开放研究问题，当前架构不包含 Safety Layer。

## 2. 设计原则

- **事实可追踪**：文档区分 Confirmed、Verified、In Progress、Candidate 和 TBD；实验结论必须关联配置、软件版本和产物。
- **配置与实现分离**：设备、路径、频率、模型、场景和评测参数进入版本化配置模板或本地配置，不散落在高层文档和核心逻辑中。
- **运行时边界明确**：真机采集、Isaac Sim、π0.5 和评测共享数据语义，但保留各自的设备依赖和运行环境。
- **数据契约优先**：真实数据和仿真数据使用同一套逻辑 episode 描述，并明确来源、时间、观测、动作、任务和结果语义。
- **按验证结果演进**：先保留可运行实现；只有出现第二个真实使用者并确认公共行为后才提取抽象。

## 3. 端到端数据流

```text
Piper X + cameras                   Isaac Sim scene
        │                                 │
        ▼                                 ▼
real data collection              simulation recording
        └──────────────┬──────────────────┘
                       ▼
             versioned episode data
                       ▼
            validation and transforms
                       ▼
             π0.5 training/inference
                       ▼
                  action chunk
                       ▼
          Isaac Sim or Piper X execution
                       ▼
        observations, events, video, result
                       ▼
                    evaluation
```

近期的最小闭环是：Isaac Sim 产生一组与 π0.5 数据映射一致的 observation，π0.5 输出 action chunk，执行侧按已记录的控制约定执行全部或部分 chunk，环境返回新 observation，直至任务结束。

## 4. 组件边界

### 4.1 真实数据采集

负责读取外部相机、腕部相机和 Piper X 状态，建立时间关系，形成 episode，并保存采集诊断信息。当前实现是只读采集链路，不负责机械臂运动、夹爪控制、复位和标定。

### 4.2 数据层

负责 episode schema、manifest、质量检查、数据版本和 OpenPI 所需的数据转换。原始数据、派生数据和训练输入必须能够追溯；转换不能覆盖原始采集证据。

### 4.3 π0.5

负责把 Piper X observation 映射为 π0.5 输入，并把 π0.5 action chunk 映射回已确认的 Piper X 或 Isaac Sim 控制语义。实现应以项目锁定的 OpenPI 版本为依据，Piper 专用映射在验证前保持 TBD。

### 4.4 Isaac Sim

负责提供可移交的 Piper X 简单抓取场景、传感器观测、动作执行、episode 重置和运行记录。场景包应由队友放入已确认版本的 Isaac Sim 后即可复现，不依赖开发者个人绝对路径。

### 4.5 执行

负责接收 π0.5 action chunk、遵照明确的坐标系和控制周期执行动作，并返回下一次观测。动作维度、单位、绝对或增量语义、chunk 执行长度、重推理条件和异常行为均须配置或记录；未确认项为 TBD。

### 4.6 评测

负责读取一次或多次 episode 的任务结果、轨迹、接触信息、视频和实验元数据，计算可复现指标。评测可以研究易碎性或损伤风险，但不能被表述为已经存在的安全控制机制。

## 5. 跨组件契约

每个 episode 至少需要描述：

- 数据来源：真实机器人或 Isaac Sim；
- schema、采集器、场景、机器人资产和软件版本；
- task 标识与自然语言描述；
- observation 字段、单位、坐标系和时间信息；
- action 字段、单位、坐标系、绝对或增量语义及其来源；
- episode 边界、重置条件、终止原因和任务结果；
- 关联的视频、诊断日志、配置与评测结果。

具体 feature 名称和形状见 [数据格式](data_format.md)。这些内容在 Piper X 与 OpenPI 的映射验证完成前可以标记为 Candidate 或 TBD，不能只靠目录或类名暗示已经确认。

## 6. 运行环境

真实采集电脑、π0.5 推理设备和 Isaac Sim 设备可以是不同机器。仓库负责维护可共享的配置模板、版本信息、数据契约和移交说明；机器路径、设备序列号、账号、权重和本地缓存不进入 Git。

第一阶段计划在队友的同一台电脑上运行 Isaac Sim 6.1.0 与 π0.5 推理，两者保持独立进程和依赖环境，通过本机连接传递 observation 与 action chunk。该部署不需要跨机器时钟映射，但仍需在执行端记录 observation、请求、响应和 action dispatch 的同机单调时间；GPU 分配、显存余量、端口和进程启动顺序须按目标机实测并写入运行配置或 manifest。

这里的连接是 loopback 进程间通信，不是跨电脑远程访问。优先沿用锁定 OpenPI 版本的 WebSocket server/client 协议，客户端连接 `127.0.0.1`；该版本官方 `serve_policy.py` 默认监听 `0.0.0.0`，目标机须记录实际监听地址，若只允许本机访问则通过最小启动修改或主机网络规则限制。若目标机实测证明 Isaac Sim 与 OpenPI 能在同一 Python 进程稳定共存，才评估进程内调用，不能为了省略本机 transport 混合两套相互冲突的依赖环境。

## 7. 仓库演进规则

| 位置 | 当前职责 | 演进方式 |
| --- | --- | --- |
| `docs/` | 项目架构、状态、数据、π0.5、仿真、评测和研究问题 | 随确认事实和实现同步更新 |
| `scripts/data_collect/` | 当前 Piper X 真实数采实现与实现说明 | 现场验证前保持原位；出现稳定包边界后再迁移 |
| `scripts/sim/` | Isaac Sim 6.1.0 policy-free episode、Piper URDF 导入和完整性检查 | 在目标 Isaac 环境运行验证后再扩展 π0.5 消费循环 |
| `configs/` | 可共享配置模板与 schema | 对应模块字段确认后添加，个人配置进入 `configs/local/` |
| `datasets/manifests/` | 外部数据版本和来源记录 | 首个可共享数据版本出现时添加 manifest |
| `assets/manifests/` | Piper X、Isaac Sim 等外部资产记录 | 资产版本确认后添加 manifest |
| `src/` | 未来稳定的可复用运行代码 | 第一个实际模块实现时创建，不预建空包 |
| `evaluation/` | 未来评测协议和指标版本 | 第一版指标获批并实现时创建 |

现阶段保留 `scripts/data_collect/` 中已有数采实现；仿真入口直接位于 `scripts/sim/`，没有提取真机/仿真的通用 runtime 层。新的源代码目录应随着一个实际可运行模块建立，并配套配置或运行说明；不为未来设想创建空包或空接口。外部 robot/scene 资产通过 manifest 记录来源、版本、许可证、校验值和安装位置，不直接复制大文件进仓库。

当前进展与未决项见 [项目状态](status.md)。
