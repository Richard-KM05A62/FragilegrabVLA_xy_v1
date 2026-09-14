# 配置目录

本目录保存可提交、可复用的配置模板和 schema。只有对应运行模块出现并确认字段后，才新增具体配置。

机器专属路径、设备序列号、CAN 通道、IP、端口、checkpoint 路径和个人实验参数放入被忽略的 `configs/local/`，不得提交密钥或隐私信息。一次实验使用的有效配置副本应随运行产物保存，并在 manifest 中记录其来源和校验值。

## 当前模板

- `experiment_v0.1.example.yaml`：统一实验语义的第一版模板，当前优先描述 Isaac Sim 6.1.0 简单抓取 instrumentation。模板中的 `TBD` 是未确认输入，不能在运行或评测时静默替换为默认值。`scripts/sim/run_instrumented_episode.py --preflight` 会拒绝 backend、scene、DOF、相机、物理步、动作和视频等运行必需字段中的 `TBD`；尚未定义的 success/evaluation reference 可以继续保留，episode result 相应为 `null`。

同一 schema 也用于真实 Piper X。真实运行将 `source.kind` 设为 `real`，省略 `source.simulator`，并在本地配置中提供采集机器所需参数。仿真与真机共享 task、robot、recording、episode 和 evaluation 语义，但不伪造另一环境不存在的信号。

当前模板关闭 policy，因为第一步只验证仿真 instrumentation。启用 π0.5 的实验需要另外记录实际 OpenPI commit、checkpoint id、transform id、normalization stats id、chunk execution rule 和 replan rule；这些字段在推理适配前不作为空配置写入。

仿真 config 的 `scripted_commands` 是固定基线输入，只使用实际 USD DOF 名称和 position target。运行器在启动后读取资产的实际 DOF 顺序、limits、drive type、stiffness 和 damping，发现不匹配或 target 越界即停止；这些数值不在模板中预填。
