# 人工工具与旧代码

本目录不是正式数采或自动离线测试入口。以下脚本仅移动/重命名，保留原行为，整理时没有执行它们。

| 新位置 | 原根目录文件 | 行为与风险 |
|---|---|---|
| `hardware/check_can_connection.py` | `test.py` | 顶层直接连接 CAN，打印连接/通信状态；不是纯离线导入。 |
| `hardware/enable_arm.py` | `test_enable.py` | 顶层连接并调用 `robot.enable()`，保持使能；人工授权后才运行。 |
| `hardware/leader_follow.py` | `test_leader.py` | 顶层连接并使能，用于主从跟随；可能产生实际运动，运行前必须通知用户。 |
| `legacy/collect_data.py` | `collect_data.py` | 早期默认摄像头样例，会使能机械臂；输出 images/joints/meta，不是当前 LeRobot 双相机格式。 |

这些人工脚本不再使用 `test*.py` 名称，避免默认测试发现将它们作为自动测试执行。仍不要直接导入它们。

正式只读双相机数采应在项目根目录使用 `python -m capture.run_aligned_capture`。自动离线测试使用 `python -m unittest discover -s tests`。
