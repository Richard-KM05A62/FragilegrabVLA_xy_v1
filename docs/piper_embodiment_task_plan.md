# PiPER-X embodiment-adaptation 数据任务规划

## 边界与来源

本计划用于 PiPER-X + π0.5 embodiment adaptation。任务必须源自本地锁定的官方 LIBERO
commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`。每个任务保留官方 arena、
furniture、camera、asset 和 placement 语义，只做可追溯的 PiPER-X robot/base 替换、可达
区收缩、物体初始位置/yaw、robot reset pose、predicate 和 instruction 变体。不拼接
非官方的统一场景，不把结果称为未修改的 LIBERO benchmark，统一称为
`LIBERO-derived PiPER-X task`。

当前只有第一个官方 LIBERO-Spatial 黑碗到盘子场景已完成 PiPER-X composition 和
本地运行验证。其他任务为锁定的设计配额，必须逐场景接入后才能执行，不得记为
Verified。

## 最终 100 条配额

| ID | 核心模板 | 官方场景来源 | 总数 | Scripted | Teleop |
|---|---|---|---:|---:|---:|
| T1 | 混合刚体 simple pick | Spatial / Object 官方场景 | 20 | 12 | 8 |
| T2 | bowl → plate | LIBERO-Spatial 第 1 任务 | 14 | 8 | 6 |
| T3 | package → basket | LIBERO-90 Living Room Scene 3/5/6 | 16 | 10 | 6 |
| T4 | package → tray | LIBERO-90 Living Room Scene 3/5/6 | 12 | 7 | 5 |
| T5 | mug → left/right plate | LIBERO-90 Study Scene 1 | 10 | 5 | 5 |
| T6 | object relative to plate | LIBERO Object / Spatial | 14 | 8 | 6 |
| T7 | book → caddy | LIBERO-90 Study Scene 1 | 6 | 2 | 4 |
| T8 | two-object sequence | LIBERO-10 Living Room Scene 1 | 8 | 2 | 6 |
|  | **合计** |  | **100** | **54** | **46** |

每类应覆盖左/中/右、近/远、不同 PiPER reset pose 和 object yaw。其中至少 17 个几何
配置同时采 scripted 和 teleop，用于检查两种示教分布差异。不纳入需要抽屉、
柜门、微波炉、炉具开关、酒架、堆叠、煎锅或初始物体不稳定支撑的任务。

## 首批 20 条 provisional norm 数据

首批不再用单场景坐标扰动充当多样性。改为 5 个官方 LIBERO 任务，每个任务
4 个官方 placement seed，共 20 条：

| 条数 | 官方 BDDL 任务 |
|---:|---|
| 4 | Spatial: `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate` |
| 4 | Object: `pick_up_the_ketchup_and_place_it_in_the_basket` |
| 4 | Object: `pick_up_the_cream_cheese_and_place_it_in_the_basket` |
| 4 | LIBERO-90: `LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray` |
| 4 | LIBERO-90: `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate` |

每条使用官方 BDDL 的 scene、objects、regions、instruction 和 placement sampling；只替换机器人并记录
PiPER-X 的 state/action。不用 prompt 改名伪造新任务，不用肉眼难以区分的小幅坐标改动凑数。

当前只有第 1 个 Spatial 任务已接入 PiPER-X composition；后 4 个任务尚未接入，因此
首批 20 条暂不可执行。已删除旧的单场景 norm20 采集配置和专用运行入口，不能用历史
单场景数据替代这里定义的五任务数据集。

## 分阶段执行

1. 先接入并单独验证上述 5 个官方任务的 PiPER-X scene composition。
2. 每个任务先生成 1 条预检，检查双视角、任务语义、可达性和时序。
3. 5 条通过后，再以每任务 4 seeds 生成完整 20 条新 run。
4. 人工检查视频，但不删除原始 episode；训练候选集另行记录。
5. 先运行项目校准统计检查数值；官方 OpenPI `compute_norm_stats.py` 必须等 Piper-specific
   training data transform/loader 对锁定 commit 实读通过后再作为训练统计。

## 历史数据状态

`gentle_demo_pilot.yaml` 和已生成的 4 条 multitask run 仅保留为 pipeline smoke。其中
pick-and-place prompt/goal 语义不一致，relocate 未抓取，container/place 存在目标盘漂移；
因此它们不进入本轮 norm 或训练集。
