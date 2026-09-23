"""显式隔离仿真位置指令与尚未确认的 π0.5 语义。"""
import numpy as np


class ObservationAdapter:
    """原始仿真观测，不做 OpenPI 命名、padding 或归一化。"""

    def encode(self, env, images):
        if set(images) != {"base_rgb", "wrist_rgb"}:
            raise ValueError("观测必须同时包含 base_rgb 和 wrist_rgb")
        return {"images": {name: np.asarray(value).copy() for name, value in images.items()},
                "joint_names": env.joint_names,
                "joint_position": env.data.qpos[env.qadr].copy(),
                "joint_velocity": env.data.qvel[env.vadr].copy(),
                "simulation_time_s": float(env.data.time),
                "units": ["rad"] * 6 + ["m", "m"]}


class ScriptedActionAdapter:
    """输入为七维、绝对位置、模型 actuator 顺序；夹爪为 joint7 米数。"""

    def decode(self, env, action):
        action = np.asarray(action, dtype=float)
        limits = env.model.actuator_ctrlrange
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("脚本动作必须为有限的七维向量")
        if np.any(action < limits[:, 0]) or np.any(action > limits[:, 1]):
            raise ValueError("动作越过模型 ctrlrange；不静默裁剪")
        return action.copy()


class Pi05Adapter:
    """保留 action chunk；未验证的映射必须显式拒绝执行。"""

    validation_points = (
        "锁定 OpenPI commit 与 pi05_base checkpoint 身份",
        "核对外部/腕部图像角色、缺失图像、尺寸、dtype 与预处理",
        "核对 state 顺序、单位、有效维度及 padding",
        "核对绝对/增量、关节/末端、坐标系和夹爪定义",
        "绑定归一化统计和 checkpoint，验证正逆 transform",
        "验证完整 chunk 形状、执行长度、重规划及取消记录",
    )

    def encode_observation(self, observation):
        raise NotImplementedError("π0.5 observation 映射为 TBD：" + "；".join(self.validation_points))

    def decode_action_chunk(self, chunk):
        raise NotImplementedError("π0.5 action 语义为 TBD；禁止当作脚本关节目标执行")
