"""MuJoCo environment assembled from the pinned official PiPER-X URDF."""
from pathlib import Path
import copy
import hashlib
import importlib.metadata
import json
import subprocess
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.optimize import least_squares
import yaml
from .adapters import ObservationAdapter, ScriptedActionAdapter
from .libero_task import compose_first_libero_task, reset_first_libero_task
from .piper_x_asset import compile_piper_x_mjcf


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_configs(directory, object_config=None):
    docs = {name: yaml.safe_load((Path(directory) / f"{name}.yaml").read_text())
            for name in ("benchmark", "robot", "object", "tasks")}
    diagnostic_path = Path(directory) / "grasp_diagnostic.yaml"
    if diagnostic_path.is_file():
        docs["grasp_diagnostic"] = yaml.safe_load(diagnostic_path.read_text(encoding="utf-8"))
    if object_config is not None:
        docs["object"] = yaml.safe_load(Path(object_config).read_text(encoding="utf-8"))
    for doc in docs.values():
        if doc.get("provenance") not in {"source-derived", "benchmark-defined", "randomized"}:
            raise ValueError("每份配置必须声明参数来源")
    c = docs["benchmark"]["values"]
    for key in ("physics_dt_s", "settle_s", "phase_s", "hold_s", "lift_m", "lift_success_m",
                "reach_tolerance_m", "slip_relative_m", "drop_height_m", "place_tolerance_m",
                "gripper_probe_s", "ik_tolerance_m", "contact_epsilon_N"):
        if not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f"{key} 必须为正有限数")
    for key in ("control_steps", "render_every_steps", "ik_max_nfev"):
        if type(c[key]) is not int or c[key] <= 0:
            raise ValueError(f"{key} 必须为正整数")
    if len(c["resolution_hw"]) != 2 or any(type(x) is not int or x <= 0 or x % 2 for x in c["resolution_hw"]):
        raise ValueError("视频分辨率必须为两个正偶数")
    for key in ("force_thresholds_N", "impulse_thresholds_Ns"):
        if not c[key] or any(not np.isfinite(x) or x <= 0 for x in c[key]):
            raise ValueError("阈值扫描必须包含正有限值")
    for key in ("grasp_offset_m", "transport_offset_m"):
        if np.asarray(c[key]).shape != (3,) or not np.isfinite(c[key]).all():
            raise ValueError(f"{key} 必须为有限三维向量")
    if "grasp_diagnostic" in docs:
        diagnostic = docs["grasp_diagnostic"]["values"]
        for key in (
            "approach_progress_m", "pregrasp_distance_m", "alignment_tolerance_deg",
            "close_start_closure", "close_near_object_distance_m", "contact_epsilon_N",
            "lift_attempt_m", "stable_lift_m", "stable_lift_hold_s", "trace_period_s",
        ):
            if not np.isfinite(diagnostic.get(key)) or diagnostic[key] <= 0:
                raise ValueError(f"grasp_diagnostic.{key} 必须为正有限数")
        if diagnostic["close_start_closure"] > 1:
            raise ValueError("grasp_diagnostic.close_start_closure 必须在 (0,1]")
        desired = np.asarray(diagnostic.get("desired_tool_z_world"), dtype=np.float64)
        if desired.shape != (3,) or not np.isfinite(desired).all() or np.linalg.norm(desired) <= 0:
            raise ValueError("grasp_diagnostic.desired_tool_z_world 必须是非零有限三维向量")
    return docs


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def asset_tree_hash(directory):
    """相对路径与文件内容共同确定资产身份。"""
    h = hashlib.sha256()
    for p in sorted(Path(directory).rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(directory)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def absolute_assets(root, directory, meshdir=""):
    for node in root.findall(".//*[@file]"):
        prefix = meshdir if node.tag == "mesh" else ""
        node.set("file", str((directory / prefix / node.get("file")).resolve()))


def _configure_piper_x_mjcf(root, robot, base_position):
    """Add explicit MuJoCo-only controls without changing official link inertials."""
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("integrator", "implicitfast")
    option.set("cone", "elliptic")
    option.set("impratio", "10")

    worldbody = root.find("worldbody")
    base = ET.Element("body", name="base_link", pos=" ".join(map(str, base_position)))
    for child in list(worldbody):
        worldbody.remove(child)
        base.append(child)
    worldbody.append(base)
    ET.SubElement(worldbody, "light", name="robot_light", mode="targetbodycom",
                  target="link6", pos="0 0 2")
    for body in base.findall(".//body"):
        body.set("gravcomp", str(robot["body_gravcomp"]))

    joint_ranges = {}
    for joint in root.findall(".//joint"):
        name = joint.get("name")
        if name in robot["arm_joints"] + robot["finger_joints"]:
            joint.set("frictionloss", str(robot["joint_frictionloss"] if name in robot["arm_joints"] else 0.0))
            joint.set("armature", str(robot["joint_armature"] if name in robot["arm_joints"] else 0.0))
            joint_ranges[name] = joint.get("range")
    expected = set(robot["arm_joints"] + robot["finger_joints"])
    if set(joint_ranges) != expected:
        raise ValueError(f"PiPER-X compiled joint set changed: {set(joint_ranges)}")

    # The official aperture mimic (finger1=+0.5*aperture, finger2=-0.5*aperture)
    # is represented by a single finger actuator and this exact equality.
    equality = ET.SubElement(root, "equality")
    ET.SubElement(equality, "joint", joint1=robot["finger_joints"][1],
                  joint2=robot["finger_joints"][0], polycoef="0 -1 0 0 0")

    actuator = ET.SubElement(root, "actuator")
    for index, name in enumerate(robot["actuators"]):
        joint = robot["arm_joints"][index] if index < 6 else robot["finger_joints"][0]
        force = "-100 100" if index < 6 else "-10 10"
        ET.SubElement(
            actuator,
            "position",
            name=name,
            joint=joint,
            kp=str(robot["actuator_kp"][index]),
            kv=str(robot["actuator_kv"][index]),
            ctrlrange=joint_ranges[joint],
            forcerange=force,
        )

    home = robot["home_qpos"]
    if len(home) != 8:
        raise ValueError("PiPER-X home_qpos must contain six arm and two finger joints")
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(keyframe, "key", name=robot["home_key"],
                  qpos=" ".join(map(str, home)),
                  ctrl=" ".join(map(str, [*home[:6], home[6]])))

    # Adjacent official collision meshes can overlap at their shared joint.  Only
    # direct kinematic neighbours are excluded; non-adjacent self-contact remains.
    contact = ET.SubElement(root, "contact")
    pairs = [("base_link", "link1"), *[(f"link{i}", f"link{i + 1}") for i in range(1, 6)],
             ("link6", robot["finger_bodies"][0]), ("link6", robot["finger_bodies"][1])]
    for first, second in pairs:
        ET.SubElement(contact, "exclude", body1=first, body2=second)

    for geom in base.findall(".//geom"):
        if geom.get("group") == "1":
            mesh_name = geom.get("mesh", "").lower()
            if "d435" in mesh_name:
                # The imported D435 OBJ has no usable material assignment. A
                # near-black override made the complete sensor read as one large
                # featureless block in policy images, so retain the official
                # geometry but use a neutral housing colour.
                geom.set("rgba", "0.42 0.44 0.47 1")
            elif "realsense" in mesh_name:
                geom.set("rgba", "0.58 0.60 0.63 1")
            elif "gripper" in mesh_name:
                geom.set("rgba", "0.46 0.48 0.51 1")
            else:
                geom.set("rgba", "0.72 0.75 0.82 1")
        else:
            # The official visual DAE geometry is rendered; collision STL remains
            # physical but invisible so the diagnostic image is not a doubled shell.
            geom.set("rgba", "0.25 0.27 0.30 0")
    return root


class MujocoQualificationEnv:
    def __init__(self, docs, robot_assets):
        self.docs = docs
        self.cfg = docs["benchmark"]["values"]
        self.robot = docs["robot"]["values"]
        self.obj = docs["object"]["values"]
        robot_assets = Path(robot_assets).resolve()
        if (self.robot.get("source_format") != "agx_arm_sim_combined_urdf"
                or self.robot.get("model") != "piper_x"):
            raise ValueError("当前 benchmark 只接受锁定的官方 PiPER-X + D435 资产")
        distribution = importlib.metadata.distribution(self.obj["package"])
        if distribution.version != self.obj["version"]:
            raise ValueError("robosuite 资产版本不符")
        suite = Path(distribution.locate_file("robosuite"))
        root, robot_evidence = compile_piper_x_mjcf(robot_assets, self.robot)
        root = _configure_piper_x_mjcf(root, self.robot, self.obj["robot_base_position_m"])
        root.find("option").set("timestep", str(self.cfg["physics_dt_s"]))
        libero_evidence = None
        if self.obj.get("scene_type") == "libero_task":
            libero_checkout = PROJECT_ROOT / self.obj["libero_checkout"]
            libero_evidence = compose_first_libero_task(root, libero_checkout, self.obj)
            self.policy_camera = self.obj["policy_camera"]
        else:
            for field in ("arena_xml", "object_xml"):
                path = suite / self.obj[field]
                if not path.is_file():
                    raise FileNotFoundError(f"场景资产不存在：{path}")
                part = ET.parse(path).getroot()
                absolute_assets(part, path.parent)
                root.find("asset").extend(copy.deepcopy(list(part.find("asset"))))
                if field == "arena_xml":
                    root.find("worldbody").extend(copy.deepcopy(list(part.find("worldbody"))))
                else:
                    body = copy.deepcopy(part.find("worldbody/body/body"))
                    body.set("name", self.obj["object_body"])
                    body.set("pos", " ".join(map(str, self.obj["object_position_m"])))
                    ET.SubElement(body, "freejoint", name="target_free")
                    root.find("worldbody").append(body)
            global_camera = self.cfg["global_camera"]
            if global_camera["name"] != self.cfg["camera"]:
                raise ValueError("global_camera.name 必须与 camera 引用一致")
            ET.SubElement(
                root.find("worldbody"),
                "camera",
                name=global_camera["name"],
                pos=" ".join(map(str, global_camera["position_m"])),
                xyaxes=" ".join(map(str, global_camera["xyaxes"])),
                fovy=str(global_camera["fovy_deg"]),
            )
            self.policy_camera = self.cfg["camera"]
        wrist_camera = self.cfg["wrist_camera"]
        wrist_parent = root.find(f".//body[@name='{wrist_camera['parent_body']}']")
        if wrist_parent is None:
            raise ValueError("腕部相机 parent_body 不存在")
        ET.SubElement(
            wrist_parent,
            "camera",
            name=wrist_camera["name"],
            pos=" ".join(map(str, robot_evidence["camera_position_link6_m"])),
            xyaxes=" ".join(map(str, robot_evidence["camera_xyaxes_link6"])),
            fovy=str(wrist_camera["fovy_deg"]),
        )
        visual = root.find("visual")
        if visual is None:
            visual = ET.SubElement(root, "visual")
        visual_map = visual.find("map")
        if visual_map is None:
            visual_map = ET.SubElement(visual, "map")
        visual_map.set("znear", str(self.cfg["render_znear_extent_fraction"]))
        if self.obj.get("scene_type") != "libero_task":
            # 新自由物体扩展 keyframe；保留原始机器人 home qpos 与 ctrl。
            for key in root.findall("keyframe/key"):
                key.set("qpos", key.get("qpos") + " " + " ".join(map(str, self.obj["object_position_m"])) + " 1 0 0 0")
        self.scene_xml = ET.tostring(root, encoding="unicode")
        self.model = mujoco.MjModel.from_xml_string(self.scene_xml)
        self.data = mujoco.MjData(self.model)
        self.joint_names = self.robot["arm_joints"] + self.robot["finger_joints"]
        self.jids = np.array([self.model.joint(x).id for x in self.joint_names])
        self.qadr = self.model.jnt_qposadr[self.jids]
        self.vadr = self.model.jnt_dofadr[self.jids]
        actual = [self.model.actuator(i).name for i in range(self.model.nu)]
        if actual != self.robot["actuators"]:
            raise ValueError("执行器顺序改变，拒绝运行")
        self.target_id = self.model.body(self.obj["object_body"]).id
        self.finger_ids = [self.model.body(x).id for x in self.robot["finger_bodies"]]
        self.wrist_id = self.model.body(self.robot["wrist_body"]).id
        self.distal_geoms = [np.flatnonzero((self.model.geom_bodyid == bid) & (self.model.geom_contype != 0))[-1]
                             for bid in self.finger_ids]
        base_id = self.model.body("base_link").id
        self.robot_bodies = {
            body_id for body_id in range(self.model.nbody)
            if body_id == base_id or self._has_ancestor(body_id, base_id)
        }
        self.renderer = None
        self.observation_adapter = ObservationAdapter()
        self.action_adapter = ScriptedActionAdapter()
        self.provenance = {**robot_evidence, "robot_model": self.robot["model"],
                           "robot_license": self.robot["license"],
                           "robosuite_version": distribution.version}
        if libero_evidence is not None:
            self.provenance.update(libero_evidence)
        else:
            arena_path = suite / self.obj["arena_xml"]
            self.provenance.update({
                "arena_provider": "robosuite",
                "arena_xml_sha256": sha256(arena_path),
                "object_xml_sha256": sha256(suite / self.obj["object_xml"]),
                "robosuite_assets_sha256": asset_tree_hash(suite / "models/assets"),
            })

    def _has_ancestor(self, body_id, ancestor_id):
        parent = int(self.model.body_parentid[body_id])
        while parent > 0:
            if parent == ancestor_id:
                return True
            parent = int(self.model.body_parentid[parent])
        return False

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key(self.robot["home_key"]).id)
        if self.obj.get("scene_type") == "libero_task":
            reset_first_libero_task(self.model, self.data, seed)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(round(self.cfg["settle_s"] / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.initial_object = self.data.xpos[self.target_id].copy()
        self.data.time = 0.0
        return self.observe()

    def grasp_position(self, data=None):
        data = self.data if data is None else data
        return data.geom_xpos[self.distal_geoms].mean(axis=0)

    def render(self, camera):
        if self.renderer is None:
            h, w = self.cfg["resolution_hw"]
            self.renderer = mujoco.Renderer(self.model, height=h, width=w)
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def observe(self):
        images = {
            "base_rgb": self.render(self.policy_camera),
            "wrist_rgb": self.render(self.cfg["wrist_camera"]["name"]),
        }
        return self.observation_adapter.encode(self, images)

    def step_physics(self, action):
        self.data.ctrl[:] = self.action_adapter.decode(self, action)
        # 先 forward 并采集当前状态的接触力，再积分相同 dt，避免状态和力错位。
        mujoco.mj_forward(self.model, self.data)
        contacts = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            contacts.append({"geom1": int(c.geom1), "geom2": int(c.geom2),
                             "body1": int(self.model.geom_bodyid[c.geom1]),
                             "body2": int(self.model.geom_bodyid[c.geom2]),
                             "normal_force_N": max(0.0, float(force[0])),
                             "contact_frame_wrench": force.tolist()})
        state = {"time_s": float(self.data.time), "object_position_m": self.data.xpos[self.target_id].copy(),
                 "grasp_position_m": self.grasp_position().copy(),
                 "wrist_rotation": self.data.xmat[self.wrist_id].reshape(3, 3).copy(),
                 "gripper_position_m": float(self.data.qpos[self.qadr[-2]]),
                 "gripper_closure": float(np.clip(
                     (self.robot["gripper_actuator_open_m"] - self.data.qpos[self.qadr[-2]])
                     / (self.robot["gripper_actuator_open_m"] - self.robot["gripper_actuator_closed_m"]),
                     0.0,
                     1.0,
                 )),
                 "qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy(), "contacts": contacts}
        mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("仿真状态出现非有限值")
        return state

    def inverse_kinematics(self, target, initial, desired_rotation=None):
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        bounds = self.model.jnt_range[self.jids[:6]]
        margin = self.cfg["joint_limit_margin"]
        desired = np.asarray(
            self.cfg["wrist_rotation_world"] if desired_rotation is None else desired_rotation,
            dtype=np.float64,
        )  # benchmark-defined：默认腕部 z 朝下。
        if desired.shape != (3, 3) or not np.isfinite(desired).all():
            raise ValueError("IK desired_rotation 必须是有限 3x3 矩阵")
        def residual(q):
            scratch.qpos[self.qadr[:6]] = q
            mujoco.mj_forward(self.model, scratch)
            rotation = scratch.xmat[self.wrist_id].reshape(3, 3)
            return np.r_[self.grasp_position(scratch) - target,
                         self.cfg["ik_orientation_weight"] * (rotation - desired).ravel()]
        result = least_squares(residual, np.clip(initial, bounds[:, 0] + margin, bounds[:, 1] - margin),
                               bounds=(bounds[:, 0] + margin, bounds[:, 1] - margin),
                               max_nfev=self.cfg["ik_max_nfev"])
        if np.linalg.norm(residual(result.x)[:3]) > self.cfg["ik_tolerance_m"]:
            raise RuntimeError("脚本 IK 目标不可达；不篡改模型或物体状态")
        return result.x

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
