"""可复现 episode、视频、模型审计与重复运行验证。"""
from pathlib import Path
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
import cv2
import mujoco
import numpy as np
from .environment import MujocoQualificationEnv, load_configs, sha256
from .evaluator import SafetyEvaluator
from .video import export_webm


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(x) for x in value]
    return value


def script_plan(env, task):
    """工程诊断脚本，读取 oracle 物体位置；不是 π0.5 baseline。"""
    c, m = env.cfg, env.model
    home = env.data.ctrl[:6].copy()
    opening = float(m.actuator_ctrlrange[-1, 1])
    closing = float(m.actuator_ctrlrange[-1, 0])
    start = env.initial_object + np.asarray(c["grasp_offset_m"])
    above = start + [0, 0, c["lift_m"]]
    qa = env.inverse_kinematics(above, home)
    qb = env.inverse_kinematics(start, qa)
    def command(q, grip):
        return np.r_[q, grip]
    plan = [("open", command(home, opening), c["phase_s"]),
            ("approach", command(qa, opening), c["phase_s"]),
            ("reach", command(qb, opening), c["phase_s"]),
            ("close", command(qb, closing), c["phase_s"]),
            ("pick", command(qa, closing), c["phase_s"]),
            ("hold", command(qa, closing), c["hold_s"] + c["phase_s"])]
    if task == "qualification":
        end = start + np.asarray(c["transport_offset_m"])
        qt = env.inverse_kinematics(end + [0, 0, c["lift_m"]], qa)
        qp = env.inverse_kinematics(end, qt)
        plan += [("transport", command(qt, closing), c["phase_s"]),
                 ("place", command(qp, closing), c["phase_s"]),
                 ("release", command(qp, opening), c["phase_s"]),
                 ("release", command(qt, opening), c["phase_s"]),
                 ("release", command(qt, opening), c["hold_s"])]
    return plan


def model_audit(env):
    """使用独立 MjData 验证 home/zero/驱动，不改变 episode 状态。"""
    # 审计使用已经组合的相同模型，目标物体保持自由；机器人从上游 home 重置。
    m, c = env.model, env.cfg
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key(env.robot["home_key"]).id)
    mujoco.mj_forward(m, d)
    home = d.qpos[env.qadr].copy()
    ranges = m.jnt_range[env.jids].copy()
    for _ in range(round(c["gripper_probe_s"] / m.opt.timestep)):
        mujoco.mj_step(m, d)
    drift = float(np.max(np.abs(d.qpos[env.qadr] - home)))
    d.ctrl[0] += c["joint_probe_delta_rad"]
    for _ in range(round(c["gripper_probe_s"] / m.opt.timestep)):
        mujoco.mj_step(m, d)
    error = float(abs(d.qpos[env.qadr[0]] - d.ctrl[0]))
    d.ctrl[-1] = m.actuator_ctrlrange[-1, 1]
    for _ in range(round(c["gripper_probe_s"] / m.opt.timestep)):
        mujoco.mj_step(m, d)
    opened = d.qpos[env.qadr[-2:]].copy()
    d.ctrl[-1] = m.actuator_ctrlrange[-1, 0]
    for _ in range(round(c["gripper_probe_s"] / m.opt.timestep)):
        mujoco.mj_step(m, d)
    closed = d.qpos[env.qadr[-2:]].copy()
    zero = mujoco.MjData(m)
    mujoco.mj_resetData(m, zero)
    mujoco.mj_forward(m, zero)
    zero_arm = zero.qpos[env.qadr].copy()
    zero_in_range = bool(np.all((zero_arm >= ranges[:, 0]) & (zero_arm <= ranges[:, 1])))
    return {"joint_names": env.joint_names, "joint_range": ranges.tolist(),
            "units": ["rad"] * 6 + ["m", "m"], "home_qpos": home.tolist(),
            "zero_qpos": zero_arm.tolist(), "zero_in_range": zero_in_range,
            "zero_finite": bool(np.isfinite(zero.qpos).all()), "zero_contacts": int(zero.ncon),
            "zero_interpretation": "零位仅检查正运动学与范围，不证明真机标定零位或无碰撞工作位",
            "static_max_drift": drift, "static_pass": drift <= c["static_tolerance_rad_m"],
            "joint1_tracking_error_rad": error, "control_pass": error <= c["tracking_tolerance_rad"],
            "gripper_open_m": opened.tolist(), "gripper_closed_m": closed.tolist(),
            "gripper_pass": bool(abs(opened[0] - m.actuator_ctrlrange[-1, 1]) <= c["gripper_tolerance_m"]
                                 and abs(opened.sum()) <= c["gripper_tolerance_m"]
                                 and np.max(np.abs(closed)) <= c["gripper_tolerance_m"]),
            "body_gravcomp": m.body_gravcomp.tolist(),
            "target_mass_kg": float(m.body_mass[env.target_id]),
            "target_inertia_kg_m2": m.body_inertia[env.target_id].tolist(),
            "target_geom_friction": m.geom_friction[m.geom_bodyid == env.target_id].tolist(),
            "actuator_ctrlrange": m.actuator_ctrlrange.tolist(),
            "actuator_forcerange": m.actuator_forcerange.tolist(),
            "camera": env.policy_camera, "camera_position_m": m.camera(env.policy_camera).pos.tolist(),
            "camera_quaternion_wxyz": m.camera(env.policy_camera).quat.tolist(),
            "camera_fovy_deg": float(m.camera(env.policy_camera).fovy[0]),
            "wrist_camera": c["wrist_camera"]["name"],
            "wrist_camera_position_m": m.camera(c["wrist_camera"]["name"]).pos.tolist(),
            "wrist_camera_quaternion_wxyz": m.camera(c["wrist_camera"]["name"]).quat.tolist(),
            "wrist_camera_fovy_deg": float(m.camera(c["wrist_camera"]["name"]).fovy[0]),
            "source_note": "连杆质量/惯量来自官方 URDF；位置执行器与 gravcomp 是 benchmark-defined，不证明真机动力学标定"}


def run_episode(config_dir, robot_assets, output, task="F1", seed=None, object_config=None,
                recording_options=None):
    docs = load_configs(config_dir, object_config=object_config)
    if task not in {"F1", "qualification"}:
        raise ValueError("F2–F5 仅为 TBD 规格，尚不能运行")
    cfg = docs["benchmark"]["values"]
    if recording_options is not None:
        allowed_recording_options = {
            "save_raw_rgb_frames", "compress_timeline",
            "record_full_physics_timeline", "generate_preview_video",
        }
        unknown = set(recording_options) - allowed_recording_options
        if unknown:
            raise ValueError(f"未知 recording option：{sorted(unknown)}")
        cfg.update(recording_options)
    if cfg["policy_enabled"]:
        raise ValueError("π0.5 映射尚未验证，禁止启用")
    if cfg["randomization_enabled"]:
        raise ValueError("随机化尚未实现，禁止静默忽略")
    seed = cfg["seed"] if seed is None else seed
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed 必须为 uint32 整数")
    cfg["seed"] = seed  # 保存 CLI 覆写后的实际配置。
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / ".incomplete").touch()
    env, video, wrist_video = None, None, None
    try:
        env = MujocoQualificationEnv(docs, robot_assets)
        observation = env.reset(seed)
        audit = model_audit(env)
        image = observation["images"]["base_rgb"]
        wrist_image = observation["images"]["wrist_rgb"]
        audit["render_nonconstant"] = bool(image.std() > 0)
        audit["wrist_render_nonconstant"] = bool(wrist_image.std() > 0)
        save_json(output / "model_audit.json", audit)
        save_json(output / "experiment_config.json", docs)
        (output / "scene.xml").write_text(env.scene_xml)
        cv2.imwrite(str(output / "reset.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output / "reset_wrist.png"), cv2.cvtColor(wrist_image, cv2.COLOR_RGB2BGR))
        dt = env.model.opt.timestep
        frame_stride = cfg["render_every_steps"]
        h, w = cfg["resolution_hw"]
        video = cv2.VideoWriter(str(output / "base_rgb.mp4"), cv2.VideoWriter_fourcc(*cfg["video_fourcc"]),
                                1 / (dt * frame_stride), (w, h))
        if not video.isOpened():
            raise RuntimeError("视频编码器未打开")
        wrist_video = cv2.VideoWriter(str(output / "wrist_rgb.mp4"), cv2.VideoWriter_fourcc(*cfg["video_fourcc"]),
                                      1 / (dt * frame_stride), (w, h))
        if not wrist_video.isOpened():
            raise RuntimeError("腕部视频编码器未打开")
        allowed = {
            tuple(sorted((env.model.body("base_link").id, env.model.body("table").id))),
            tuple(sorted(env.finger_ids)),
        }
        evaluator = SafetyEvaluator(cfg, env.initial_object, env.target_id, env.finger_ids,
                                    env.robot_bodies, allowed, task)
        trajectory_hash = hashlib.sha256()
        rgb_hashes = {"base_rgb": hashlib.sha256(), "wrist_rgb": hashlib.sha256()}
        steps, frames = 0, 0
        save_raw_rgb_frames = bool(cfg.get("save_raw_rgb_frames", True))
        compress_timeline = bool(cfg.get("compress_timeline", False))
        generate_preview_video = bool(cfg.get("generate_preview_video", True))
        record_full_physics_timeline = bool(cfg.get("record_full_physics_timeline", True))
        if save_raw_rgb_frames:
            (output / "streams/base_rgb").mkdir(parents=True)
            (output / "streams/wrist_rgb").mkdir(parents=True)
        timeline_name = "timeline.jsonl.gz" if compress_timeline else "timeline.jsonl"
        timeline_path = output / timeline_name
        timeline_open = gzip.open if compress_timeline else open
        with timeline_open(timeline_path, "wt", encoding="utf-8") as log:
            def record(row):
                log.write(json.dumps(native(row), ensure_ascii=False, allow_nan=False) + "\n")
            action = env.data.ctrl.copy()
            for phase, target, duration in script_plan(env, task):
                record({"record_type": "phase_event", "phase": phase, "simulation_time_s": float(env.data.time)})
                n = round(duration / dt)
                start_action = action.copy()
                for k in range(n):
                    if k % cfg["control_steps"] == 0:
                        # 插值按控制块末端取值，末块精确到达目标。
                        fraction = min(1.0, (k + cfg["control_steps"]) / n)
                        action = start_action + fraction * (target - start_action)
                        action = env.action_adapter.decode(env, action)
                        record({"record_type": "execution", "simulation_time_s": float(env.data.time),
                                "requested_action": action.tolist(), "executed_action": action.tolist(),
                                "source": "scripted_joint_position_targets", "phase": phase})
                    state = env.step_physics(action)
                    evaluator.update(state, phase, dt)
                    trajectory_hash.update(state["qpos"].tobytes())
                    trajectory_hash.update(state["qvel"].tobytes())
                    if record_full_physics_timeline:
                        record({"record_type": "contact", "episode_physics_step": steps, "phase": phase, **state})
                    if steps % frame_stride == 0:
                        mujoco.mj_forward(env.model, env.data)
                        obs = env.observe()
                        images = obs.pop("images")
                        image_refs = {}
                        for name, writer in (("base_rgb", video), ("wrist_rgb", wrist_video)):
                            rgb = images[name]
                            rgb_hashes[name].update(rgb.tobytes())
                            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                            if save_raw_rgb_frames:
                                path = f"streams/{name}/{frames:06d}.npy"
                                np.save(output / path, rgb)
                                image_refs[name] = path
                            else:
                                image_refs[name] = {"video": f"{name}.mp4", "frame_index": frames}
                        record({"record_type": "observation", "episode_physics_step": steps,
                                "image_refs": image_refs, **obs})
                        frames += 1
                    steps += 1
        video.release()
        video = None
        wrist_video.release()
        wrist_video = None
        capture = cv2.VideoCapture(str(output / "base_rgb.mp4"))
        decoded = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[:2] != (h, w):
                raise RuntimeError("视频解码尺寸不符")
            decoded += 1
        capture.release()
        if decoded != frames or frames == 0:
            raise RuntimeError("视频解码帧数不符")
        preview = None
        if generate_preview_video:
            preview = export_webm(output, frames, (h, w), 1 / (dt * frame_stride),
                                  cfg["preview_fourcc"])
        metrics = evaluator.result()
        save_json(output / "metrics.json", metrics)
        repo = Path(__file__).resolve().parents[3]
        def git(*args):
            result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
            return result.stdout.strip() if result.returncode == 0 else None
        code_hash = hashlib.sha256()
        for p in sorted(Path(__file__).parent.glob("*.py")):
            code_hash.update(p.name.encode()); code_hash.update(p.read_bytes())
        manifest = {"episode_schema_version": "fragilegrab.episode/0.1", "profile": "mujoco_qualification/0.1",
                    "episode_id": output.name, "source": {"kind": "simulation", "simulator": "mujoco"},
                    "robot": {"name": "piper_x", "source": "AgileX official pinned agx_arm_sim PiPER-X+D435 URDF/USD"},
                    "object": env.obj, "task": {"id": task, **docs["tasks"]["values"][task]},
                    "seed": seed, "provenance": {**env.provenance, "code_commit": git("rev-parse", "HEAD"),
                        "dirty": git("status", "--porcelain"), "module_sha256": code_hash.hexdigest(),
                        "python": platform.python_version(), "platform": platform.platform(),
                        "packages": {x: importlib.metadata.version(x) for x in ("mujoco", "robosuite", "numpy", "scipy", "opencv-python", "PyYAML")},
                        "MUJOCO_GL": os.environ.get("MUJOCO_GL")},
                    "experiment_config": {"storage_ref": "experiment_config.json", "sha256": sha256(output / "experiment_config.json")},
                    "action": {"source": "scripted_joint_position_targets", "semantics": "绝对位置：六轴 rad + joint7 m；按 actuator 顺序", "pi05_inference": False},
                    "clocks": {"physics_dt_s": dt, "scope": "episode simulation", "unit": "s"},
                    "streams": {"robot_state": {"names": env.joint_names, "units": ["rad"]*6+["m"]*2},
                                "object_state": {"frame": "world", "unit": "m"},
                                "contact": {"force_unit": "N", "impulse_unit": "N*s",
                                            "sample": "每物理步评测汇总" if not record_full_physics_timeline else "每物理步积分前",
                                            "raw_timeline_saved": record_full_physics_timeline},
                                "base_rgb": {"shape": [h, w, 3], "dtype": "uint8", "fps": 1/(dt*frame_stride)},
                                "wrist_rgb": {"shape": [h, w, 3], "dtype": "uint8", "fps": 1/(dt*frame_stride)}},
                    "termination": {"status": "complete", "reason": "script_horizon"}, "result": metrics,
                    "artifacts": {"timeline": timeline_name, "video": "base_rgb.mp4",
                                  "wrist_video": "wrist_rgb.mp4", "audit": "model_audit.json"},
                    "integrity": {"steps": steps, "frames": frames, "decoded_frames": decoded,
                                  "trajectory_sha256": trajectory_hash.hexdigest(),
                                  "raw_base_rgb_sha256": rgb_hashes["base_rgb"].hexdigest(),
                                  "raw_wrist_rgb_sha256": rgb_hashes["wrist_rgb"].hexdigest(),
                                  "timeline_sha256": sha256(timeline_path),
                                  "video_sha256": sha256(output / "base_rgb.mp4"),
                                  "wrist_video_sha256": sha256(output / "wrist_rgb.mp4"),
                                  "raw_rgb_frame_storage": "npy" if save_raw_rgb_frames else "video_only"}}
        if preview is not None:
            manifest["artifacts"].update({"preview_video": "base_rgb.webm",
                                          "preview_manifest": "preview_manifest.json"})
            manifest["integrity"].update({"preview_video_sha256": preview["sha256"],
                                           "preview_decoded_frames": preview["frames"]})
        save_json(output / "episode_manifest.json", manifest)
        (output / ".incomplete").unlink()
        return manifest
    except BaseException as exc:
        save_json(output / "failure.json", {"status": "failed", "error": str(exc)})
        raise
    finally:
        if video is not None:
            video.release()
        if wrist_video is not None:
            wrist_video.release()
        if env is not None:
            env.close()
