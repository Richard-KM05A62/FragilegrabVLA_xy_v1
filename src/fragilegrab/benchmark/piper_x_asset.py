"""Build MuJoCo input from AgileX's pinned PiPER-X + D435 simulation asset."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def verify_piper_x_checkout(checkout: Path, robot: dict) -> tuple[Path, dict]:
    checkout = Path(checkout).resolve()
    commit = _git(checkout, "rev-parse", "HEAD")
    if commit != robot["commit"]:
        raise ValueError(f"agx_arm_sim commit mismatch: expected {robot['commit']}, got {commit}")
    submodule = checkout / robot["urdf_submodule"]
    submodule_commit = _git(submodule, "rev-parse", "HEAD")
    if submodule_commit != robot["urdf_submodule_commit"]:
        raise ValueError(
            "agx_arm_urdf submodule commit mismatch: "
            f"expected {robot['urdf_submodule_commit']}, got {submodule_commit}"
        )
    tracked_changes = _git(checkout, "status", "--porcelain", "--untracked-files=no")
    if tracked_changes:
        raise ValueError("official agx_arm_sim checkout has tracked local changes")

    paths = {
        key: checkout / robot[key]
        for key in (
            "combined_urdf",
            "isaac_usd",
            "official_mjcf_converter",
            "camera_stand_mesh",
            "d435_mesh",
        )
    }
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        expected = robot[f"{key}_sha256"]
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"{key} hash mismatch: expected {expected}, got {actual}")
    return checkout, {"commit": commit, "submodule_commit": submodule_commit, **paths}


def _rpy_matrix(text: str | None) -> np.ndarray:
    r, p, y = (float(x) for x in (text or "0 0 0").split())
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _fixed_transform(root: ET.Element, start: str, end: str) -> tuple[np.ndarray, np.ndarray]:
    joints = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        joints.setdefault(parent, []).append((child, joint))

    def visit(link: str, rotation: np.ndarray, position: np.ndarray):
        if link == end:
            return rotation, position
        for child, joint in joints.get(link, []):
            if joint.get("type") != "fixed":
                continue
            origin = joint.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            child_rotation = rotation @ _rpy_matrix(origin.get("rpy"))
            child_position = position + rotation @ xyz
            result = visit(child, child_rotation, child_position)
            if result is not None:
                return result
        return None

    result = visit(start, np.eye(3), np.zeros(3))
    if result is None:
        raise ValueError(f"official fixed chain {start}->{end} is missing")
    return result


def _camera_pose(root: ET.Element, robot: dict) -> tuple[list[float], list[float]]:
    expected = robot["camera_fixed_joints"]
    observed = {}
    for name in expected:
        joint = root.find(f"joint[@name='{name}']")
        if joint is None:
            raise ValueError(f"official camera joint is missing: {name}")
        origin = joint.find("origin")
        observed[name] = {
            "parent": joint.find("parent").get("link"),
            "child": joint.find("child").get("link"),
            "xyz": [float(x) for x in origin.get("xyz", "0 0 0").split()],
            "rpy": [float(x) for x in origin.get("rpy", "0 0 0").split()],
        }
    if observed != expected:
        raise ValueError(f"official PiPER-X camera mount contract changed: {observed}")

    rotation, position = _fixed_transform(root, robot["wrist_body"], robot["camera_frame"])
    position = position + rotation @ np.asarray(robot["camera_color_offset_camera_link_m"])
    optical_rotation = rotation @ _rpy_matrix(
        " ".join(str(value) for value in robot["camera_optical_rpy_camera_link"])
    )
    # ROS optical frame: +X right, +Y down, +Z forward. MuJoCo: +X right, +Y up, -Z forward.
    x_axis = optical_rotation[:, 0]
    y_axis = -optical_rotation[:, 1]
    return position.tolist(), np.r_[x_axis, y_axis].tolist()


def _resolve_package_uri(checkout: Path, uri: str) -> Path:
    if not uri.startswith("package://"):
        raise ValueError(f"unexpected mesh URI: {uri}")
    package, relative = uri[len("package://") :].split("/", 1)
    package_roots = {
        "agx_arm_description": checkout / "agx_arm_description",
        "realsense2_description": checkout / "realsense2_description",
    }
    if package not in package_roots:
        raise ValueError(f"unsupported package URI: {uri}")
    return (package_roots[package] / relative).resolve()


def _convert_dae(source: Path, destination: Path) -> None:
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("PiPER-X visual conversion requires trimesh and pycollada") from exc
    scene = trimesh.load(source, force="scene")
    geometry = scene.to_geometry()
    if isinstance(geometry, trimesh.Scene):
        geometry = trimesh.util.concatenate(tuple(geometry.geometry.values()))
    geometry.export(destination)


def materialize_piper_x_urdf(checkout: Path, robot: dict, paths: dict) -> tuple[ET.Element, dict]:
    root = ET.parse(paths["combined_urdf"]).getroot()
    position, xyaxes = _camera_pose(root, robot)
    cache = checkout / ".fragilegrab_cache" / "piper_x_visual_obj"
    cache.mkdir(parents=True, exist_ok=True)
    converted = {}
    for mesh in root.findall(".//mesh"):
        source = _resolve_package_uri(checkout, mesh.get("filename"))
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.suffix.lower() == ".dae":
            identity = _sha256(source)
            destination = cache / f"{identity[:16]}_{source.stem}.obj"
            if not destination.exists():
                _convert_dae(source, destination)
            source = destination
            converted[source.name] = identity
        mesh.set("filename", str(source))

    extension = ET.SubElement(root, "mujoco")
    ET.SubElement(
        extension,
        "compiler",
        strippath="false",
        discardvisual="false",
        balanceinertia="false",
    )
    return root, {
        "camera_position_link6_m": position,
        "camera_xyaxes_link6": xyaxes,
        "camera_sensor_frame": robot["camera_sensor_frame"],
        "converted_visual_meshes": converted,
    }


def compile_piper_x_mjcf(checkout: Path, robot: dict) -> tuple[ET.Element, dict]:
    checkout, verified = verify_piper_x_checkout(checkout, robot)
    urdf_root, derived = materialize_piper_x_urdf(checkout, robot, verified)
    urdf_xml = ET.tostring(urdf_root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(urdf_xml)
    with tempfile.NamedTemporaryFile(suffix=".xml") as handle:
        mujoco.mj_saveLastXML(handle.name, model)
        mjcf_root = ET.parse(handle.name).getroot()
    evidence = {
        "repository": robot["repository"],
        "commit": verified["commit"],
        "urdf_submodule_commit": verified["submodule_commit"],
        "combined_urdf_sha256": _sha256(verified["combined_urdf"]),
        "isaac_usd_sha256": _sha256(verified["isaac_usd"]),
        "official_mjcf_converter_sha256": _sha256(verified["official_mjcf_converter"]),
        "materialized_urdf_sha256": hashlib.sha256(urdf_xml.encode()).hexdigest(),
        "conversion": "official combined PiPER-X+D435 URDF; DAE visuals converted to OBJ; official STL collisions retained",
        **derived,
    }
    return mjcf_root, evidence
