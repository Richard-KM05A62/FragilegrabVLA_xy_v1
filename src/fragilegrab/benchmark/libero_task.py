"""Compose the locked first LIBERO task without importing its robot runtime."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from robosuite.models.objects import MujocoXMLObject


LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
TASK_ID = "libero_spatial_pick_black_bowl_table_center_to_plate"

REGIONS = {
    "akita_black_bowl_1": (-0.10, -0.01, -0.05, 0.01),
    "akita_black_bowl_2": (0.00, 0.30, 0.02, 0.32),
    "plate_1": (0.05, 0.19, 0.07, 0.21),
    "cookies_1": (0.06, 0.02, 0.08, 0.04),
    "glazed_rim_porcelain_ramekin_1": (-0.21, 0.19, -0.19, 0.21),
    "wooden_cabinet_1": (0.02, -0.28, 0.04, -0.26),
    "flat_stove_1": (-0.42, -0.15, -0.40, -0.13),
}

MOVABLE_OBJECTS = tuple(list(REGIONS)[:5])
FIXTURE_YAW = {
    "wooden_cabinet_1": (2.6613777765410678, 2.7242096296128633),
    "flat_stove_1": (0.0, 0.0),
}
TABLE_OFFSET = np.array([0.0, 0.0, 0.90])
TABLE_FULL_SIZE = np.array([1.0, 1.2, 0.05])
OBJECT_HORIZONTAL_RADIUS = float(np.linalg.norm([0.025, 0.025]))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _absolute_files(root: ET.Element, directory: Path) -> None:
    for node in root.findall(".//*[@file]"):
        node.set("file", str((directory / node.get("file")).resolve()))


def _set_xyz(node: ET.Element, key: str, values) -> None:
    node.set(key, " ".join(str(float(x)) for x in values))


def _merge_assets(destination: ET.Element, source: ET.Element) -> None:
    """Match robosuite's asset merge behavior by de-duplicating named assets."""
    existing = {(node.tag, node.get("name")) for node in destination if node.get("name")}
    for node in source:
        identity = (node.tag, node.get("name"))
        if node.get("name") and identity in existing:
            continue
        destination.append(copy.deepcopy(node))
        if node.get("name"):
            existing.add(identity)


def _configure_arena(arena: ET.Element) -> None:
    """Apply LIBERO_Tabletop_Manipulation's official arena and camera settings."""
    table_full_size = TABLE_FULL_SIZE
    table_offset = TABLE_OFFSET
    half = table_full_size / 2
    table = arena.find("worldbody/body[@name='table']")
    _set_xyz(table, "pos", table_offset - np.array([0.0, 0.0, half[2]]))
    for name in ("table_collision", "table_visual"):
        _set_xyz(table.find(f"geom[@name='{name}']"), "size", half)
    _set_xyz(table.find("geom[@name='table_collision']"), "friction", [0.6, 0.005, 0.0001])
    _set_xyz(table.find("site[@name='table_top']"), "pos", [0.0, 0.0, half[2]])
    leg_z = (table_offset[2] - half[2]) / 2.0
    for name, x, y in zip(
        ("table_leg1_visual", "table_leg2_visual", "table_leg3_visual", "table_leg4_visual"),
        (0.4, -0.4, -0.4, 0.4),
        (0.5, 0.5, -0.5, -0.5),
    ):
        leg = table.find(f"geom[@name='{name}']")
        _set_xyz(leg, "pos", [x, y, -leg_z])
        _set_xyz(leg, "size", [0.025, leg_z])

    textures = arena.find("asset")
    for texture_name, filename in (
        ("texplane", "light-gray-floor-tile.png"),
        ("tex-wall", "light-gray-plaster.png"),
    ):
        texture = textures.find(f"texture[@name='{texture_name}']")
        texture.set("file", str(Path(texture.get("file")).parent / filename))
    for camera, pos, quat in (
        ("agentview", [0.6586131746834771, 0.0, 1.6103500240372423],
         [0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]),
        ("frontview", [1.0, 0.0, 1.48], [0.56, 0.43, 0.43, 0.56]),
    ):
        node = arena.find(f"worldbody/camera[@name='{camera}']")
        _set_xyz(node, "pos", pos)
        _set_xyz(node, "quat", quat)


def _object(path: Path, name: str, *, fixed: bool) -> MujocoXMLObject:
    return MujocoXMLObject(
        str(path),
        name=name,
        joints=None if fixed else [dict(type="free", damping="0.0005")],
        obj_type="all",
        duplicate_collision_geoms=False,
    )


def _official_table_object_quat(angle: float = np.pi / 2) -> np.ndarray:
    """Reproduce LIBERO TableRegionSampler's quaternion as assigned to MuJoCo qpos."""
    return np.array([np.sin(angle / 2), 0.0, 0.0, np.cos(angle / 2)])


def compose_first_libero_task(root: ET.Element, checkout: Path, config: dict) -> dict:
    """Merge the exact first locked LIBERO-Spatial task into a PiPER-X MJCF root."""
    checkout = checkout.resolve()
    commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if commit != LIBERO_COMMIT or commit != config["libero_commit"]:
        raise ValueError(f"LIBERO commit mismatch: expected {LIBERO_COMMIT}, got {commit}")
    package = checkout / "libero/libero"
    bddl = package / config["bddl_file"]
    if _sha256(bddl) != config["bddl_sha256"]:
        raise ValueError("LIBERO BDDL hash mismatch")
    if config["task_id"] != TASK_ID:
        raise ValueError(f"unsupported LIBERO task: {config['task_id']}")

    scene = package / "assets/scenes/libero_tabletop_base_style.xml"
    arena = ET.parse(scene).getroot()
    _absolute_files(arena, scene.parent)
    _configure_arena(arena)
    _merge_assets(root.find("asset"), arena.find("asset"))
    root.find("worldbody").extend(copy.deepcopy(list(arena.find("worldbody"))))

    assets = package / "assets"
    specs = [
        ("akita_black_bowl_1", "stable_scanned_objects/akita_black_bowl/akita_black_bowl.xml", False, [-0.075, 0.0], None),
        ("akita_black_bowl_2", "stable_scanned_objects/akita_black_bowl/akita_black_bowl.xml", False, [0.01, 0.31], None),
        ("plate_1", "stable_scanned_objects/plate/plate.xml", False, [0.06, 0.20], None),
        ("cookies_1", "stable_hope_objects/cookies/cookies.xml", False, [0.07, 0.03], None),
        ("glazed_rim_porcelain_ramekin_1", "stable_scanned_objects/glazed_rim_porcelain_ramekin/glazed_rim_porcelain_ramekin.xml", False, [-0.20, 0.20], None),
        ("wooden_cabinet_1", "articulated_objects/wooden_cabinet.xml", True, [0.03, -0.27], 2.692793703),
        ("flat_stove_1", "articulated_objects/flat_stove.xml", True, [-0.41, -0.14], 0.0),
    ]
    free_qpos = []
    object_evidence = {}
    for name, relative, fixed, xy, yaw in specs:
        path = assets / relative
        obj = _object(path, name, fixed=fixed)
        _merge_assets(root.find("asset"), obj.asset)
        body = copy.deepcopy(obj.get_obj())
        z_offset = 0.01 if not fixed else 0.01 - TABLE_FULL_SIZE[2]
        z = TABLE_OFFSET[2] + z_offset - float(obj.bottom_offset[-1])
        quat = (
            _official_table_object_quat()
            if yaw is None
            else np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        )
        _set_xyz(body, "pos", [xy[0], xy[1], z])
        _set_xyz(body, "quat", quat)
        root.find("worldbody").append(body)
        if not fixed:
            free_qpos.extend([xy[0], xy[1], z, *quat])
        object_evidence[name] = {"xml": relative, "sha256": _sha256(path), "fixed": fixed}

    for key in root.findall("keyframe/key"):
        key.set("qpos", key.get("qpos") + " " + " ".join(map(str, free_qpos)))
    return {
        "libero_repository": config["libero_repository"],
        "libero_commit": commit,
        "libero_task_id": TASK_ID,
        "libero_bddl": config["bddl_file"],
        "libero_bddl_sha256": _sha256(bddl),
        "libero_scene_sha256": _sha256(scene),
        "libero_objects": object_evidence,
        "placement": "LIBERO 0.1.0 TableRegionSampler semantics ported locally (legacy NumPy RNG, boundary and z offsets)",
        "policy_camera": "agentview",
    }


def reset_first_libero_task(model, data, seed: int) -> None:
    """Port LIBERO 0.1.0's sampler sequence without importing its robot runtime."""
    rng = np.random.RandomState(seed)
    placed: list[tuple[float, float, float]] = []
    for name, (x0, y0, x1, y1) in REGIONS.items():
        rng.randint(1)  # MultiRegionRandomSampler selects one BDDL rectangle first.
        radius = OBJECT_HORIZONTAL_RADIUS if name in MOVABLE_OBJECTS else 0.0
        for _ in range(5000):
            x = rng.uniform(x0 + radius, x1 - radius)
            y = rng.uniform(y0 + radius, y1 - radius)
            if name not in MOVABLE_OBJECTS or all(
                np.linalg.norm((x - ox, y - oy)) > radius + other_radius
                for ox, oy, other_radius in placed
            ):
                break
        if name in MOVABLE_OBJECTS:
            rng.uniform(np.pi / 2, np.pi / 2)  # Fixed rotation still consumes RNG in LIBERO.
            joint = model.joint(f"{name}_joint0")
            address = int(model.jnt_qposadr[joint.id])
            data.qpos[address : address + 7] = [x, y, 0.97, *_official_table_object_quat()]
            placed.append((x, y, radius))
        else:
            yaw = rng.uniform(*FIXTURE_YAW[name])
            body = model.body(f"{name}_main")
            model.body_pos[body.id] = [x, y, 0.905]
            model.body_quat[body.id] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
