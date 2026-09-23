from __future__ import annotations

import copy
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fragilegrab.benchmark.environment import MujocoQualificationEnv, load_configs
from fragilegrab.benchmark.piper_x_asset import (
    _camera_pose,
    _fixed_transform,
    verify_piper_x_checkout,
)
from fragilegrab.benchmark.runner import model_audit


DOCS = load_configs(ROOT / "configs/benchmark")
ROBOT = DOCS["robot"]["values"]


def _assets() -> Path:
    path = Path(os.environ.get("PIPER_X_ASSETS", ROOT / "assets/vendor/agx_arm_sim"))
    if not path.is_dir():
        pytest.skip("official PiPER-X assets are not installed; run fetch_piper_x.py")
    return path


def test_official_commit_hashes_and_camera_mount_contract() -> None:
    checkout, paths = verify_piper_x_checkout(_assets(), ROBOT)
    assert checkout == _assets().resolve()
    root = ET.parse(paths["combined_urdf"]).getroot()
    position, xyaxes = _camera_pose(root, ROBOT)
    assert np.isfinite(position).all()
    assert np.asarray(xyaxes).shape == (6,)
    camera_rotation, camera_link_position = _fixed_transform(
        root, ROBOT["wrist_body"], ROBOT["camera_frame"]
    )
    expected_color_position = camera_link_position + camera_rotation @ np.array([0.0, 0.015, 0.0])
    np.testing.assert_allclose(position, expected_color_position)

    changed = copy.deepcopy(root)
    changed.find("joint[@name='camera_stand_joint']/origin").set("xyz", "9 9 9")
    with pytest.raises(ValueError, match="camera mount contract changed"):
        _camera_pose(changed, ROBOT)


def test_compiled_piper_x_joint_mimic_home_and_cameras() -> None:
    env = MujocoQualificationEnv(DOCS, _assets())
    try:
        observation = env.reset(7)
        assert env.joint_names == ROBOT["arm_joints"] + ROBOT["finger_joints"]
        np.testing.assert_allclose(env.model.actuator_ctrlrange[-1], [0.0, 0.05])
        audit = model_audit(env)
        assert audit["static_pass"] and audit["control_pass"] and audit["gripper_pass"]
        assert observation["images"]["base_rgb"].std() > 0
        assert observation["images"]["wrist_rgb"].std() > 0
        assert env.policy_camera == "agentview"
        assert env.model.body("akita_black_bowl_1_main").id == env.target_id
        for body in (
            "akita_black_bowl_2_main",
            "plate_1_main",
            "cookies_1_main",
            "glazed_rim_porcelain_ramekin_1_main",
            "wooden_cabinet_1_main",
            "flat_stove_1_main",
        ):
            assert env.model.body(body).id >= 0
        assert env.provenance["libero_task_id"] == (
            "libero_spatial_pick_black_bowl_table_center_to_plate"
        )
        assert env.provenance["camera_sensor_frame"] == "camera_color_optical_frame"

        scene = ET.fromstring(env.scene_xml)
        table = scene.find("worldbody/body[@name='table']")
        np.testing.assert_allclose(
            np.fromstring(table.find("geom[@name='table_collision']").get("friction"), sep=" "),
            [0.6, 0.005, 0.0001],
        )
        for name, expected in zip(
            ("table_leg1_visual", "table_leg2_visual", "table_leg3_visual", "table_leg4_visual"),
            ([0.4, 0.5, -0.4375], [-0.4, 0.5, -0.4375],
             [-0.4, -0.5, -0.4375], [0.4, -0.5, -0.4375]),
        ):
            leg = table.find(f"geom[@name='{name}']")
            np.testing.assert_allclose(np.fromstring(leg.get("pos"), sep=" "), expected)
            np.testing.assert_allclose(np.fromstring(leg.get("size"), sep=" "), [0.025, 0.4375])
        assert scene.find("asset/texture[@name='texplane']").get("file").endswith(
            "light-gray-floor-tile.png"
        )
        assert scene.find("asset/texture[@name='tex-wall']").get("file").endswith(
            "light-gray-plaster.png"
        )
        np.testing.assert_allclose(env.model.body("wooden_cabinet_1_main").pos[2], 0.905)
        np.testing.assert_allclose(env.model.body("flat_stove_1_main").pos[2], 0.905)

        import mujoco
        assert env.model.neq == 1
        assert env.model.eq_type[0] == mujoco.mjtEq.mjEQ_JOINT
        assert env.model.eq_obj1id[0] == env.model.joint("gripper_joint2").id
        assert env.model.eq_obj2id[0] == env.model.joint("gripper_joint1").id
        np.testing.assert_allclose(env.model.eq_data[0, :2], [0.0, -1.0])
    finally:
        env.close()
