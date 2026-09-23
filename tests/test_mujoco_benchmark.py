"""指标因果反例与未知策略映射拒绝测试；不需要 GPU 或资产下载。"""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fragilegrab.benchmark.environment import load_configs
from fragilegrab.benchmark.evaluator import SafetyEvaluator
from fragilegrab.benchmark.adapters import Pi05Adapter, ScriptedActionAdapter

CFG = load_configs(Path(__file__).resolve().parents[1] / "configs/benchmark")["benchmark"]["values"]


def evaluator(task="F1"):
    return SafetyEvaluator({**CFG, "grasp_offset_m": [0, 0, 0]}, [0, 0, 0], 10, [7, 8], range(1, 9), set(), task)


def state(z, contacts=(), x=0, grasp=None):
    return {"object_position_m": [x, 0, z], "grasp_position_m": [x, 0, z] if grasp is None else grasp,
            "wrist_rotation": np.eye(3), "contacts": list(contacts)}


def contact(a, b, f):
    return {"body1": a, "body2": b, "normal_force_N": f}


FINGERS = [contact(7, 10, 2), contact(8, 10, 3)]


def test_force_impulse_uses_each_physics_dt_and_aggregates_pairs():
    e = evaluator()
    e.update(state(0, [contact(7, 10, 2), contact(7, 10, 3)]), "close", .01)
    e.update(state(0, [contact(7, 10, 1)]), "close", .02)
    e.update(state(0), "close", .01)
    e.update(state(0, [contact(7, 10, 2)]), "close", .01)
    r = e.result()
    assert r["peak_contact_force_N"] == 5
    assert r["contact_impulse_Ns"] == pytest.approx(.09)
    assert r["max_pair_contact_event_impulse_Ns"] == pytest.approx(.07)
    assert not r["task_success"]


def test_lift_requires_bilateral_contact_and_continuous_hold():
    e = evaluator()
    e.update(state(.08), "hold", 2)
    assert not e.result()["task_success"]
    e.update(state(.08, FINGERS), "hold", .6)
    e.update(state(.08), "hold", .01)
    e.update(state(.08, FINGERS), "hold", .6)
    assert not e.result()["task_success"]
    e.update(state(.08, FINGERS), "hold", .5)
    assert e.result()["task_success"]
    e.update(state(0), "hold", .1)
    assert not e.result()["task_success"]
    assert e.result()["drop"]


def test_slip_relative_to_wrist_and_collision_counts_onsets():
    e = evaluator()
    e.update(state(.08, FINGERS), "pick", .1)
    hit = contact(5, 0, 1)
    e.update(state(.08, [*FINGERS, hit], grasp=[.04, 0, .08]), "pick", .1)
    e.update(state(.08, [hit]), "pick", .1)
    assert e.result()["slip"]
    assert e.result()["unintended_collision_events"] == 1
    e.update(state(.08), "pick", .1)
    e.update(state(.08, [hit]), "pick", .1)
    assert e.result()["unintended_collision_events"] == 2


def test_allowed_gripper_self_contact_is_not_unintended_collision():
    e = SafetyEvaluator(
        {**CFG, "grasp_offset_m": [0, 0, 0]},
        [0, 0, 0],
        10,
        [7, 8],
        range(1, 9),
        {(7, 8)},
        "F1",
    )
    e.update(state(0, [contact(7, 8, 1)]), "close", 0.01)
    assert e.result()["unintended_collision_events"] == 0


def test_place_cannot_pass_without_pick_and_transport():
    e = evaluator("qualification")
    e.update(state(0), "release", 5)
    assert not e.result()["task_success"]


def test_unverified_pi05_chunk_is_never_executed():
    adapter = Pi05Adapter()
    with pytest.raises(NotImplementedError):
        adapter.decode_action_chunk(np.zeros((50, 32)))
    with pytest.raises(NotImplementedError):
        adapter.encode_observation({})


def test_action_rejects_nan_wrong_shape_and_range():
    class Model:
        actuator_ctrlrange = np.array([[-1, 1]] * 7)
    class Env:
        model = Model()
    adapter = ScriptedActionAdapter()
    for action in [np.zeros(8), np.full(7, np.nan), np.full(7, 2)]:
        with pytest.raises(ValueError):
            adapter.decode(Env(), action)


def test_vscode_preview_is_vp8_without_changing_original(tmp_path):
    import cv2
    from fragilegrab.benchmark.video import export_webm

    frames = tmp_path / "streams/base_rgb"
    frames.mkdir(parents=True)
    for i in range(4):
        image = np.zeros((32, 48, 3), dtype=np.uint8)
        image[:, :, 0] = i * 40
        np.save(frames / f"{i:06d}.npy", image)
    original = tmp_path / "base_rgb.mp4"
    original.write_bytes(b"original-evidence")
    info = export_webm(tmp_path, 4, (32, 48), 20)
    assert info["codec"] == "VP8"
    assert info["frames"] == 4
    assert original.read_bytes() == b"original-evidence"
    reader = cv2.VideoCapture(str(tmp_path / "base_rgb.webm"))
    code = int(reader.get(cv2.CAP_PROP_FOURCC))
    reader.release()
    assert "".join(chr((code >> (8 * i)) & 255) for i in range(4)) == "VP80"
    with pytest.raises(FileExistsError):
        export_webm(tmp_path, 4, (32, 48), 20)
