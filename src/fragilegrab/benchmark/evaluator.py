"""逐物理步评测：只解释测量，不修改动作，不模拟真实破裂。"""
import numpy as np


class GraspTendencyEvaluator:
    """把已有仿真真值解释为 L0–L6；不修改动作或任务状态。"""

    LABELS = {
        0: "L0_random_or_no_target_relation",
        1: "L1_approach",
        2: "L2_pregrasp_region",
        3: "L3_grasp_alignment",
        4: "L4_close_and_contact",
        5: "L5_secure_contact_without_stable_lift",
        6: "L6_stable_lift",
    }

    def __init__(self, cfg, initial_state, target_id, finger_ids):
        self.cfg = cfg
        self.target_id = int(target_id)
        self.finger_ids = set(finger_ids)
        self.initial_object = np.asarray(initial_state["object_position_m"], dtype=np.float64)
        initial_delta = self.initial_object - np.asarray(initial_state["grasp_position_m"], dtype=np.float64)
        self.initial_distance = float(np.linalg.norm(initial_delta))
        self.min_distance = self.initial_distance
        self.min_distance_time_s = 0.0
        self.max_lift = 0.0
        self.approached = False
        self.pregrasp = False
        self.aligned = False
        self.close_started = False
        self.close_time_s = None
        self.close_distance_m = None
        self.close_near_object = False
        self.contact = False
        self.bilateral_contact = False
        self.lift_attempt = False
        self.stable_lift = False
        self.bilateral_lift_hold_s = 0.0
        self.max_bilateral_lift_hold_s = 0.0
        self.stage_first_time_s = {"L0": 0.0}
        self.trace = []

    @staticmethod
    def _orientation_error_deg(rotation, desired_tool_z):
        tool_z = np.asarray(rotation, dtype=np.float64).reshape(3, 3)[:, 2]
        desired = np.asarray(desired_tool_z, dtype=np.float64)
        desired /= np.linalg.norm(desired)
        cosine = float(np.clip(np.dot(tool_z, desired), -1.0, 1.0))
        return float(np.degrees(np.arccos(cosine)))

    def update(self, state, dt):
        time_s = float(state["time_s"])
        obj = np.asarray(state["object_position_m"], dtype=np.float64)
        ee = np.asarray(state["grasp_position_m"], dtype=np.float64)
        distance = float(np.linalg.norm(obj - ee))
        if distance < self.min_distance:
            self.min_distance = distance
            self.min_distance_time_s = time_s
        orientation_error = self._orientation_error_deg(
            state["wrist_rotation"], self.cfg["desired_tool_z_world"]
        )
        finger_contact = set()
        for contact in state["contacts"]:
            if contact["normal_force_N"] <= self.cfg["contact_epsilon_N"]:
                continue
            bodies = {contact["body1"], contact["body2"]}
            if self.target_id in bodies:
                finger_contact |= bodies & self.finger_ids
        any_contact = bool(finger_contact)
        bilateral = finger_contact == self.finger_ids
        closure = float(state["gripper_closure"])
        lift = float(obj[2] - self.initial_object[2])
        self.max_lift = max(self.max_lift, lift)

        self.approached |= self.initial_distance - self.min_distance >= self.cfg["approach_progress_m"]
        self.pregrasp |= distance <= self.cfg["pregrasp_distance_m"]
        self.aligned |= self.pregrasp and orientation_error <= self.cfg["alignment_tolerance_deg"]
        if not self.close_started and closure >= self.cfg["close_start_closure"]:
            self.close_started = True
            self.close_time_s = time_s
            self.close_distance_m = distance
            self.close_near_object = distance <= self.cfg["close_near_object_distance_m"]
        self.contact |= any_contact
        self.bilateral_contact |= bilateral
        self.lift_attempt |= lift >= self.cfg["lift_attempt_m"]
        lifted_with_contact = bilateral and lift >= self.cfg["stable_lift_m"]
        self.bilateral_lift_hold_s = self.bilateral_lift_hold_s + dt if lifted_with_contact else 0.0
        self.max_bilateral_lift_hold_s = max(
            self.max_bilateral_lift_hold_s, self.bilateral_lift_hold_s
        )
        self.stable_lift |= self.bilateral_lift_hold_s >= self.cfg["stable_lift_hold_s"]

        stage = self.stage()
        key = f"L{stage}"
        self.stage_first_time_s.setdefault(key, time_s)
        if not self.trace or time_s - self.trace[-1]["time_s"] >= self.cfg["trace_period_s"] - 1e-9:
            self.trace.append({
                "time_s": time_s,
                "ee_object_distance_m": distance,
                "orientation_error_deg": orientation_error,
                "gripper_closure": closure,
                "finger_contact_count": len(finger_contact),
                "object_lift_m": lift,
                "stage": stage,
            })

    def stage(self):
        if self.stable_lift:
            return 6
        if self.bilateral_contact:
            return 5
        if self.close_near_object and self.contact:
            return 4
        if self.aligned:
            return 3
        if self.pregrasp:
            return 2
        if self.approached:
            return 1
        return 0

    def result(self):
        stage = self.stage()
        return {
            "schema_version": "fragilegrab.grasp_tendency/0.1",
            "highest_stage": stage,
            "highest_stage_label": self.LABELS[stage],
            "initial_ee_object_distance_m": self.initial_distance,
            "minimum_ee_object_distance_m": self.min_distance,
            "minimum_distance_time_s": self.min_distance_time_s,
            "approach_progress_m": self.initial_distance - self.min_distance,
            "entered_pregrasp_region": self.pregrasp,
            "grasp_alignment_observed": self.aligned,
            "gripper_close_started": self.close_started,
            "gripper_close_time_s": self.close_time_s,
            "gripper_close_ee_object_distance_m": self.close_distance_m,
            "gripper_close_near_object": self.close_near_object,
            "finger_object_contact": self.contact,
            "bilateral_finger_contact": self.bilateral_contact,
            "lift_attempt": self.lift_attempt,
            "object_maximum_lift_m": self.max_lift,
            "maximum_bilateral_lift_hold_s": self.max_bilateral_lift_hold_s,
            "stable_lift": self.stable_lift,
            "stage_first_time_s": self.stage_first_time_s,
            "trace": self.trace,
        }


class SafetyEvaluator:
    def __init__(self, cfg, initial_position, target_id, finger_ids, robot_bodies, allowed_pairs, task):
        self.cfg, self.initial = cfg, np.asarray(initial_position)
        self.target_id, self.finger_ids = target_id, set(finger_ids)
        self.robot_bodies, self.allowed_pairs = set(robot_bodies), set(allowed_pairs)
        self.task = task
        self.peak, self.impulse, self.max_event_impulse = 0.0, 0.0, 0.0
        self.active_impulses = {}
        self.active_unintended = set()
        self.collision_count = 0
        self.slip, self.drop = False, False
        self.ever_lifted = False
        self.relative_at_grasp = None
        self.hold_time, self.max_hold_time = 0.0, 0.0
        self.place_hold = 0.0
        self.reached, self.picked, self.transported, self.placed = False, False, False, False
        self.max_height = 0.0

    def update(self, state, phase, dt):
        pos = np.asarray(state["object_position_m"])
        grasp = np.asarray(state["grasp_position_m"])
        relative = np.asarray(state["wrist_rotation"]).T @ (pos - grasp)
        height = float(pos[2] - self.initial[2])
        self.max_height = max(self.max_height, height)
        finger_contact = set()
        support_contact = False
        pair_forces, unintended = {}, set()
        for c in state["contacts"]:
            force = c["normal_force_N"]
            if force <= self.cfg["contact_epsilon_N"]:
                continue
            a, b = c["body1"], c["body2"]
            bodies = {a, b}
            pair = tuple(sorted((a, b)))
            if self.target_id in bodies:
                pair_forces[pair] = pair_forces.get(pair, 0.0) + force
                finger_contact |= bodies & self.finger_ids
                if not bodies & self.robot_bodies:
                    support_contact = True
            intended_grasp = self.target_id in bodies and bool(bodies & self.finger_ids)
            if bodies & self.robot_bodies and not intended_grasp and pair not in self.allowed_pairs:
                unintended.add(pair)
        total = sum(pair_forces.values())
        self.peak = max(self.peak, total)
        self.impulse += total * dt
        for pair in set(self.active_impulses) - set(pair_forces):
            self.max_event_impulse = max(self.max_event_impulse, self.active_impulses.pop(pair))
        for pair, force in pair_forces.items():
            self.active_impulses[pair] = self.active_impulses.get(pair, 0.0) + force * dt
        self.collision_count += len(unintended - self.active_unintended)
        self.active_unintended = unintended
        bilateral = finger_contact == self.finger_ids
        if np.linalg.norm(pos + np.asarray(self.cfg["grasp_offset_m"]) - grasp) <= self.cfg["reach_tolerance_m"]:
            self.reached = True
        lifted = bilateral and height >= self.cfg["lift_success_m"]
        self.hold_time = self.hold_time + dt if lifted else 0.0
        self.max_hold_time = max(self.max_hold_time, self.hold_time)
        if lifted:
            self.ever_lifted = True
        if self.reached and self.hold_time + 1e-9 >= self.cfg["hold_s"]:
            self.picked = True
        goal = self.initial + np.asarray(self.cfg["transport_offset_m"])
        supported_placement = (phase == "place" and support_contact
                               and np.linalg.norm(pos[:2] - goal[:2]) <= self.cfg["place_tolerance_m"])
        if self.ever_lifted and phase != "release" and not supported_placement:
            if self.relative_at_grasp is None:
                self.relative_at_grasp = relative.copy()
            if np.linalg.norm(relative - self.relative_at_grasp) > self.cfg["slip_relative_m"]:
                self.slip = True
            if height < self.cfg["drop_height_m"] and not bilateral:
                self.drop = True
        goal = self.initial + np.asarray(self.cfg["transport_offset_m"])
        if self.picked and lifted and np.linalg.norm(pos[:2] - goal[:2]) <= self.cfg["place_tolerance_m"]:
            self.transported = True
        placed_now = (phase == "release" and self.transported and not finger_contact
                      and abs(height) < self.cfg["drop_height_m"]
                      and np.linalg.norm(pos[:2] - goal[:2]) <= self.cfg["place_tolerance_m"])
        self.place_hold = self.place_hold + dt if placed_now else 0.0
        if self.place_hold + 1e-9 >= self.cfg["hold_s"]:
            self.placed = True

    def result(self):
        event_max = max([self.max_event_impulse, *self.active_impulses.values()])
        success = (self.reached and self.hold_time + 1e-9 >= self.cfg["hold_s"]) if self.task == "F1" else self.placed
        return {"protocol": "fragilemanip.safety/0.1", "task_success": bool(success),
                "qualification": {"reach": self.reached, "pick": self.picked,
                                  "transport": self.transported, "place": self.placed},
                "peak_contact_force_N": self.peak, "contact_impulse_Ns": self.impulse,
                "max_pair_contact_event_impulse_Ns": event_max,
                "slip": self.slip, "drop": self.drop, "unintended_collision_events": self.collision_count,
                "max_lift_m": self.max_height, "max_bilateral_hold_s": self.max_hold_time,
                "threshold_sweep": [{"force_N": f, "event_impulse_Ns": j,
                    "safe_success": bool(success and self.peak <= f and event_max <= j
                                         and not self.slip and not self.drop and self.collision_count == 0)}
                    for f in self.cfg["force_thresholds_N"] for j in self.cfg["impulse_thresholds_Ns"]],
                "interpretation": "benchmark-defined 刚体接触代理；不代表玻璃或陶瓷真实断裂"}
