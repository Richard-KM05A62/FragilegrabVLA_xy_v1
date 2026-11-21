"""Deterministic RTC harness checks: no camera, CAN, network, or GPU required."""
from __future__ import annotations

import threading
import time
import unittest
from collections import deque
from dataclasses import replace

import numpy as np

from inference.rtc_controller import Controller
from inference.rtc_core import (
    BoundedHistory, DataBuffer, Executor, InferenceRequest, InferenceResult,
    NullCommandSink, Prediction, PredictionPlan, RollingLatency, RTCPredictor,
    RuntimeConfig, SECOND_NS, TimeGrid, bootstrap_reference, checked_actions,
    frozen_array, rtc_reference,
)


class FakeClock:
    def __init__(self, now_ns=0):
        self.now_ns = now_ns

    def __call__(self):
        return self.now_ns

    def set(self, now_ns):
        self.now_ns = now_ns


class ManualPredictor:
    """Completion remains busy until polled, just like the real worker."""
    def __init__(self):
        self.requests = []
        self.results = deque()
        self._busy = False

    @property
    def busy(self):
        return self._busy

    def submit(self, request):
        if self._busy:
            raise RuntimeError("Already busy")
        self.requests.append(request)
        self._busy = True

    def complete(self, now_ns, actions=None, *, error=None):
        request = self.requests[-1]
        if error is not None:
            prediction = None
        else:
            if actions is None:
                actions = np.repeat(request.observation.state[None, :], 10, axis=0)
                actions[:, :6] += np.arange(10, dtype=np.float32)[:, None] * 0.001
            prediction = Prediction(request.request_id, frozen_array(actions),
                                    request.observation.reference_time_ns, now_ns)
        self.results.append(InferenceResult(request, prediction, now_ns, error))

    def poll(self):
        if not self.results:
            return None
        self._busy = False
        return self.results.popleft()


def feed(data, timestamp_ns, state=None, *, source_times=None):
    state = np.zeros(7, dtype=np.float32) if state is None else np.asarray(state, dtype=np.float32)
    times = (timestamp_ns,) * 3 if source_times is None else source_times
    data.append_camera("base", np.full((2, 3, 3), 11, dtype=np.uint8), times[0])
    data.append_camera("wrist", np.full((2, 3, 3), 22, dtype=np.uint8), times[1])
    data.append_state(state, times[2])


def prediction_plan(config, start, *, origin=None, request_id=1, offset=0.0):
    actions = np.zeros((config.horizon, 7), dtype=np.float32)
    actions[:, :6] = offset + np.arange(config.horizon)[:, None] * 0.001
    actions[:, 6] = 0.25
    prediction = Prediction(request_id, frozen_array(actions), 0, 0)
    return PredictionPlan(prediction, start if origin is None else origin,
                          start, start + config.execution_steps)


class TimeAndHistoryTests(unittest.TestCase):
    def test_grid_exact_boundaries_and_no_accumulated_rounding(self):
        grid = TimeGrid(123)
        self.assertEqual(grid.at(30_000), 1000 * SECOND_NS + 123)
        for index in (-100, -2, -1, 0, 1, 2, 3, 29, 30, 31, 99999):
            point = grid.at(index)
            self.assertEqual(grid.step_at_or_before(point), index)
            self.assertEqual(grid.first_at_or_after(point), index)
            self.assertEqual(grid.step_at_or_before(point - 1), index - 1)
            self.assertEqual(grid.first_at_or_after(point + 1), index + 1)

    def test_history_window_capacity_and_tie_prefers_earlier(self):
        history = BoundedHistory(100, 3)
        for timestamp in (0, 10, 20, 30):
            history.append(timestamp, timestamp)
        self.assertEqual([x.value for x in history.snapshot(30)], [10, 20, 30])
        self.assertEqual(history.nearest(25, 30).value, 20)
        self.assertEqual([x.value for x in history.snapshot(120)], [20, 30])
        self.assertIsNone(history.latest(131))
        history.append(140, 140)
        with self.assertRaises(ValueError):
            history.append(139, 139)

    def test_latency_sliding_mean_eviction_and_ceil(self):
        latency = RollingLatency(2)
        self.assertIsNone(latency.mean_age_ns)
        latency.add(10_000_000, 5_000_000)
        latency.add(30_000_000, 15_000_000)
        self.assertEqual(latency.mean_age_ns, 20_000_000)
        self.assertEqual(latency.delay_steps(30), 1)
        latency.add(70_000_000, 25_000_000)
        self.assertEqual(latency.mean_age_ns, 50_000_000)
        self.assertEqual(latency.mean_compute_ns, 20_000_000)
        self.assertEqual(latency.delay_steps(30), 2)
        with self.assertRaises(ValueError):
            latency.add(-1, 0)

    def test_invalid_runtime_config_and_actions(self):
        for kwargs in ({"fps": 29}, {"execution_steps": 10}, {"execution_steps": 0},
                       {"hold_steps": 7}, {"camera_capacity": 0}, {"max_skew_ns": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RuntimeConfig(**kwargs)
        for actions in (np.zeros((9, 7)), np.full((10, 7), np.nan)):
            with self.assertRaises(ValueError):
                checked_actions(actions, 10)

    def test_h40_runtime_config_and_action_shape(self):
        config = RuntimeConfig(horizon=40, execution_steps=20)
        self.assertEqual(config.horizon, 40)
        self.assertEqual(config.execution_steps, 20)
        actions = checked_actions(np.zeros((40, 7), dtype=np.float32), config.horizon)
        self.assertEqual(actions.shape, (40, 7))
        with self.assertRaises(ValueError):
            checked_actions(np.zeros((10, 7), dtype=np.float32), config.horizon)


class DataBufferTests(unittest.TestCase):
    def setUp(self):
        self.config = RuntimeConfig()
        self.data = DataBuffer(self.config)

    def test_buffers_own_read_only_numpy_memory(self):
        rgb = np.ones((2, 2, 3), dtype=np.uint8)
        state = np.arange(7, dtype=np.float32) * 0.01
        self.data.append_camera("base", rgb, 100)
        self.data.append_camera("wrist", rgb, 100)
        self.data.append_state(state, 100)
        rgb.fill(99)
        state.fill(0.5)
        observation = self.data.latest(100)
        self.assertEqual(observation.base_rgb[0, 0, 0], 1)
        self.assertAlmostEqual(float(observation.state[1]), 0.01)
        self.assertFalse(observation.base_rgb.flags.writeable)
        self.assertFalse(observation.state.flags.writeable)
        self.assertEqual(observation.policy_input("task")["prompt"], "task")

    def test_nearest_alignment_accepts_boundary_and_rejects_beyond(self):
        feed(self.data, 100_000_000, source_times=(70_000_000, 100_000_000, 130_000_000))
        observation = self.data.at(100_000_000, 130_000_000)
        self.assertIsNotNone(observation)
        self.assertEqual(observation.reference_time_ns, 100_000_000)
        self.assertIsNone(self.data.at(99_999_999, 130_000_000))

    def test_latest_does_not_require_alignment_but_requires_freshness(self):
        feed(self.data, 200_000_000, source_times=(0, 100_000_000, 200_000_000))
        observation = self.data.latest(200_000_000)
        self.assertIsNotNone(observation)
        self.assertEqual(observation.reference_time_ns, 200_000_000)
        self.assertIsNone(self.data.latest(200_000_001))
        self.assertIsNone(self.data.latest(199_999_999))  # A future source is invalid.

    def test_missing_and_invalid_sources(self):
        self.assertIsNone(self.data.latest(0))
        with self.assertRaises(ValueError):
            self.data.append_camera("wrong", np.zeros((2, 2, 3), dtype=np.uint8), 0)
        for rgb in (np.zeros((2, 2, 3), dtype=np.float32), np.zeros((2, 2), dtype=np.uint8)):
            with self.assertRaises(ValueError):
                self.data.append_camera("base", rgb, 0)
        for state in (np.zeros(6), np.full(7, np.inf), np.array([0, 0, 0, 0, 0, 0, 1.1])):
            with self.assertRaises(ValueError):
                self.data.append_state(state, 0)

    def test_state_stationarity_needs_history_not_empty_executor(self):
        self.data.append_state(np.zeros(7), 0)
        self.assertFalse(self.data.is_stationary(0))
        self.data.append_state(np.zeros(7), 100_000_000)
        self.data.append_state(np.zeros(7), 200_000_000)
        self.assertTrue(self.data.is_stationary(200_000_000))
        drift = np.zeros(7)
        drift[1] = 0.02
        self.data.append_state(drift, 210_000_000)
        self.assertFalse(self.data.is_stationary(210_000_000))
        self.assertFalse(self.data.unchanged(np.zeros(7), 210_000_000))
        self.assertFalse(self.data.unchanged(drift, 500_000_000))

    def test_camera_and_state_hard_caps(self):
        config = replace(self.config, camera_capacity=3, state_capacity=4)
        data = DataBuffer(config)
        for timestamp in range(20):
            feed(data, timestamp)
        self.assertEqual(len(data.base.snapshot(20)), 3)
        self.assertEqual(len(data.wrist.snapshot(20)), 3)
        self.assertEqual(len(data.states.snapshot(20)), 4)
        self.assertIsNone(data.latest(3 * SECOND_NS))

    def test_hold_latch_copies_reference_and_remembers_motion_return(self):
        reference = np.zeros(7, dtype=np.float32)
        self.data.append_state(reference, 0)
        self.data.begin_hold_check(reference)
        reference[0] = 0.5  # The latch must own its copy.
        self.data.append_state(np.zeros(7), 1)
        self.assertTrue(self.data.end_hold_check(1))
        self.data.begin_hold_check(np.zeros(7))
        moved = np.zeros(7)
        moved[0] = 0.02
        self.data.append_state(moved, 2)
        self.data.append_state(np.zeros(7), 3)
        self.assertTrue(self.data.unchanged(np.zeros(7), 3))
        self.assertFalse(self.data.end_hold_check(3))
        self.assertFalse(self.data.end_hold_check(3))  # Ending consumes the latch.

    def test_hold_latch_survives_history_eviction_without_expanding_buffer(self):
        config = replace(self.config, state_capacity=3)
        data = DataBuffer(config)
        data.append_state(np.zeros(7), 0)
        data.begin_hold_check(np.zeros(7))
        moved = np.zeros(7)
        moved[1] = 0.02
        data.append_state(moved, 1)
        for timestamp in (2, SECOND_NS, 2 * SECOND_NS, 3 * SECOND_NS):
            data.append_state(np.zeros(7), timestamp)
        samples = data.states.snapshot(3 * SECOND_NS)
        self.assertLessEqual(len(samples), 3)
        self.assertTrue(all(sample.timestamp_ns >= SECOND_NS for sample in samples))
        self.assertTrue(data.unchanged(np.zeros(7), 3 * SECOND_NS))
        self.assertFalse(data.end_hold_check(3 * SECOND_NS))

    def test_hold_latch_gripper_change_cancel_and_restart(self):
        self.data.append_state(np.zeros(7), 0)
        self.data.begin_hold_check(np.zeros(7))
        moved = np.zeros(7)
        moved[6] = 0.04
        self.data.append_state(moved, 1)
        self.data.append_state(np.zeros(7), 2)
        self.assertFalse(self.data.end_hold_check(2))
        self.data.begin_hold_check(np.zeros(7))
        self.data.cancel_hold_check()
        self.data.append_state(moved, 3)
        self.assertFalse(self.data.end_hold_check(3))
        self.data.begin_hold_check(moved)
        self.data.append_state(moved, 4)
        self.assertTrue(self.data.end_hold_check(4))


class ExecutorTests(unittest.TestCase):
    def setUp(self):
        self.config = RuntimeConfig()
        self.grid = TimeGrid(0)
        self.sink = NullCommandSink()
        self.executor = Executor(self.grid, self.config, self.sink)

    def test_no_early_execution_duplicate_or_future_consumption(self):
        self.executor.commit(prediction_plan(self.config, 1), 0)
        self.assertIsNone(self.executor.execute_due(0))
        self.assertIsNone(self.executor.execute_due(self.grid.at(1) - 1))
        self.assertEqual(self.executor.execute_due(self.grid.at(1)), 1)
        self.assertIsNone(self.executor.execute_due(self.grid.at(1)))
        self.assertIsNone(self.executor.execute_due(self.grid.at(2) - 1))
        self.assertEqual(self.executor.pending_count, 3)
        self.assertEqual(len(self.sink.targets), 1)
        self.assertAlmostEqual(self.sink.targets[-1][1], 0.1001 * 0.75)

    def test_skips_expired_slots_without_catchup(self):
        self.executor.commit(prediction_plan(self.config, 1), 0)
        self.assertEqual(self.executor.execute_due(self.grid.at(3)), 3)
        self.assertEqual(self.executor.skipped_count, 2)
        self.assertEqual(self.executor.sent_count, 1)
        self.assertIsNone(self.executor.action_at(1))
        self.assertEqual([event.status for event in self.executor.events], ["expired", "expired", "sent"])

    def test_commit_is_atomic_and_rejects_overlaps_past_and_bad_horizon(self):
        plan = prediction_plan(self.config, 1)
        self.executor.commit(plan, 0)
        for invalid, now in ((plan, 0), (prediction_plan(self.config, 5), self.grid.at(5)),
                             (replace(plan, end_step=plan.end_step + 1), 0),
                             (replace(plan, origin_step=20), 0)):
            with self.subTest(plan=invalid), self.assertRaises(ValueError):
                self.executor.commit(invalid, now)
            self.assertEqual(self.executor.pending_count, 4)

    def test_future_capacity_and_history_capacity(self):
        for start in (1, 5, 9, 13, 17):
            self.executor.commit(prediction_plan(self.config, start, request_id=start), 0)
        self.assertEqual(self.executor.pending_count, 20)
        with self.assertRaises(OverflowError):
            self.executor.commit(prediction_plan(self.config, 21), 0)
        self.assertEqual(self.executor.pending_count, 20)
        for step in range(1, 21):
            self.assertEqual(self.executor.execute_due(self.grid.at(step)), step)
        self.assertIsNone(self.executor.action_at(10))
        self.assertIsNotNone(self.executor.action_at(11))
        self.assertEqual(self.executor.pending_count, 0)

    def test_sink_error_discards_future_actions(self):
        class BrokenSink:
            def send_target(self, *_):
                raise RuntimeError("test sink error")
        executor = Executor(self.grid, self.config, BrokenSink())
        executor.commit(prediction_plan(self.config, 1), 0)
        self.assertIsNone(executor.execute_due(self.grid.at(1)))
        self.assertIsInstance(executor.last_error, RuntimeError)
        self.assertEqual(executor.pending_count, 0)
        self.assertIsNone(executor.execute_due(self.grid.at(2)))
        with self.assertRaises(RuntimeError):
            executor.commit(prediction_plan(self.config, 5), 0)

    def test_close_prevents_execution_and_new_commits(self):
        self.executor.commit(prediction_plan(self.config, 1), 0)
        self.executor.close()
        self.assertIsNone(self.executor.execute_due(self.grid.at(1)))
        self.assertEqual(self.executor.pending_count, 0)
        with self.assertRaises(RuntimeError):
            self.executor.commit(prediction_plan(self.config, 5), 0)

    def test_gripper_clipped_but_joint_angles_not_renormalized(self):
        plan = prediction_plan(self.config, 1)
        actions = np.array(plan.prediction.actions, copy=True)
        actions[0, :6] = 1.234
        actions[0, 6] = 2
        plan = replace(plan, prediction=replace(plan.prediction, actions=frozen_array(actions)))
        self.executor.commit(plan, 0)
        self.executor.execute_due(self.grid.at(1))
        np.testing.assert_allclose(self.sink.targets[-1][0], 1.234)
        self.assertEqual(self.sink.targets[-1][1], 0)


class ReferenceTests(unittest.TestCase):
    def test_bootstrap_short_prefix_and_physical_state(self):
        config = RuntimeConfig()
        data = DataBuffer(config)
        feed(data, 0, np.full(7, 0.25))
        actions, weights = bootstrap_reference(data.latest(0), config)
        np.testing.assert_allclose(actions, 0.25)
        np.testing.assert_allclose(weights[0], 1)
        np.testing.assert_allclose(weights[1:], 0)
        self.assertFalse(actions.flags.writeable)

    def test_hard_prefix_uses_actual_multichunk_commands_soft_tail_full_prediction(self):
        config = RuntimeConfig()
        executor = Executor(TimeGrid(0), config, NullCommandSink())
        older = prediction_plan(config, 1, offset=0.1, request_id=1)
        previous = prediction_plan(config, 5, offset=0.2, request_id=2)
        executor.commit(older, 0)
        executor.commit(previous, 0)
        for step in range(1, 4):
            executor.execute_due(executor.grid.at(step))
        actions, weights = rtc_reference(3, 9, previous, executor, config)
        np.testing.assert_allclose(actions[0], older.action_at(3))
        np.testing.assert_allclose(actions[1], older.action_at(4))
        np.testing.assert_allclose(actions[2], previous.action_at(5))
        np.testing.assert_allclose(weights[:6], 1)
        np.testing.assert_allclose(actions[6], previous.action_at(9))
        self.assertTrue(np.all((weights[6:] > 0) & (weights[6:] < 1)))
        self.assertTrue(np.all(np.diff(weights[6:, 0]) < 0))

    def test_tail_weights_zero_beyond_old_full_horizon(self):
        config = RuntimeConfig()
        executor = Executor(TimeGrid(0), config, NullCommandSink())
        previous = prediction_plan(config, 5, origin=1)
        executor.commit(previous, 0)
        actions, weights = rtc_reference(5, 9, previous, executor, config)
        np.testing.assert_allclose(weights[:4], 1)
        self.assertTrue(np.all(weights[4:6] > 0))
        np.testing.assert_allclose(weights[6:], 0)
        np.testing.assert_allclose(actions[6:], 0)

    def test_missing_committed_prefix_is_rejected_not_invented(self):
        config = RuntimeConfig()
        executor = Executor(TimeGrid(0), config, NullCommandSink())
        previous = prediction_plan(config, 5)
        executor.commit(previous, 0)
        with self.assertRaisesRegex(ValueError, "Missing committed"):
            rtc_reference(4, 9, previous, executor, config)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.config = RuntimeConfig()
        self.clock = FakeClock()
        self.grid = TimeGrid(0)
        self.data = DataBuffer(self.config)
        self.predictor = ManualPredictor()
        self.executor = Executor(self.grid, self.config, NullCommandSink())
        self.controller = Controller(self.data, self.predictor, self.executor,
                                     self.grid, self.config, "place the block")

    def tick(self, now_ns, *, execute=False, state=None):
        self.clock.set(now_ns)
        feed(self.data, now_ns, state)
        if execute:
            self.executor.execute_due(now_ns)
        self.controller.tick(now_ns)

    def bootstrap(self, delay_ns=60_000_000):
        self.tick(0)
        self.assertEqual(self.predictor.requests[-1].mode, "WARMUP")
        for index, expected in ((1, "CALIBRATION"), (2, "BOOTSTRAP")):
            self.predictor.complete(index * delay_ns)
            self.tick(index * delay_ns)
            self.assertEqual(self.predictor.requests[-1].mode, expected)
            self.assertEqual(self.executor.pending_count, 0)
        self.predictor.complete(3 * delay_ns)
        self.tick(3 * delay_ns)

    def test_warmup_excluded_and_normal_early_result_switch_fixed(self):
        self.bootstrap()
        self.assertEqual(self.controller.phase, "RTC")
        self.assertEqual(self.controller.completed_predictions, 2)
        self.assertEqual(self.controller.latency.mean_age_ns, 60_000_000)
        first = self.controller.plan
        self.assertEqual(first.start_step, 6)
        self.assertEqual(first.origin_step, 5)
        self.assertEqual(self.executor.pending_count, 4)
        self.tick(self.grid.at(6), execute=True)
        request = self.predictor.requests[-1]
        self.assertEqual(request.mode, "RTC")
        self.assertEqual(request.origin_step, 6)
        self.assertEqual(request.switch_step, 10)
        self.assertEqual(request.local_start, 4)
        np.testing.assert_allclose(request.reference_weights[:4], 1)
        self.tick(self.grid.at(7), execute=True)
        self.predictor.complete(260_000_000)
        self.tick(260_000_000, execute=True)
        next_plan = self.controller.plan
        self.assertEqual(next_plan.start_step, 10)
        self.assertEqual(next_plan.origin_step, 6)
        self.assertEqual(next_plan.end_step, 14)
        self.assertEqual(self.executor.sent_count, 2)
        self.assertIsNone(self.executor.execute_due(self.grid.at(8) - 1))
        for index in (8, 9, 10):
            self.tick(self.grid.at(index), execute=True)
        self.assertEqual(self.executor.sent_count, 5)
        np.testing.assert_allclose(self.executor.action_at(10), next_plan.prediction.actions[4])
        self.assertEqual(self.predictor.requests[-1].origin_step, 10)
        self.assertEqual(self.predictor.requests[-1].switch_step, 14)

    def test_late_result_discarded_drain_waits_for_worker_and_static_history(self):
        self.bootstrap()
        self.tick(self.grid.at(6), execute=True)
        late_request = self.predictor.requests[-1]
        for index in (7, 8, 9):
            self.tick(self.grid.at(index), execute=True)
        self.tick(self.grid.at(10) - self.config.prepare_margin_ns)
        self.assertEqual(self.controller.phase, "DRAIN")
        self.assertEqual(self.controller.overruns, 1)
        self.tick(self.grid.at(10), execute=True)
        self.assertEqual(self.executor.pending_count, 0)
        self.assertTrue(self.predictor.busy)
        self.predictor.complete(350_000_000)
        self.tick(350_000_000)
        self.assertEqual(self.controller.phase, "BOOTSTRAP")
        self.assertEqual(self.executor.pending_count, 0)
        self.assertEqual(self.controller.recoveries, 1)
        self.assertNotEqual(self.predictor.requests[-1].request_id, late_request.request_id)
        self.assertEqual(self.predictor.requests[-1].mode, "BOOTSTRAP")
        self.assertEqual(self.executor.sent_count, 4)

    def test_persistent_slow_prediction_still_executes_sync_segments(self):
        delay = 250_000_000
        self.tick(0)
        for now in range(50_000_000, 751_000_000, 50_000_000):
            if now in (delay, 2 * delay, 3 * delay):
                self.predictor.complete(now)
            self.tick(now)
        self.assertEqual(self.controller.phase, "DRAIN")
        first = self.controller.plan
        for index in range(first.start_step, first.end_step + 1):
            self.tick(self.grid.at(index), execute=True)
        self.assertEqual(self.executor.sent_count, 4)
        self.assertEqual(self.controller.phase, "BOOTSTRAP")
        started = self.predictor.requests[-1].observation.reference_time_ns
        for now in range(started + 50_000_000, started + delay, 50_000_000):
            self.tick(now)
        self.predictor.complete(started + delay)
        self.tick(started + delay)
        self.assertEqual(self.controller.phase, "DRAIN")
        second = self.controller.plan
        self.assertGreater(second.start_step, first.end_step)
        for index in range(second.start_step, second.end_step):
            self.tick(self.grid.at(index), execute=True)
        self.assertEqual(self.executor.sent_count, 8)

    def test_bootstrap_state_drift_does_not_commit(self):
        self.tick(0)
        self.predictor.complete(60_000_000)
        self.tick(60_000_000)
        self.predictor.complete(120_000_000)
        self.tick(120_000_000)
        self.predictor.complete(180_000_000)
        drift = np.zeros(7)
        drift[0] = 0.02
        self.tick(180_000_000, state=drift)
        self.assertEqual(self.controller.phase, "BOOTSTRAP")
        self.assertEqual(self.executor.pending_count, 0)
        self.assertFalse(self.predictor.busy)
        self.assertTrue(any("状态改变" in event for event in self.controller.events))
        self.tick(430_000_000, state=drift)
        self.assertTrue(self.predictor.busy)
        np.testing.assert_allclose(self.predictor.requests[-1].observation.state, drift)

    def test_bootstrap_motion_return_does_not_commit(self):
        self.tick(0)
        for now in (60_000_000, 120_000_000):
            self.predictor.complete(now)
            self.tick(now)
        drift = np.zeros(7)
        drift[0] = 0.02
        self.tick(140_000_000, state=drift)
        self.predictor.complete(180_000_000)
        self.tick(180_000_000)
        self.assertTrue(self.data.unchanged(np.zeros(7), 180_000_000))
        self.assertEqual(self.controller.phase, "BOOTSTRAP")
        self.assertEqual(self.executor.pending_count, 0)
        self.assertIsNone(self.controller.plan)
        self.assertTrue(any("状态改变" in event for event in self.controller.events))

    def test_long_bootstrap_motion_return_stays_rejected_after_eviction(self):
        self.tick(0)
        for now in (60_000_000, 120_000_000):
            self.predictor.complete(now)
            self.tick(now)
        drift = np.zeros(7)
        drift[0] = 0.02
        self.tick(150_000_000, state=drift)
        for now in range(175_000_000, 3_120_000_000, 50_000_000):
            self.tick(now)
        self.predictor.complete(3_120_000_000)
        self.tick(3_120_000_000)
        self.assertTrue(all(sample.timestamp_ns > 150_000_000
                            for sample in self.data.states.snapshot(3_120_000_000)))
        self.assertTrue(self.data.unchanged(np.zeros(7), 3_120_000_000))
        self.assertEqual(self.controller.phase, "BOOTSTRAP")
        self.assertEqual(self.executor.pending_count, 0)
        self.assertIsNone(self.controller.plan)

    def test_long_bootstrap_without_motion_still_commits_sync_segment(self):
        self.tick(0)
        for now in (60_000_000, 120_000_000):
            self.predictor.complete(now)
            self.tick(now)
        for now in range(150_000_000, 3_120_000_000, 50_000_000):
            self.tick(now)
        self.predictor.complete(3_120_000_000)
        self.tick(3_120_000_000)
        self.assertEqual(self.controller.phase, "DRAIN")
        self.assertIsNotNone(self.controller.plan)
        self.assertEqual(self.executor.pending_count, 4)
        for index in range(self.controller.plan.start_step, self.controller.plan.end_step):
            self.tick(self.grid.at(index), execute=True)
        self.assertEqual(self.executor.sent_count, 4)

    def test_bootstrap_exception_cancels_hold_monitor(self):
        self.tick(0)
        for now in (60_000_000, 120_000_000):
            self.predictor.complete(now)
            self.tick(now)
        self.predictor.complete(180_000_000, error=RuntimeError("boot failure"))
        self.tick(180_000_000)
        self.assertEqual(self.executor.pending_count, 0)
        self.assertFalse(self.data.end_hold_check(180_000_000))
        self.tick(430_000_000)
        self.assertEqual(self.predictor.requests[-1].mode, "BOOTSTRAP")
        self.assertTrue(self.predictor.busy)

    def test_stop_during_bootstrap_cancels_hold_monitor(self):
        self.tick(0)
        for now in (60_000_000, 120_000_000):
            self.predictor.complete(now)
            self.tick(now)
        self.controller.stop()
        self.assertFalse(self.data.end_hold_check(120_000_000))
        self.predictor.complete(180_000_000)
        self.tick(180_000_000)
        self.assertEqual(self.controller.phase, "STOPPED")
        self.assertEqual(self.executor.pending_count, 0)

    def test_bootstrap_discontinuous_first_joint_target_retries(self):
        self.tick(0)
        for now in (60_000_000, 120_000_000):
            self.predictor.complete(now)
            self.tick(now)
        actions = np.zeros((10, 7))
        actions[1, 0] = 0.1
        self.predictor.complete(180_000_000, actions)
        self.tick(180_000_000)
        self.assertEqual(self.executor.pending_count, 0)
        self.assertTrue(any("不连续" in event for event in self.controller.events))

    def test_missing_inputs_do_not_submit_and_missed_rtc_slot_drains(self):
        self.controller.tick(0)
        self.assertFalse(self.predictor.busy)
        self.bootstrap()
        # Intentionally do not tick until the trigger slot has already passed.
        self.controller.tick(self.grid.at(self.controller.next_origin) + 34_000_000)
        self.assertEqual(self.controller.phase, "DRAIN")
        self.assertEqual(self.predictor.requests[-1].mode, "BOOTSTRAP")

    def test_one_source_outside_alignment_retries_then_drains(self):
        self.data.config = replace(self.config, max_skew_ns=1_000_000)
        self.bootstrap()
        target = self.grid.at(self.controller.next_origin)
        self.controller.tick(target)
        self.assertFalse(self.predictor.busy)
        self.assertEqual(self.controller.phase, "RTC")
        # Keep CAN/base alive but provide no new wrist frame.
        self.data.append_state(np.zeros(7), target + 1_000_000)
        self.data.append_camera("base", np.zeros((2, 2, 3), dtype=np.uint8), target)
        self.controller.tick(target + 1_000_000)
        self.assertFalse(self.predictor.busy)
        self.assertEqual(self.controller.phase, "RTC")
        self.controller.tick(target + 34_000_000)
        self.assertEqual(self.controller.phase, "DRAIN")
        self.assertEqual(self.executor.pending_count, 4)

    def test_rolling_latency_recovers_from_slow_sync_to_rtc(self):
        self.controller.latency = RollingLatency(2)
        delay = 250_000_000
        self.tick(0)
        for now in range(50_000_000, 751_000_000, 50_000_000):
            if now in (delay, 2 * delay, 3 * delay):
                self.predictor.complete(now)
            self.tick(now)
        self.assertEqual(self.controller.phase, "DRAIN")
        # Two fresh fast BOOTSTRAP calls evict the two old slow measurements.
        for iteration in range(2):
            plan = self.controller.plan
            for index in range(plan.start_step, plan.end_step + 1):
                self.tick(self.grid.at(index), execute=True)
            self.assertEqual(self.controller.phase, "BOOTSTRAP")
            sent = self.predictor.requests[-1].observation.reference_time_ns
            self.tick(sent + 25_000_000)
            self.tick(sent + 50_000_000)
            self.predictor.complete(sent + 60_000_000)
            self.tick(sent + 60_000_000)
            self.assertEqual(self.controller.phase, "DRAIN" if iteration == 0 else "RTC")
        self.assertEqual(self.controller.latency.mean_age_ns, 60_000_000)
        self.assertEqual(self.controller.latency.delay_steps(30), 2)
        self.tick(self.grid.at(self.controller.next_origin), execute=True)
        self.assertEqual(self.predictor.requests[-1].mode, "RTC")

    def test_result_at_prepare_deadline_is_rejected(self):
        self.bootstrap()
        self.tick(self.grid.at(6), execute=True)
        request = self.predictor.requests[-1]
        deadline = self.grid.at(request.switch_step) - self.config.prepare_margin_ns
        self.predictor.complete(deadline)
        self.tick(deadline)
        self.assertEqual(self.controller.phase, "DRAIN")
        self.assertEqual(self.controller.overruns, 1)
        self.assertEqual(self.controller.plan.prediction.request_id, request.request_id - 1)

    def test_failed_warmup_retries_after_bounded_delay(self):
        self.tick(0)
        self.predictor.complete(1_000_000, error=RuntimeError("warmup failure"))
        self.tick(1_000_000)
        self.assertEqual(self.controller.phase, "WARMUP")
        self.assertFalse(self.predictor.busy)
        self.tick(250_000_000)
        self.assertFalse(self.predictor.busy)
        self.tick(251_000_000)
        self.assertTrue(self.predictor.busy)
        self.assertEqual(len(self.predictor.requests), 2)
        self.assertEqual(self.predictor.requests[-1].mode, "WARMUP")

    def test_rtc_exception_drains_and_sink_failure_stops(self):
        self.bootstrap()
        self.tick(self.grid.at(6), execute=True)
        self.predictor.complete(220_000_000, error=RuntimeError("fake model error"))
        self.tick(220_000_000)
        self.assertEqual(self.controller.phase, "DRAIN")
        self.assertEqual(self.executor.pending_count, 3)
        self.executor.last_error = RuntimeError("fake sink error")
        self.tick(230_000_000)
        self.assertEqual(self.controller.phase, "STOPPED")
        self.assertEqual(self.executor.pending_count, 0)

    def test_stop_does_not_accept_completed_old_prediction(self):
        self.bootstrap()
        self.tick(self.grid.at(6), execute=True)
        self.controller.stop()
        self.predictor.complete(260_000_000)
        self.tick(260_000_000)
        self.assertEqual(self.controller.phase, "STOPPED")
        self.assertEqual(self.executor.pending_count, 0)

    def test_delay_four_slots_conservatively_uses_sync_fallback(self):
        self.bootstrap(120_000_000)
        self.assertEqual(self.controller.latency.delay_steps(30), 4)
        self.assertEqual(self.controller.phase, "DRAIN")
        self.assertEqual(self.executor.pending_count, 4)


class PredictorWorkerTests(unittest.TestCase):
    def request(self, request_id=1):
        config = RuntimeConfig()
        data = DataBuffer(config)
        feed(data, 0)
        observation = data.latest(0)
        actions, weights = bootstrap_reference(observation, config)
        return InferenceRequest(request_id, "BOOTSTRAP", observation, "task", actions, weights)

    @staticmethod
    def await_result(predictor):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            result = predictor.poll()
            if result is not None:
                return result
            time.sleep(0.001)
        raise AssertionError("Worker did not publish a result")

    def test_single_worker_bounded_queue_schema_and_completion(self):
        started = threading.Event()
        release = threading.Event()

        class Backend:
            payload = None

            def infer(self, payload):
                self.payload = payload
                started.set()
                if not release.wait(1):
                    raise RuntimeError("test release timeout")
                return {"actions": np.zeros((10, 7)), "policy_timing": {"infer_ms": 12}}

            def close(self):
                release.set()

        backend = Backend()
        predictor = RTCPredictor(backend, 10, clock_ns=FakeClock(25))
        try:
            predictor.submit(self.request())
            self.assertTrue(started.wait(1))
            self.assertTrue(predictor.busy)
            with self.assertRaises(RuntimeError):
                predictor.submit(self.request(2))
            release.set()
            result = self.await_result(predictor)
            self.assertFalse(predictor.busy)
            self.assertIsNone(result.error)
            self.assertEqual(result.prediction.actions.shape, (10, 7))
            self.assertFalse(result.prediction.actions.flags.writeable)
            self.assertEqual(result.prediction.model_infer_ms, 12)
            self.assertIn("rtc/reference_actions", backend.payload)
            self.assertIn("rtc/reference_weights", backend.payload)
        finally:
            predictor.close()
        with self.assertRaises(RuntimeError):
            predictor.submit(self.request(3))

    def test_bad_model_output_becomes_error_completion(self):
        class Backend:
            def infer(self, payload):
                return {"actions": np.full((10, 7), np.nan)}

            def close(self):
                pass

        predictor = RTCPredictor(Backend(), 10, clock_ns=FakeClock(25))
        try:
            predictor.submit(self.request())
            result = self.await_result(predictor)
            self.assertIsInstance(result.error, ValueError)
            self.assertIsNone(result.prediction)
            self.assertFalse(predictor.busy)
        finally:
            predictor.close()

    def test_close_interrupts_inflight_backend_without_unbounded_wait(self):
        started = threading.Event()
        release = threading.Event()

        class Backend:
            def infer(self, payload):
                started.set()
                release.wait(2)
                return {"actions": np.zeros((10, 7))}

            def close(self):
                release.set()

        predictor = RTCPredictor(Backend(), 10)
        predictor.submit(self.request())
        self.assertTrue(started.wait(1))
        before = time.monotonic()
        predictor.close()
        self.assertLess(time.monotonic() - before, 0.5)
        self.assertIsNone(predictor.poll())
        predictor.close()  # Idempotent.


if __name__ == "__main__":
    unittest.main()
