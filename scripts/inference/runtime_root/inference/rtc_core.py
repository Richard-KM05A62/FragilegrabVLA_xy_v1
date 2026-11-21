"""Hardware-independent data, timing, inference and execution primitives."""
from __future__ import annotations

import math
import queue
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Callable, Generic, Protocol, TypeVar

import numpy as np

SECOND_NS = 1_000_000_000
ACTION_DIM = 7
T = TypeVar("T")


def frozen_array(value: Any, *, dtype: Any = np.float32) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def checked_actions(value: Any, horizon: int) -> np.ndarray:
    result = frozen_array(value)
    if result.shape != (horizon, ACTION_DIM) or not np.isfinite(result).all():
        raise ValueError(f"Expected finite absolute actions [{horizon},7], got {result.shape}")
    return result


@dataclass(frozen=True)
class RuntimeConfig:
    fps: int = 30
    horizon: int = 10
    execution_steps: int = 4
    hold_steps: int = 1
    latency_window: int = 20
    buffer_window_ns: int = 2 * SECOND_NS
    camera_capacity: int = 90
    state_capacity: int = 256
    max_skew_ns: int = 30_000_000
    max_input_age_ns: int = 200_000_000
    prepare_margin_ns: int = 2_000_000
    stationary_window_ns: int = 200_000_000
    stationary_joint_tolerance: float = 0.01
    stationary_gripper_tolerance: float = 0.03
    bootstrap_max_joint_jump: float = 0.08
    retry_delay_ns: int = 250_000_000
    open_width_m: float = 0.1001

    def __post_init__(self) -> None:
        if self.fps != 30:
            raise ValueError("This Piper checkpoint requires 30 Hz")
        if not 1 <= self.execution_steps < self.horizon:
            raise ValueError("execution_steps must be in [1,horizon)")
        if not 1 <= self.hold_steps <= self.horizon - self.execution_steps:
            raise ValueError("hold_steps must leave enough free actions")
        positive = (self.latency_window, self.buffer_window_ns, self.camera_capacity,
                    self.state_capacity, self.max_input_age_ns, self.stationary_window_ns,
                    self.stationary_joint_tolerance, self.stationary_gripper_tolerance,
                    self.bootstrap_max_joint_jump, self.open_width_m)
        if any(not np.isfinite(v) or v <= 0 for v in positive):
            raise ValueError("Buffer sizes, windows and tolerances must be positive")
        if min(self.max_skew_ns, self.prepare_margin_ns, self.retry_delay_ns) < 0:
            raise ValueError("Time tolerances must be nonnegative")


@dataclass(frozen=True)
class TimeGrid:
    epoch_ns: int
    fps: int = 30

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("Time-grid frequency must be positive")

    def at(self, step: int) -> int:
        # Compute from the epoch, rather than accumulating a rounded period.
        return self.epoch_ns + (step * SECOND_NS + self.fps // 2) // self.fps

    def step_at_or_before(self, now_ns: int) -> int:
        step = (now_ns - self.epoch_ns) * self.fps // SECOND_NS
        while self.at(step) > now_ns:
            step -= 1
        while self.at(step + 1) <= now_ns:
            step += 1
        return step

    def first_at_or_after(self, now_ns: int) -> int:
        step = self.step_at_or_before(now_ns)
        return step if self.at(step) == now_ns else step + 1


@dataclass(frozen=True)
class TimedSample(Generic[T]):
    timestamp_ns: int
    value: T


class BoundedHistory(Generic[T]):
    def __init__(self, window_ns: int, capacity: int) -> None:
        if window_ns <= 0 or capacity <= 0:
            raise ValueError("History window and capacity must be positive")
        self.window_ns = window_ns
        self._items: deque[TimedSample[T]] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def append(self, value: T, timestamp_ns: int) -> None:
        with self._lock:
            if self._items and timestamp_ns < self._items[-1].timestamp_ns:
                raise ValueError("Source timestamps must be monotonic")
            self._items.append(TimedSample(timestamp_ns, value))
            self._trim(timestamp_ns)

    def _trim(self, now_ns: int) -> None:
        cutoff = now_ns - self.window_ns
        while self._items and self._items[0].timestamp_ns < cutoff:
            self._items.popleft()

    def snapshot(self, now_ns: int) -> tuple[TimedSample[T], ...]:
        with self._lock:
            self._trim(now_ns)
            return tuple(self._items)

    def latest(self, now_ns: int) -> TimedSample[T] | None:
        items = self.snapshot(now_ns)
        return items[-1] if items else None

    def nearest(self, target_ns: int, now_ns: int) -> TimedSample[T] | None:
        items = self.snapshot(now_ns)
        # Equal distances prefer the earlier sample.
        return min(items, key=lambda x: (abs(x.timestamp_ns-target_ns), x.timestamp_ns)) if items else None


@dataclass(frozen=True)
class Observation:
    observation_id: int
    reference_time_ns: int
    base_rgb: np.ndarray
    wrist_rgb: np.ndarray
    state: np.ndarray
    base_time_ns: int
    wrist_time_ns: int
    state_time_ns: int

    def policy_input(self, prompt: str) -> dict[str, Any]:
        return {"observation/image": self.base_rgb,
                "observation/wrist_image": self.wrist_rgb,
                "observation/state": self.state, "prompt": prompt}


class DataBuffer:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.base: BoundedHistory[np.ndarray] = BoundedHistory(config.buffer_window_ns, config.camera_capacity)
        self.wrist: BoundedHistory[np.ndarray] = BoundedHistory(config.buffer_window_ns, config.camera_capacity)
        self.states: BoundedHistory[np.ndarray] = BoundedHistory(config.buffer_window_ns, config.state_capacity)
        self._observation_id = 0
        self._hold_lock = threading.Lock()
        self._hold_reference: np.ndarray | None = None
        self._hold_changed = False

    def append_camera(self, source: str, rgb: np.ndarray, timestamp_ns: int) -> None:
        if source not in ("base", "wrist"):
            raise ValueError(f"Unknown camera {source}")
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Camera input must be RGB uint8 HWC")
        getattr(self, source).append(frozen_array(rgb, dtype=np.uint8), timestamp_ns)

    def append_state(self, state: np.ndarray, timestamp_ns: int) -> None:
        value = frozen_array(state)
        if value.shape != (7,) or not np.isfinite(value).all() or not 0 <= value[6] <= 1:
            raise ValueError("State must be finite [q1..q6 rad, closedness in 0..1]")
        self.states.append(value, timestamp_ns)
        with self._hold_lock:
            if self._hold_reference is not None:
                self._hold_changed |= not self._near_state(value, self._hold_reference)

    def _near_state(self, state: np.ndarray, reference: np.ndarray) -> bool:
        return bool(np.max(np.abs(state[:6]-reference[:6])) <= self.config.stationary_joint_tolerance
                    and abs(float(state[6]-reference[6])) <= self.config.stationary_gripper_tolerance)

    def begin_hold_check(self, reference: np.ndarray) -> None:
        """Single bounded latch catches motion-and-return even across buffer eviction."""
        with self._hold_lock:
            self._hold_reference = frozen_array(reference)
            self._hold_changed = False

    def end_hold_check(self, now_ns: int) -> bool:
        with self._hold_lock:
            reference, changed = self._hold_reference, self._hold_changed
            self._hold_reference = None
        return bool(reference is not None and not changed and self.unchanged(reference, now_ns))

    def cancel_hold_check(self) -> None:
        with self._hold_lock:
            self._hold_reference = None

    def _observation(self, reference_ns: int, now_ns: int, nearest: bool) -> Observation | None:
        histories = (self.base, self.wrist, self.states)
        samples = [h.nearest(reference_ns, now_ns) if nearest else h.latest(now_ns) for h in histories]
        if any(sample is None for sample in samples):
            return None
        base, wrist, state = samples
        assert base is not None and wrist is not None and state is not None
        if any(not 0 <= now_ns - sample.timestamp_ns <= self.config.max_input_age_ns for sample in samples):
            return None
        if nearest and any(abs(sample.timestamp_ns-reference_ns) > self.config.max_skew_ns for sample in samples):
            return None
        self._observation_id += 1
        return Observation(self._observation_id, reference_ns, base.value, wrist.value, state.value,
                           base.timestamp_ns, wrist.timestamp_ns, state.timestamp_ns)

    def latest(self, now_ns: int) -> Observation | None:
        return self._observation(now_ns, now_ns, False)

    def at(self, target_ns: int, now_ns: int) -> Observation | None:
        return self._observation(target_ns, now_ns, True)

    def unchanged(self, reference: np.ndarray, now_ns: int) -> bool:
        latest = self.states.latest(now_ns)
        return bool(latest is not None
                    and 0 <= now_ns-latest.timestamp_ns <= self.config.max_input_age_ns
                    and self._near_state(latest.value, reference))

    def is_stationary(self, now_ns: int) -> bool:
        samples = self.states.snapshot(now_ns)
        recent = [x for x in samples if x.timestamp_ns >= now_ns-self.config.stationary_window_ns]
        if len(recent) < 3 or now_ns-recent[-1].timestamp_ns > self.config.max_input_age_ns:
            return False
        if recent[-1].timestamp_ns-recent[0].timestamp_ns < self.config.stationary_window_ns * 0.8:
            return False
        values = np.stack([x.value for x in recent])
        return bool(np.max(np.ptp(values[:, :6], axis=0)) <= self.config.stationary_joint_tolerance
                    and np.ptp(values[:, 6]) <= self.config.stationary_gripper_tolerance)


class RollingLatency:
    """Finite sliding mean in nanoseconds; warmup samples are excluded by caller."""
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("Latency capacity must be positive")
        self._samples: deque[tuple[int, int]] = deque()
        self._capacity = capacity
        self._age_sum = self._compute_sum = 0

    def add(self, age_ns: int, compute_ns: int) -> None:
        if min(age_ns, compute_ns) < 0:
            raise ValueError("Latency cannot be negative")
        if len(self._samples) == self._capacity:
            age, compute = self._samples.popleft()
            self._age_sum -= age
            self._compute_sum -= compute
        self._samples.append((age_ns, compute_ns))
        self._age_sum += age_ns
        self._compute_sum += compute_ns

    @property
    def mean_age_ns(self) -> float | None:
        return self._age_sum/len(self._samples) if self._samples else None

    @property
    def mean_compute_ns(self) -> float | None:
        return self._compute_sum/len(self._samples) if self._samples else None

    def delay_steps(self, fps: int) -> int | None:
        mean = self.mean_age_ns
        return math.ceil(mean*fps/SECOND_NS) if mean is not None else None


@dataclass(frozen=True)
class InferenceRequest:
    request_id: int
    mode: str
    observation: Observation
    prompt: str
    reference_actions: np.ndarray
    reference_weights: np.ndarray
    origin_step: int | None = None
    switch_step: int | None = None
    local_start: int = 0


@dataclass(frozen=True)
class Prediction:
    request_id: int
    actions: np.ndarray
    request_sent_ns: int
    received_ns: int
    model_infer_ms: float | None = None


@dataclass(frozen=True)
class InferenceResult:
    request: InferenceRequest
    prediction: Prediction | None
    completed_ns: int
    error: Exception | None = None


class PolicyBackend(Protocol):
    def infer(self, payload: dict[str, Any]) -> dict[str, Any]: ...
    def close(self) -> None: ...


class RTCPredictor:
    """A bounded, daemon worker; no uninterruptible ThreadPool shutdown."""
    def __init__(self, backend: PolicyBackend, horizon: int, *, clock_ns: Callable[[], int] = time.monotonic_ns) -> None:
        self.backend, self.horizon, self.clock_ns = backend, horizon, clock_ns
        self._jobs: queue.Queue[InferenceRequest | None] = queue.Queue(maxsize=1)
        self._results: queue.Queue[InferenceResult] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._busy = False
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._worker, name="RTCInference", daemon=True)
        self._thread.start()

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def submit(self, request: InferenceRequest) -> None:
        with self._lock:
            if self._closed.is_set() or self._busy:
                raise RuntimeError("Predictor is closed or busy")
            self._busy = True
            self._jobs.put_nowait(request)

    def poll(self) -> InferenceResult | None:
        try:
            result = self._results.get_nowait()
        except queue.Empty:
            return None
        with self._lock:
            self._busy = False
        return result

    def _worker(self) -> None:
        while not self._closed.is_set():
            request = self._jobs.get()
            if request is None or self._closed.is_set():
                break
            sent_ns = self.clock_ns()
            try:
                payload = request.observation.policy_input(request.prompt)
                payload["rtc/reference_actions"] = request.reference_actions
                payload["rtc/reference_weights"] = request.reference_weights
                payload["rtc/request_id"] = request.request_id
                response = self.backend.infer(payload)
                if response.get("rtc/request_id", request.request_id) != request.request_id:
                    raise ValueError("RTC response request id mismatch")
                actions = checked_actions(response["actions"], self.horizon)
                timing = response.get("policy_timing", {}).get("infer_ms")
                ready_ns = self.clock_ns()
                prediction = Prediction(request.request_id, actions, sent_ns, ready_ns, timing)
                result = InferenceResult(request, prediction, ready_ns)
            except Exception as error:
                result = InferenceResult(request, None, self.clock_ns(), error)
            if not self._closed.is_set():
                self._results.put_nowait(result)

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        try:
            self.backend.close()
        finally:
            try:
                self._jobs.put_nowait(None)
            except queue.Full:
                pass
            self._thread.join(timeout=1)


@dataclass(frozen=True)
class PredictionPlan:
    prediction: Prediction
    origin_step: int
    start_step: int
    end_step: int

    def action_at(self, global_step: int) -> np.ndarray | None:
        index = global_step-self.origin_step
        return self.prediction.actions[index] if 0 <= index < len(self.prediction.actions) else None


@dataclass(frozen=True)
class ScheduledAction:
    global_step: int
    execute_time_ns: int
    action: np.ndarray
    prediction_id: int
    local_index: int


@dataclass(frozen=True)
class ExecutionEvent:
    global_step: int
    expected_ns: int
    actual_ns: int
    status: str
    prediction_id: int | None = None


class CommandSink(Protocol):
    def send_target(self, joints_rad: np.ndarray, gripper_width_m: float) -> None: ...


class NullCommandSink:
    """Explicit hardware placeholder, never calls any robot control SDK."""
    def __init__(self) -> None:
        self.targets: deque[tuple[np.ndarray, float]] = deque(maxlen=100)

    def send_target(self, joints_rad: np.ndarray, gripper_width_m: float) -> None:
        self.targets.append((frozen_array(joints_rad), gripper_width_m))


class Executor:
    def __init__(self, grid: TimeGrid, config: RuntimeConfig, sink: CommandSink) -> None:
        self.grid, self.config, self.sink = grid, config, sink
        self._pending: dict[int, ScheduledAction] = {}
        self._history: OrderedDict[int, ScheduledAction] = OrderedDict()
        self.events: deque[ExecutionEvent] = deque(maxlen=256)
        self._lock = threading.RLock()
        self._last_processed: int | None = None
        self._in_flight: ScheduledAction | None = None
        self._closed = False
        self.last_error: Exception | None = None
        self.sent_count = self.skipped_count = 0

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending) + int(self._in_flight is not None)

    def commit(self, plan: PredictionPlan, now_ns: int) -> None:
        actions = checked_actions(plan.prediction.actions, self.config.horizon)
        if plan.end_step-plan.start_step != self.config.execution_steps:
            raise ValueError("A commit must contain exactly execution_steps")
        scheduled = []
        for step in range(plan.start_step, plan.end_step):
            local = step-plan.origin_step
            if not 0 <= local < len(actions):
                raise ValueError("Commit exceeds prediction horizon")
            scheduled.append(ScheduledAction(step, self.grid.at(step), actions[local], plan.prediction.request_id, local))
        with self._lock:
            if self._closed or self.last_error is not None:
                raise RuntimeError("Executor is not accepting plans")
            if self.grid.at(plan.start_step) <= now_ns:
                raise ValueError("Cannot commit an action at or before the current time")
            if self._last_processed is not None and plan.start_step <= self._last_processed:
                raise ValueError("Cannot reuse processed clock slots")
            if any(item.global_step in self._pending for item in scheduled):
                raise ValueError("Overlapping action plans")
            if len(self._pending)+len(scheduled)+int(self._in_flight is not None) > 2*self.config.horizon:
                raise OverflowError("Future action buffer capacity exceeded")
            self._pending.update({item.global_step: item for item in scheduled})

    def action_at(self, step: int) -> np.ndarray | None:
        with self._lock:
            item = self._pending.get(step) or self._history.get(step)
            if self._in_flight is not None and self._in_flight.global_step == step:
                item = self._in_flight
            return item.action if item is not None else None

    def execute_due(self, now_ns: int) -> int | None:
        step = self.grid.step_at_or_before(now_ns)
        with self._lock:
            if self._closed or self.last_error is not None:
                return None
            if self._last_processed is not None and step <= self._last_processed:
                return None
            self._last_processed = step
            for old in sorted(k for k in self._pending if k < step):
                item = self._pending.pop(old)
                self.skipped_count += 1
                self.events.append(ExecutionEvent(old, item.execute_time_ns, now_ns, "expired", item.prediction_id))
            item = self._pending.pop(step, None)
            if item is None:
                return None
            self._in_flight = item
        try:
            width_m = self.config.open_width_m*(1-float(np.clip(item.action[6], 0, 1)))
            self.sink.send_target(item.action[:6], width_m)
            with self._lock:
                self.sent_count += 1
                self._history[step] = item
                while len(self._history) > self.config.horizon:
                    self._history.popitem(last=False)
                self.events.append(ExecutionEvent(step, item.execute_time_ns, now_ns, "sent", item.prediction_id))
            return step
        except Exception as error:
            with self._lock:
                self.last_error = error
                self._pending.clear()
                self.events.append(ExecutionEvent(step, item.execute_time_ns, now_ns, "error", item.prediction_id))
            return None
        finally:
            with self._lock:
                self._in_flight = None

    def run(self, stop: threading.Event, *, clock_ns: Callable[[], int] = time.monotonic_ns) -> None:
        while not stop.is_set() and not self._closed:
            now = clock_ns()
            self.execute_due(now)
            next_ns = self.grid.at(self.grid.step_at_or_before(now)+1)
            stop.wait(max(0, (next_ns-clock_ns())/SECOND_NS))

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._pending.clear()


def bootstrap_reference(observation: Observation, config: RuntimeConfig) -> tuple[np.ndarray, np.ndarray]:
    actions = np.repeat(observation.state[None, :], config.horizon, axis=0)
    weights = np.zeros_like(actions)
    weights[:config.hold_steps] = 1
    return frozen_array(actions), frozen_array(weights)


def rtc_reference(origin_step: int, switch_step: int, previous: PredictionPlan,
                  executor: Executor, config: RuntimeConfig) -> tuple[np.ndarray, np.ndarray]:
    """Hard prefix uses committed commands; soft tail uses the previous full plan."""
    hard_length = switch_step-origin_step
    if not 0 <= hard_length < config.horizon:
        raise ValueError("Invalid hard prefix length")
    overlap_end = min(config.horizon, previous.origin_step+config.horizon-origin_step)
    if overlap_end < hard_length:
        raise ValueError("Previous plan does not cover committed prefix")
    actions = np.zeros((config.horizon, 7), dtype=np.float32)
    weights = np.zeros_like(actions)
    for index in range(config.horizon):
        step = origin_step+index
        if index < hard_length:
            action = executor.action_at(step)
            if action is None:
                raise ValueError(f"Missing committed reference at global step {step}")
            actions[index], weights[index] = action, 1
        elif index < overlap_end:
            action = previous.action_at(step)
            if action is None:
                raise ValueError("Missing soft-tail action")
            actions[index] = action
            fraction = (overlap_end-index)/(overlap_end-hard_length+1)
            weights[index] = fraction*np.expm1(fraction)/(math.e-1)
    return frozen_array(actions), frozen_array(weights)
