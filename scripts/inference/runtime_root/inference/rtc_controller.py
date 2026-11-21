"""Nonblocking RTC state machine. Only Executor owns the command timer."""
from __future__ import annotations

from collections import deque
from typing import Callable, Protocol

from .rtc_core import (
    DataBuffer, Executor, InferenceRequest, InferenceResult, PredictionPlan,
    RollingLatency, RuntimeConfig, SECOND_NS, TimeGrid, bootstrap_reference, rtc_reference,
)


class Predictor(Protocol):
    @property
    def busy(self) -> bool: ...
    def submit(self, request: InferenceRequest) -> None: ...
    def poll(self) -> InferenceResult | None: ...


class Controller:
    def __init__(self, data: DataBuffer, predictor: Predictor, executor: Executor,
                 grid: TimeGrid, config: RuntimeConfig, prompt: str,
                 *, on_event: Callable[[str], None] | None = None) -> None:
        if not prompt.strip():
            raise ValueError("Task prompt cannot be empty")
        self.data, self.predictor, self.executor = data, predictor, executor
        self.grid, self.config, self.prompt = grid, config, prompt
        self.phase = "WARMUP"
        self.latency = RollingLatency(config.latency_window)
        self.plan: PredictionPlan | None = None
        self.next_origin: int | None = None
        self.next_switch: int | None = None
        self.active_request: int | None = None
        self.events: deque[str] = deque(maxlen=128)
        self.on_event = on_event
        self._next_id = 0
        self._initial_start = True
        self._retry_after_ns = 0
        self._last_drain_reason: str | None = None
        self.completed_predictions = self.overruns = self.recoveries = 0

    def _event(self, text: str) -> None:
        self.events.append(text)
        if self.on_event is not None:
            self.on_event(text)

    def _drain(self, reason: str) -> None:
        if self.phase != "DRAIN" and reason != self._last_drain_reason:
            self._event(f"回退同步启动：{reason}")
        self._last_drain_reason = reason
        self.phase = "DRAIN"
        self.active_request = None
        self.next_origin = self.next_switch = None

    def stop(self) -> None:
        self.phase = "STOPPED"
        self.active_request = None
        self.data.cancel_hold_check()
        self.executor.close()

    def tick(self, now_ns: int) -> None:
        if self.phase == "STOPPED":
            return
        if self.executor.last_error is not None:
            self._event(f"下发适配器失败：{self.executor.last_error}")
            self.stop()
            return
        result = self.predictor.poll()
        if result is not None:
            self._complete(result, now_ns)
        if self.phase == "STOPPED" or now_ns < self._retry_after_ns:
            return
        if self.phase == "DRAIN":
            finished = self.plan is None or now_ns >= self.grid.at(self.plan.end_step)
            if finished and not self.executor.pending_count and not self.predictor.busy:
                if self._initial_start or self.data.is_stationary(now_ns):
                    self.phase = "BOOTSTRAP"
                    self.recoveries += 1
            else:
                return
        if self.phase in ("WARMUP", "CALIBRATION", "BOOTSTRAP"):
            if not self.predictor.busy:
                if self.phase == "BOOTSTRAP" and not self._initial_start and not self.data.is_stationary(now_ns):
                    return
                observation = self.data.latest(now_ns)
                if observation is not None:
                    actions, weights = bootstrap_reference(observation, self.config)
                    self._submit(self.phase, observation, actions, weights)
        elif self.phase == "RTC":
            self._rtc_tick(now_ns)

    def _submit(self, mode, observation, actions, weights, *, origin=None, switch=None, local_start=0) -> None:
        self._next_id += 1
        request = InferenceRequest(self._next_id, mode, observation, self.prompt,
                                   actions, weights, origin, switch, local_start)
        if mode == "BOOTSTRAP":
            self.data.begin_hold_check(observation.state)
        try:
            self.predictor.submit(request)
        except Exception:
            if mode == "BOOTSTRAP":
                self.data.cancel_hold_check()
            raise
        self.active_request = request.request_id

    def _complete(self, result: InferenceResult, now_ns: int) -> None:
        request, prediction = result.request, result.prediction
        if prediction is not None and request.mode != "WARMUP":
            self.latency.add(prediction.received_ns-request.observation.reference_time_ns,
                             prediction.received_ns-prediction.request_sent_ns)
            self.completed_predictions += 1
        if request.request_id != self.active_request:
            return  # Completion of an invalidated/late request.
        self.active_request = None
        if result.error is not None or prediction is None:
            if request.mode == "BOOTSTRAP":
                self.data.cancel_hold_check()
            self._event(f"推理失败，稍后重试：{result.error}")
            self._retry_after_ns = now_ns+self.config.retry_delay_ns
            if request.mode == "RTC":
                self._drain("推理异常")
            return
        if request.mode == "WARMUP":
            self.phase = "CALIBRATION"
            self._event("模型预热完成，测量稳定延迟。")
            return
        if request.mode == "CALIBRATION":
            self.phase = "BOOTSTRAP"
            self._event("延迟测量完成，准备静止启动。")
            return
        if request.mode == "BOOTSTRAP":
            if not self.data.end_hold_check(now_ns):
                self._event("静止启动期间状态改变，重新获取观测。")
                self._retry_after_ns = now_ns+self.config.retry_delay_ns
                return
            target = prediction.actions[self.config.hold_steps, :6]
            if max(abs(target-request.observation.state[:6])) > self.config.bootstrap_max_joint_jump:
                self._event("首目标与当前姿态不连续，重新启动。")
                self._retry_after_ns = now_ns+self.config.retry_delay_ns
                return
            start = self.grid.first_at_or_after(now_ns+self.config.prepare_margin_ns)
            plan = PredictionPlan(prediction, start-self.config.hold_steps,
                                  start, start+self.config.execution_steps)
        else:
            assert request.origin_step is not None and request.switch_step is not None
            deadline = self.grid.at(request.switch_step)-self.config.prepare_margin_ns
            if max(now_ns, prediction.received_ns) >= deadline:
                self.overruns += 1
                self._drain("结果超期")
                return
            plan = PredictionPlan(prediction, request.origin_step, request.switch_step,
                                  request.switch_step+self.config.execution_steps)
        try:
            self.executor.commit(plan, now_ns)
        except (ValueError, RuntimeError, OverflowError) as error:
            self._drain(f"计划无法提交：{error}")
            return
        self.plan = plan
        self._initial_start = False
        self.phase = "RTC"
        if request.mode == "RTC":
            self._last_drain_reason = None
        self._schedule_next(now_ns)

    def _schedule_next(self, now_ns: int) -> None:
        assert self.plan is not None
        delay = self.latency.delay_steps(self.config.fps)
        compute = self.latency.mean_compute_ns
        if (delay is None or delay > self.config.execution_steps
                or delay+self.config.execution_steps > self.config.horizon
                or compute is None or compute*self.config.fps > self.config.execution_steps*SECOND_NS):
            self._drain("平均延迟不满足实时条件")
            return
        padding = (self.config.horizon-self.config.execution_steps-delay)//2
        if delay+padding > self.config.execution_steps:
            self._drain("观测前缀长于首段已提交动作，无法安全衔接")
            return
        self.next_switch = self.plan.end_step
        self.next_origin = self.next_switch-delay-padding
        if self.grid.at(self.next_origin) < now_ns:
            self._drain("下一触发时刻已过去")

    def _rtc_tick(self, now_ns: int) -> None:
        assert self.plan is not None and self.next_origin is not None and self.next_switch is not None
        target_ns = self.grid.at(self.next_origin)
        deadline = self.grid.at(self.next_switch)-self.config.prepare_margin_ns
        if now_ns >= deadline:
            self.overruns += 1
            self._drain("推理未能赶上切换时刻")
            return
        if self.predictor.busy:
            return
        if now_ns < target_ns:
            return
        if now_ns-target_ns >= SECOND_NS//self.config.fps:
            self._drain("错过观测触发时刻")
            return
        observation = self.data.at(target_ns, now_ns)
        if observation is None:
            return  # Bounded retry during this trigger slot; executor keeps running.
        try:
            actions, weights = rtc_reference(self.next_origin, self.next_switch,
                                             self.plan, self.executor, self.config)
        except ValueError as error:
            self._drain(str(error))
            return
        self._submit("RTC", observation, actions, weights, origin=self.next_origin,
                     switch=self.next_switch, local_start=self.next_switch-self.next_origin)
