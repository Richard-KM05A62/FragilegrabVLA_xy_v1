"""Piper RTC harness. Both modes use a null actuator: NEVER commands the robot."""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from pathlib import Path

from .rtc_controller import Controller
from .rtc_core import DataBuffer, Executor, NullCommandSink, RTCPredictor, RuntimeConfig, TimeGrid
from .rtc_sources import Sources, SyntheticBackend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--demo", action="store_true", help="Synthetic observations/policy, no devices or model")
    modes.add_argument("--read-only", action="store_true", help="Real observations/model, no enable or movement")
    parser.add_argument("--task", help="Required in read-only mode; use the trained task prompt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--execution-steps", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--latency-window", type=int, default=20)
    parser.add_argument("--duration-s", type=float, default=0, help="0 = run until Ctrl+C")
    parser.add_argument("--demo-latency-ms", type=float, default=60)
    parser.add_argument("--response-timeout-s", type=float, default=180, help="Includes first JAX compilation")
    parser.add_argument("--norm-stats", type=Path,
                        default=Path(__file__).resolve().parents[1] / "models/assets/local/piper_x_recordings_v000/norm_stats.json")
    parser.add_argument("--base-serial")
    parser.add_argument("--wrist-serial")
    parser.add_argument("--orbbec-sdk-root", type=Path)
    parser.add_argument("--summary", type=Path, help="Optional final diagnostic JSON (not a dataset)")
    args = parser.parse_args()
    if args.read_only and not (args.task or "").strip():
        parser.error("--read-only requires --task with the trained task prompt")
    if (not 1 <= args.port <= 65535 or (args.task is not None and not args.task.strip())
            or not all(math.isfinite(x) for x in (args.duration_s, args.demo_latency_ms, args.response_timeout_s))
            or min(args.duration_s, args.demo_latency_ms) < 0 or args.response_timeout_s <= 0):
        parser.error("Invalid task, port, duration or timeout")
    try:
        config = RuntimeConfig(horizon=args.horizon, execution_steps=args.execution_steps,
                               latency_window=args.latency_window)
    except ValueError as exc:
        parser.error(str(exc))
    if args.read_only and not args.norm_stats.is_file():
        parser.error(f"Normalization stats missing: {args.norm_stats}")
    print("执行接口为占位：不会使能或下发任何机械臂/夹爪运动指令。", flush=True)
    if args.demo:
        print("模拟测试：不连接相机、CAN 或真实模型。", flush=True)
        backend = SyntheticBackend(config, args.demo_latency_ms)
    else:
        from .rtc_transport import WebsocketRTCBackend
        backend = WebsocketRTCBackend(args.host, args.port, config, args.norm_stats,
                                      response_timeout=args.response_timeout_s)
        print(f"RTC 服务与数据规范校验通过，stats={backend.metadata['norm_stats_sha256'][:12]}。", flush=True)
    stop = threading.Event()
    data = DataBuffer(config)
    grid = TimeGrid(time.monotonic_ns())
    sink = NullCommandSink()
    executor = Executor(grid, config, sink)
    predictor = RTCPredictor(backend, config.horizon)
    sources = Sources(data, stop)
    controller = Controller(data, predictor, executor, grid, config, args.task or "timing test",
                            on_event=lambda text: print(text.replace("\n", " ")[:220], flush=True))
    execution_thread = threading.Thread(target=executor.run, args=(stop,), name="RTCExecutor", daemon=True)
    start = time.monotonic()
    next_report = start + 5
    try:
        if args.demo:
            sources.start_demo(sink)
        else:
            sources.start_live(args.base_serial, args.wrist_serial, args.orbbec_sdk_root)
        execution_thread.start()
        print("等待三路输入，随后自动预热与运行；Ctrl+C 退出。", flush=True)
        while not stop.is_set() and controller.phase != "STOPPED":
            now = time.monotonic_ns()
            controller.tick(now)
            if args.duration_s and time.monotonic() - start >= args.duration_s:
                break
            if time.monotonic() >= next_report:
                latency = controller.latency.mean_age_ns
                latency_text = "测量中" if latency is None else f"{latency / 1e6:.1f} ms"
                if controller.phase == "WARMUP" and not predictor.busy:
                    names = ("外部相机", "腕部相机", "机械臂状态")
                    latest = [history.latest(now) for history in (data.base, data.wrist, data.states)]
                    missing = [name for name, sample in zip(names, latest)
                               if sample is None or now - sample.timestamp_ns > config.max_input_age_ns]
                    print(f"等待有效输入：{', '.join(missing) or '正在准备模型请求'}。", flush=True)
                else:
                    print(f"{controller.phase} | 平均延迟 {latency_text} | "
                          f"模拟下发 {executor.sent_count} | 队列 {executor.pending_count} | 超期 {controller.overruns}", flush=True)
                next_report = time.monotonic() + 5
            stop.wait(0.001)
    except KeyboardInterrupt:
        print("正在停止。", flush=True)
    finally:
        stop.set()
        controller.stop()
        predictor.close()
        if execution_thread.ident is not None:
            execution_thread.join(timeout=1)
        unfinished = sources.join()
        if unfinished:
            print(f"仍等待 SDK 退出：{', '.join(unfinished)}。", flush=True)
        diagnostic = {
            "mode": "synthetic_demo" if args.demo else "read_only_model",
            "motion_enabled": False, "elapsed_s": time.monotonic() - start,
            "completed_predictions": controller.completed_predictions,
            "simulated_commands": executor.sent_count, "expired_commands": executor.skipped_count,
            "overruns": controller.overruns, "recoveries": controller.recoveries,
            "mean_latency_ms": None if controller.latency.mean_age_ns is None else controller.latency.mean_age_ns / 1e6,
            "events": list(controller.events),
            "execution_events": [event.__dict__ for event in executor.events],
        }
        if args.summary:
            # Optional generated output, not source-file editing.
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已停止：模拟下发 {executor.sent_count}，跳过过期动作 {executor.skipped_count}，"
              f"超期 {controller.overruns}。真机运动指令始终为 0。", flush=True)
    failure = sources.failure()
    if failure is not None:
        source, error = failure
        raise RuntimeError(f"{source} 输入失败：{error}") from error
    if executor.last_error is not None:
        raise RuntimeError(f"Executor failed: {executor.last_error}") from executor.last_error


if __name__ == "__main__":
    main()
