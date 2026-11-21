"""Linux RTC launcher using Linux camera/CAN adapters.

All scheduling and safety behavior comes from ``run_piper_rtc.py``. Plain demo
and read-only modes retain ``NullCommandSink``; explicit Linux execute flags
install the bounded Piper hardware sink.
"""

from __future__ import annotations

import os
import sys
import builtins

import numpy as np


def main() -> None:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("This launcher is only for Linux")
    from . import run_piper_rtc as common
    from .rtc_sources_linux import SourcesLinux

    execute_inverted = "--execute-inverted" in sys.argv
    execute_direct = "--execute-direct" in sys.argv
    if execute_inverted and execute_direct:
        raise RuntimeError("choose only one of --execute-inverted and --execute-direct")
    if execute_inverted or execute_direct:
        sys.argv.remove("--execute-inverted" if execute_inverted else "--execute-direct")
        if "--read-only" not in sys.argv:
            raise RuntimeError("Linux execution requires --read-only real-device mode")
        from .rtc_sink_linux import DirectPiperCommandSinkLinux, InvertedPiperCommandSinkLinux
        common.NullCommandSink = InvertedPiperCommandSinkLinux if execute_inverted else DirectPiperCommandSinkLinux
        base_predictor = common.RTCPredictor

        class LinuxDebugRTCPredictor(base_predictor):
            def poll(self):
                result = super().poll()
                if result is not None and result.prediction is not None:
                    raw = np.asarray(result.prediction.actions[:, 6], dtype=np.float64)
                    width_mm = 100.1 * (1.0 - np.clip(raw, 0.0, 1.0))
                    print(
                        "RTC夹爪模型原始输出 closedness="
                        f"{np.array2string(raw, precision=6, separator=',')} | "
                        "单次换算宽度_mm="
                        f"{np.array2string(width_mm, precision=4, separator=',')}",
                        flush=True,
                    )
                return result

        common.RTCPredictor = LinuxDebugRTCPredictor
        original_print = builtins.print

        def linux_execution_print(*values, **kwargs):
            rewritten = []
            for value in values:
                if isinstance(value, str):
                    value = value.replace(
                        "执行接口为占位：不会使能或下发任何机械臂/夹爪运动指令。",
                        "Linux 真机执行器已启用：关节和夹爪命令通过安全检查后会下发。",
                    )
                    value = value.replace("模拟下发", "真机下发")
                    value = value.replace("真机运动指令始终为 0。", "真机执行模式已结束。")
                rewritten.append(value)
            original_print(*rewritten, **kwargs)

        # Override only this imported module's global print lookup. Read-only
        # mode and the original cross-platform source remain unchanged.
        common.print = linux_execution_print
        print(
            f"Linux RTC 真机模式：六关节模型目标{'整体取反' if execute_inverted else '直接下发'}，"
            "执行器默认速度 15%；夹爪按宽度目标下发。",
            flush=True,
        )

    # The common launcher stores Sources as a module global. Replace it only in
    # this process so the original file remains untouched.
    common.Sources = SourcesLinux
    if "--read-only" in sys.argv:
        print(
            "Linux devices: "
            f"ORBBEC_SDK_ROOT={os.environ.get('ORBBEC_SDK_ROOT', '<use --orbbec-sdk-root>')}, "
            f"PIPER_CAN_INTERFACE={os.environ.get('PIPER_CAN_INTERFACE', 'socketcan')}, "
            f"PIPER_CAN_CHANNEL={os.environ.get('PIPER_CAN_CHANNEL', 'can0')}",
            flush=True,
        )
    common.main()


if __name__ == "__main__":
    main()
