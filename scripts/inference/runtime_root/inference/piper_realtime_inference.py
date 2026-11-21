"""Compatibility entry point for the new RTC harness.

Both --demo and --read-only use a null actuator. The old, non-RTC async
chunk skeleton and its movement hooks have been replaced by run_piper_rtc.
"""
from .run_piper_rtc import main


if __name__ == "__main__":
    main()
