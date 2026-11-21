"""Serve Piper pi0.5 RTC in the OpenPI JAX/GPU environment.

Example (checkpoint root contains params/ and assets/):
  python -m inference.serve_piper_rtc --checkpoint /path/to/models --port 8000

Run this script with the OpenPI environment active and its editable package
available. No camera or CAN modules are imported by this server.
"""

from __future__ import annotations

import argparse
import logging
import math


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi05_piper_finetune")
    parser.add_argument("--checkpoint", required=True, help="Root containing params/ and assets/")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-steps", type=int, default=10,
                        help="Initial OpenPI step count; validate latency on the actual GPU")
    parser.add_argument("--max-guidance", type=float, default=5.0,
                        help="Initial conservative scale; not a validated optimum")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must lie in [1,65535]")
    if args.num_steps < 1:
        parser.error("--num-steps must be positive")
    if not math.isfinite(args.max_guidance) or args.max_guidance < 0:
        parser.error("--max-guidance must be finite and nonnegative")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Defer GPU imports so --help also works in the robot-only environment.
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from .piper_rtc_policy import create_rtc_policy

    policy = create_rtc_policy(
        args.config, args.checkpoint, num_steps=args.num_steps, max_guidance=args.max_guidance
    )
    logging.info("RTC ready: horizon=%s, steps=%s, stats=%s", policy.horizon, args.num_steps,
                 policy.metadata["norm_stats_sha256"])
    logging.info("Serving at ws://%s:%s", args.host, args.port)
    WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    main()
