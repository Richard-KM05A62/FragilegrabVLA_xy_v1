"""与锁定 OpenPI WebSocket policy server 兼容的轻量客户端。"""

from __future__ import annotations

import functools
import time
from typing import Any

import msgpack
import numpy as np


def _pack_array(value: Any) -> Any:
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"不支持的 NumPy dtype：{value.dtype}")
    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    return value


def _unpack_array(value: dict) -> Any:
    if b"__ndarray__" in value:
        return np.ndarray(buffer=value[b"data"], dtype=np.dtype(value[b"dtype"]), shape=value[b"shape"])
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


_packb = functools.partial(msgpack.packb, default=_pack_array)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


class Pi05WebsocketClient:
    """协议与 OpenPI commit 215abfb 的 openpi-client 保持一致。"""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        api_key: str | None = None,
        connect_attempts: int = 1,
        retry_interval_s: float = 5.0,
    ) -> None:
        if not host or not 0 < port < 65536:
            raise ValueError("policy host/port 无效")
        if connect_attempts <= 0:
            raise ValueError("connect_attempts 必须为正整数")
        try:
            import websockets.sync.client as websocket_client
        except ImportError as exc:
            raise RuntimeError("缺少 websockets；请安装 scripts/sim/requirements-mujoco.txt") from exc

        uri = host if host.startswith(("ws://", "wss://")) else f"ws://{host}:{port}"
        if host.startswith(("ws://", "wss://")) and host.rsplit(":", 1)[-1].isdigit() is False:
            uri = f"{host}:{port}"
        headers = {"Authorization": f"Api-Key {api_key}"} if api_key else None
        last_error: BaseException | None = None
        for attempt in range(connect_attempts):
            try:
                self._ws = websocket_client.connect(
                    uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    ping_interval=None,
                    open_timeout=None,
                )
                self.metadata = _unpackb(self._ws.recv())
                break
            except (ConnectionError, OSError) as exc:
                last_error = exc
                if attempt + 1 < connect_attempts:
                    time.sleep(retry_interval_s)
        else:
            raise RuntimeError(f"无法连接 policy server：{uri}") from last_error

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        self._ws.send(_packb(observation))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"OpenPI inference server 返回错误：\n{response}")
        value = _unpackb(response)
        if not isinstance(value, dict):
            raise RuntimeError("OpenPI inference response 顶层不是字典")
        return value

    def close(self) -> None:
        self._ws.close()

    def __enter__(self) -> "Pi05WebsocketClient":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
