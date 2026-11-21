"""Offline RTC transport contract and shutdown tests; no network or hardware."""
from __future__ import annotations

import hashlib
import pickle
import sys
import threading
import types
import unittest
from collections import deque
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from inference.rtc_core import RuntimeConfig
from inference.rtc_transport import WebsocketRTCBackend, stats_sha256, validate_metadata


STATS_HASH = "a" * 64


def valid_metadata() -> dict:
    return {"rtc_enabled": True, "horizon": 10, "action_dim": 7, "fps": 30,
            "action_space": "absolute_joint_position", "joint_unit": "rad",
            "gripper": "closedness", "norm_stats_sha256": STATS_HASH}


def encoded(value) -> bytes:
    # The fake codec exercises transport framing without installing msgpack.
    return pickle.dumps(value)


class FakeSocket:
    def __init__(self, *responses) -> None:
        self.responses = deque(responses)
        self.sent = []
        self.recv_timeouts = []
        self.close_count = 0
        self.closed = threading.Event()

    def send(self, message) -> None:
        if self.closed.is_set():
            raise ConnectionError("socket closed")
        self.sent.append(message)

    def recv(self, *, timeout):
        self.recv_timeouts.append(timeout)
        if self.closed.is_set():
            raise ConnectionError("socket closed")
        if not self.responses:
            raise AssertionError("Unexpected extra recv: the transport reused a stale socket")
        response = self.responses.popleft()
        if isinstance(response, BaseException):
            raise response
        return response(self) if callable(response) else response

    def close(self) -> None:
        self.close_count += 1
        self.closed.set()


class BlockingReceive:
    """A receive that only socket.close() or explicit test cleanup can release."""
    def __init__(self, value) -> None:
        self.value = value
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, socket: FakeSocket):
        self.entered.set()
        while not self.release.wait(0.005):
            if socket.closed.is_set():
                raise ConnectionError("recv interrupted by close")
        if socket.closed.is_set():
            raise ConnectionError("recv interrupted by close")
        return self.value


class FakeConnector:
    def __init__(self, *connections) -> None:
        self.connections = deque(connections)
        self.calls = []

    def __call__(self, uri, **options):
        self.calls.append((uri, options))
        if not self.connections:
            raise AssertionError("Unexpected extra connection attempt")
        connection = self.connections.popleft()
        if isinstance(connection, BaseException):
            raise connection
        return connection() if callable(connection) else connection


class MetadataTests(unittest.TestCase):
    def test_valid_metadata_accepted(self) -> None:
        validate_metadata(valid_metadata(), RuntimeConfig(), STATS_HASH)

    def test_h40_metadata_requires_h40_runtime_and_stats(self) -> None:
        h40_hash = "4" * 64
        metadata = valid_metadata()
        metadata["horizon"] = 40
        metadata["norm_stats_sha256"] = h40_hash
        config = RuntimeConfig(horizon=40, execution_steps=20)
        validate_metadata(metadata, config, h40_hash)
        with self.assertRaisesRegex(ValueError, "horizon"):
            validate_metadata(metadata, RuntimeConfig(), h40_hash)
        with self.assertRaisesRegex(ValueError, "norm_stats_sha256"):
            validate_metadata(metadata, config, STATS_HASH)

    def test_every_contract_field_must_match(self) -> None:
        wrong_values = {"rtc_enabled": False, "horizon": 50, "action_dim": 32,
                        "fps": 20, "action_space": "joint_delta", "joint_unit": "degree",
                        "gripper": "width_m", "norm_stats_sha256": "b" * 64}
        for key, value in wrong_values.items():
            with self.subTest(field=key):
                metadata = valid_metadata()
                metadata[key] = value
                with self.assertRaisesRegex(ValueError, key):
                    validate_metadata(metadata, RuntimeConfig(), STATS_HASH)

    def test_every_contract_field_is_required(self) -> None:
        for key in valid_metadata():
            with self.subTest(field=key):
                metadata = valid_metadata()
                del metadata[key]
                with self.assertRaisesRegex(ValueError, key):
                    validate_metadata(metadata, RuntimeConfig(), STATS_HASH)

    def test_extra_metadata_does_not_break_contract(self) -> None:
        metadata = valid_metadata()
        metadata["server_version"] = "test"
        validate_metadata(metadata, RuntimeConfig(), STATS_HASH)

    def test_hash_uses_exact_stats_file_bytes(self) -> None:
        content = b'{"q01": [0.0], "q99": [1.0]}\n'
        with patch.object(Path, "read_bytes", return_value=content) as read:
            self.assertEqual(stats_sha256(Path("unused-stats.json")),
                             hashlib.sha256(content).hexdigest())
            read.assert_called_once_with()


class TransportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        codec = types.ModuleType("openpi_client.msgpack_numpy")
        codec.packb = encoded
        codec.unpackb = pickle.loads
        package = types.ModuleType("openpi_client")
        package.msgpack_numpy = codec
        self.client_module = types.ModuleType("websockets.sync.client")
        self.client_module.connect = lambda *_args, **_kwargs: None
        self.stack.enter_context(patch.dict(sys.modules, {
            "openpi_client": package,
            "openpi_client.msgpack_numpy": codec,
            "websockets.sync.client": self.client_module,
        }))
        self.stack.enter_context(patch("inference.rtc_transport.stats_sha256", return_value=STATS_HASH))
        self.backends = []
        self.addCleanup(self._close_backends)

    def _close_backends(self) -> None:
        for backend in self.backends:
            backend.close()

    def backend(self, *connections, **options):
        connector = FakeConnector(*connections)
        self.client_module.connect = connector
        backend = WebsocketRTCBackend("example.invalid", 8000, RuntimeConfig(),
                                      Path("unused-stats.json"), **options)
        self.backends.append(backend)
        return backend, connector

    def start_infer(self, backend):
        outcome = {}

        def worker():
            try:
                outcome["result"] = backend.infer({"observation/state": [0] * 7})
            except BaseException as exc:
                outcome["error"] = exc

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return thread, outcome

    def test_successful_roundtrip_and_bounded_connection_options(self) -> None:
        socket = FakeSocket(encoded(valid_metadata()), encoded({"actions": [[0] * 7]}))
        backend, connector = self.backend(socket, connect_timeout=1.5, response_timeout=2.5)
        payload = {"prompt": "task", "rtc/reference_actions": [[0] * 7]}
        result = backend.infer(payload)
        self.assertEqual(result, {"actions": [[0] * 7]})
        self.assertEqual(pickle.loads(socket.sent[0]), payload)
        self.assertEqual(socket.recv_timeouts, [1.5, 2.5])
        self.assertEqual(connector.calls, [("ws://example.invalid:8000", {
            "open_timeout": 1.5, "close_timeout": 1, "compression": None,
            "max_size": 32 * 1024 * 1024})])
        self.assertEqual(backend.metadata, valid_metadata())

    def test_timeout_discards_socket_and_never_uses_stale_response(self) -> None:
        old = FakeSocket(encoded(valid_metadata()), TimeoutError("inference timed out"),
                         encoded({"stale": True}))
        fresh = FakeSocket(encoded(valid_metadata()), encoded({"fresh": True}))
        backend, connector = self.backend(old, fresh)
        with self.assertRaisesRegex(TimeoutError, "inference timed out"):
            backend.infer({"request": 1})
        self.assertTrue(old.closed.is_set())
        self.assertIsNone(backend._ws)
        self.assertEqual(backend.infer({"request": 2}), {"fresh": True})
        self.assertEqual(len(connector.calls), 2)
        self.assertEqual(len(old.sent), 1)
        self.assertEqual(len(old.responses), 1)  # Stale response remains unread.

    def test_one_reconnect_attempt_per_infer_not_an_infinite_retry(self) -> None:
        old = FakeSocket(encoded(valid_metadata()), TimeoutError("response timeout"))
        fresh = FakeSocket(encoded(valid_metadata()), encoded({"ok": True}))
        backend, connector = self.backend(old, TimeoutError("connect timeout"), fresh)
        with self.assertRaises(TimeoutError):
            backend.infer({})
        with self.assertRaisesRegex(TimeoutError, "connect timeout"):
            backend.infer({})
        self.assertEqual(len(connector.calls), 2)
        self.assertEqual(backend.infer({}), {"ok": True})
        self.assertEqual(len(connector.calls), 3)

    def test_failed_initial_connection_is_not_retried(self) -> None:
        connector = FakeConnector(TimeoutError("offline"))
        self.client_module.connect = connector
        with self.assertRaisesRegex(TimeoutError, "offline"):
            WebsocketRTCBackend("example.invalid", 8000, RuntimeConfig(), Path("unused"))
        self.assertEqual(len(connector.calls), 1)

    def test_bad_handshake_closes_socket(self) -> None:
        bad_contract = valid_metadata()
        bad_contract["joint_unit"] = "degree"
        for message in ("text instead of binary", encoded(["not a mapping"]), encoded(bad_contract)):
            with self.subTest(message=repr(message)[:80]):
                socket = FakeSocket(message)
                self.client_module.connect = FakeConnector(socket)
                with self.assertRaises(ValueError):
                    WebsocketRTCBackend("example.invalid", 8000, RuntimeConfig(), Path("unused"))
                self.assertTrue(socket.closed.is_set())

    def test_reconnect_checks_contract_again(self) -> None:
        old = FakeSocket(encoded(valid_metadata()), TimeoutError("timeout"))
        metadata = valid_metadata()
        metadata["norm_stats_sha256"] = "b" * 64
        changed = FakeSocket(encoded(metadata))
        backend, _ = self.backend(old, changed)
        with self.assertRaises(TimeoutError):
            backend.infer({})
        with self.assertRaisesRegex(ValueError, "norm_stats_sha256"):
            backend.infer({})
        self.assertTrue(changed.closed.is_set())
        self.assertIsNone(backend._ws)

    def test_model_error_and_nonmapping_response_drop_connection(self) -> None:
        for response, error_type in (("model error: bad observation", RuntimeError),
                                     (encoded([1, 2, 3]), ValueError),
                                     (b"invalid encoded data", pickle.UnpicklingError)):
            with self.subTest(response=repr(response)):
                socket = FakeSocket(encoded(valid_metadata()), response)
                backend, _ = self.backend(socket)
                with self.assertRaises(error_type):
                    backend.infer({})
                self.assertTrue(socket.closed.is_set())
                self.assertIsNone(backend._ws)

    def test_close_is_idempotent_and_prevents_later_reconnect(self) -> None:
        socket = FakeSocket(encoded(valid_metadata()))
        backend, connector = self.backend(socket)
        backend.close()
        backend.close()
        self.assertEqual(socket.close_count, 1)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            backend.infer({})
        self.assertEqual(len(connector.calls), 1)

    def test_close_interrupts_connected_inference_recv(self) -> None:
        blocked = BlockingReceive(encoded({"unexpected": True}))
        socket = FakeSocket(encoded(valid_metadata()), blocked)
        backend, _ = self.backend(socket)
        thread, outcome = self.start_infer(backend)
        try:
            self.assertTrue(blocked.entered.wait(1), "worker did not enter recv")
            backend.close()
            thread.join(1)
            self.assertFalse(thread.is_alive(), "close did not interrupt inference recv")
            self.assertIsInstance(outcome.get("error"), ConnectionError)
            self.assertNotIn("result", outcome)
            self.assertIsNone(backend._ws)
        finally:
            blocked.release.set()
            thread.join(1)

    def test_close_during_connect_does_not_publish_new_socket(self) -> None:
        old = FakeSocket(encoded(valid_metadata()), TimeoutError("timeout"))
        connection_entered = threading.Event()
        connection_release = threading.Event()
        fresh = FakeSocket(encoded(valid_metadata()), encoded({"must_not_send": True}))

        def connecting():
            connection_entered.set()
            if not connection_release.wait(2):
                raise TimeoutError("test connection barrier timed out")
            return fresh

        backend, _ = self.backend(old, connecting)
        with self.assertRaises(TimeoutError):
            backend.infer({})
        thread, outcome = self.start_infer(backend)
        try:
            self.assertTrue(connection_entered.wait(1))
            backend.close()
            connection_release.set()
            thread.join(1)
            self.assertFalse(thread.is_alive())
            self.assertIsInstance(outcome.get("error"), RuntimeError)
            self.assertTrue(fresh.closed.is_set())
            self.assertEqual(fresh.sent, [])
            self.assertIsNone(backend._ws)
        finally:
            connection_release.set()
            thread.join(1)

    def test_close_interrupts_reconnect_metadata_recv(self) -> None:
        old = FakeSocket(encoded(valid_metadata()), TimeoutError("timeout"))
        blocked = BlockingReceive(encoded(valid_metadata()))
        fresh = FakeSocket(blocked)
        backend, _ = self.backend(old, fresh)
        with self.assertRaises(TimeoutError):
            backend.infer({})
        thread, outcome = self.start_infer(backend)
        try:
            self.assertTrue(blocked.entered.wait(1), "worker did not enter metadata recv")
            backend.close()
            thread.join(0.5)
            interrupted = not thread.is_alive()
        finally:
            blocked.release.set()
            thread.join(1)
        self.assertTrue(interrupted, "close must also interrupt a connecting socket's metadata recv")
        self.assertNotIn("result", outcome)
        self.assertIsNone(backend._ws)
        self.assertTrue(fresh.closed.is_set())

    def test_invalid_timeouts_fail_before_dependency_import_or_connect(self) -> None:
        for options in ({"connect_timeout": 0}, {"response_timeout": -1}):
            with self.subTest(options=options):
                with patch.dict(sys.modules, {"openpi_client": None}):
                    with self.assertRaisesRegex(ValueError, "positive"):
                        WebsocketRTCBackend("example.invalid", 8000, RuntimeConfig(),
                                            Path("unused"), **options)


if __name__ == "__main__":
    unittest.main()
