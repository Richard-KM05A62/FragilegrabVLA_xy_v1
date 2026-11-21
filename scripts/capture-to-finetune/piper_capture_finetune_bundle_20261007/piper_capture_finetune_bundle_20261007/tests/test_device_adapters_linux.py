import tempfile
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

from capture.device_adapters_linux import (
    OrbbecDaBaiDC1AdapterLinux,
    PiperCanAdapterLinux,
    find_orbbec_library,
)


class LinuxDeviceAdapterTest(unittest.TestCase):
    def test_finds_common_sdk_layout_without_loading_library(self):
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "SDK/lib/libOrbbecSDK.so"
            library.parent.mkdir(parents=True)
            library.touch()
            self.assertEqual(find_orbbec_library(directory), library.resolve())

    def test_missing_sdk_root_is_actionable(self):
        with self.assertRaisesRegex(RuntimeError, "ORBBEC_SDK_ROOT"):
            find_orbbec_library(None)

    def test_orbbec_rejects_serial_missing_from_sdk_before_native_selection(self):
        class FakeOrbbec(OrbbecDaBaiDC1AdapterLinux):
            def __init__(self):
                super().__init__("usb-descriptor-only")
                self.calls = []

            def _load_sdk(self):
                self._lib = object()

            def _call(self, name, *args):
                self.calls.append(name)
                return {
                    "ob_create_context": 1,
                    "ob_query_device_list": 2,
                    "ob_device_list_device_count": 1,
                    "ob_device_list_get_device_serial_number": b"",
                    "ob_delete_device_list": None,
                }[name]

        adapter = FakeOrbbec()
        with self.assertRaisesRegex(RuntimeError, "not exposed by the Orbbec SDK"):
            adapter.start()
        self.assertNotIn("ob_device_list_get_device_by_serial_number", adapter.calls)
        self.assertNotIn("ob_create_pipeline_with_device", adapter.calls)
        self.assertIn("ob_delete_device_list", adapter.calls)

    def test_piper_can_settings_are_passed_to_vendor_factory(self):
        captured = {}
        robot = object()
        fake = types.ModuleType("pyAgxArm")

        def create_agx_arm_config(**kwargs):
            captured.update(kwargs)
            return kwargs

        class Factory:
            @staticmethod
            def create_arm(config):
                self.assertEqual(config, captured)
                return robot

        fake.create_agx_arm_config = create_agx_arm_config
        fake.AgxArmFactory = Factory
        with mock.patch.dict(sys.modules, {"pyAgxArm": fake}):
            adapter = PiperCanAdapterLinux(channel="can0", interface="socketcan")
        self.assertIs(adapter._robot, robot)
        self.assertEqual(
            captured,
            {"robot": "piper_x", "comm": "can", "channel": "can0", "interface": "socketcan"},
        )

    def test_piper_linux_defaults_use_socketcan(self):
        captured = {}
        fake = types.ModuleType("pyAgxArm")
        fake.create_agx_arm_config = lambda **kwargs: captured.update(kwargs) or kwargs
        fake.AgxArmFactory = types.SimpleNamespace(create_arm=lambda config: object())
        with mock.patch.dict(sys.modules, {"pyAgxArm": fake}), mock.patch.dict(
            "os.environ", {}, clear=True
        ):
            PiperCanAdapterLinux()
        self.assertEqual(captured["interface"], "socketcan")
        self.assertEqual(captured["channel"], "can0")


if __name__ == "__main__":
    unittest.main()
