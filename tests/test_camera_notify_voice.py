"""Camera-init failure popups speak researcher voice, and name the cause.

A camera-not-initialized popup shipped the raw
`ValueError: <kind> registry has no real drivers and no null fallback.`
text -- exception class name, registry / null-fallback internals, and a
doubled period (the exception message already ends in '.') -- straight to
the user. The exception detail belongs to the log, behind the typed
outcome as its cause; the body the person reads must not duplicate it.

That ValueError was also the wrong cause: it is what the registry raised
when every camera driver found nothing, so an unplugged camera read "Could
not connect to the camera". The registry now raises the cause it found,
and a Classic's missing firmware file or unloadable driver says so.

Exercises the production classification of a camera's connect failure and
the words its typed outcome carries.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from drivers.exceptions import CameraFirmwareMissingError
from drivers.registry import DriverFallback, NoLiveDriverError
from modules.exceptions import CameraNotAvailableError
from modules.lumascope_api._lumascope import _camera_failure_cause


def _body(exc) -> str:
    return str(CameraNotAvailableError(_camera_failure_cause(exc)))


class TestCameraNotifyVoice:
    def test_no_camera_found_is_not_detected_and_its_body_is_clean(self):
        exc = NoLiveDriverError(
            'camera', DriverFallback('not_detected', ('PylonCamera', 'IDSCamera'))
        )
        assert _camera_failure_cause(exc) == 'camera_not_detected'
        body = _body(exc)
        assert body  # non-empty, actionable
        assert 'NoLiveDriverError' not in body
        assert 'registry' not in body
        assert '..' not in body  # no doubled period

    def test_no_camera_found_with_a_driver_that_could_not_load_names_the_driver(self):
        exc = NoLiveDriverError(
            'camera',
            DriverFallback(
                'not_detected',
                ('PylonCamera',),
                declined=('fx2: bundled libusb not in use: no library file',),
            ),
        )
        assert _camera_failure_cause(exc) == 'camera_driver_unavailable'
        assert 'libusb' not in _body(exc)

    def test_a_held_camera_port_is_port_in_use(self):
        exc = NoLiveDriverError('camera', DriverFallback('port_in_use', ('PylonCamera',)))
        assert _camera_failure_cause(exc) == 'camera_port_in_use'

    def test_a_missing_firmware_file_is_its_own_cause(self):
        exc = CameraFirmwareMissingError(['/opt/lvp/firmware/LumascopeClassic.hex'])
        assert _camera_failure_cause(exc) == 'camera_firmware_missing'
        assert '/opt/lvp' not in _body(exc)

    def test_camera_in_use_body_is_clean(self):
        # pypylon RuntimeException is matched by type-name string.
        exc = type('RuntimeException', (Exception,), {})('device busy 0xdead')
        body = _body(exc)
        assert 'RuntimeException' not in body
        assert '0xdead' not in body

    def test_permission_error_body_is_clean(self):
        body = _body(PermissionError('COM7 access denied by winerror 5'))
        assert 'winerror' not in body
        assert 'COM7 access denied' not in body

    def test_file_not_found_body_is_clean(self):
        body = _body(FileNotFoundError('/dev/video0 missing'))
        assert '/dev/video0' not in body
