"""The tests the serial-port guard test runs in a subprocess.

Not collected by the suite (the name is not test_*.py); tests/guards/
test_serial_ports_are_refused.py names it on a pytest command line.
"""

import serial
import serial.tools.list_ports as list_ports


def test_a_real_scope_swallows_the_refusal_into_null_boards(monkeypatch):
    """Passes on its own terms: a real scope's board registries hand back null
    boards when the real constructors find no port. The guard must fail it anyway."""
    from _lumascope.drivers.registry import DriverFallback, NoLiveDriverError
    from _lumascope.modules.lumascope_api import Lumascope, _lumascope
    from tests.settings_fixtures import installation_of_its_own

    def no_camera(name='auto', **kwargs):
        # A stand-in by design: the subject is the boards' serial ports, and
        # a real camera search would reach the host's camera SDKs.
        raise NoLiveDriverError('camera', DriverFallback('not_detected', ()))

    monkeypatch.setattr(_lumascope.camera_registry, 'create', no_camera)
    instance = Lumascope(source_path=installation_of_its_own(), warn_pre_release=False)
    try:
        assert instance._motion_driver is not None
    finally:
        instance.disconnect()


def test_a_bare_serialboard_given_a_port_touches_nothing():
    from _lumascope.drivers.serialboard import SerialBoard

    board = SerialBoard(vid=0, pid=0, label='TEST', port='test-port')
    assert board.found is True


def test_pyserial_is_the_real_one():
    assert serial.Serial.__qualname__ == 'Serial'
    assert list_ports.comports.__module__.startswith('serial')
