"""The tests the serial-port guard test runs in a subprocess.

Not collected by the suite (the name is not test_*.py); tests/guards/
test_serial_ports_are_refused.py names it on a pytest command line.
"""

import serial
import serial.tools.list_ports as list_ports


def test_the_diagnostic_path_swallows_the_refusal_into_null_boards():
    """Passes on its own terms: create_diagnostic hands back null boards when
    the real constructors find no port. The guard must fail it anyway."""
    from modules.lumascope_api import Lumascope

    instance = Lumascope.create_diagnostic()
    try:
        assert instance._motion_driver is not None
    finally:
        instance.disconnect()


def test_a_bare_serialboard_given_a_port_touches_nothing():
    from drivers.serialboard import SerialBoard

    board = SerialBoard(vid=0, pid=0, label='TEST', port='test-port')
    assert board.found is True


def test_pyserial_is_the_real_one():
    assert serial.Serial.__qualname__ == 'Serial'
    assert list_ports.comports.__module__.startswith('serial')
