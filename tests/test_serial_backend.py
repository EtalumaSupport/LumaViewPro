# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A board driver takes its port from a backend: discovery and open only.

The simulator stands in for a real port at this seam and nowhere else, so
the properties that matter are that every discovery and every open a board
makes goes through the backend it was given, and that the production backend
still reaches pyserial's module attributes at call time (the suite's
real-port refusal depends on that).
"""

import serial
from serial.tools.list_ports_common import ListPortInfo

from drivers import serial_backend
from drivers.ledboard import LEDBoard
from drivers.motorboard import MotorBoard
from drivers.serialboard import SerialBoard

VID, PID = 0x1234, 0x5678


def _port(device, vid=VID, pid=PID):
    info = ListPortInfo(device)
    info.vid, info.pid = vid, pid
    return info


class RecordingBackend:
    """Offers ports and opens them as pyserial loopbacks, recording each call."""

    def __init__(self, ports):
        self._ports = ports
        self.comports_calls = 0
        self.opened: list[str] = []

    def comports(self):
        self.comports_calls += 1
        return list(self._ports)

    def open(self, **kwargs):
        self.opened.append(kwargs['port'])
        return serial.serial_for_url(
            'loop://', timeout=kwargs['timeout'], write_timeout=kwargs['write_timeout']
        )


def test_discovery_matches_vid_pid_from_the_backend():
    backend = RecordingBackend([_port('sim-other', pid=0x1), _port('sim-board')])

    board = SerialBoard(VID, PID, '[Test]', backend=backend)

    assert backend.comports_calls == 1
    assert board.found is True
    assert board.port == 'sim-board'


def test_open_goes_through_the_backend_and_the_driver_uses_what_it_returns():
    backend = RecordingBackend([_port('sim-board')])
    board = SerialBoard(VID, PID, '[Test]', backend=backend)

    board._open_serial()

    assert backend.opened == ['sim-board']
    board.driver.write(b'PING\n')
    assert board.driver.readline() == b'PING\n'
    board.driver.close()


def test_a_failed_open_rescans_through_the_same_backend():
    backend = RecordingBackend([_port('sim-moved')])
    board = SerialBoard(VID, PID, '[Test]', port='sim-gone', backend=backend)
    first_open = backend.open

    def open_fails_once(**kwargs):
        if kwargs['port'] == 'sim-gone':
            backend.opened.append('sim-gone')
            raise serial.SerialException('gone')
        return first_open(**kwargs)

    backend.open = open_fails_once

    board._open_serial()

    assert backend.opened == ['sim-gone', 'sim-moved']
    assert backend.comports_calls == 1
    assert board.port == 'sim-moved'
    board.driver.close()


def test_led_and_motor_boards_pass_their_backend_down():
    for board_cls in (LEDBoard, MotorBoard):
        backend = RecordingBackend([])

        board = board_cls(backend=backend)

        assert board._backend is backend, board_cls.__name__
        assert backend.comports_calls >= 1, board_cls.__name__
        assert board.found is False, board_cls.__name__


def test_pyserial_backend_resolves_pyserial_at_call_time(monkeypatch):
    opened, enumerated = [], []

    def fake_serial(**kwargs):
        opened.append(kwargs['port'])
        return 'the-port'

    def fake_comports(include_links=False):
        enumerated.append(include_links)
        return []

    monkeypatch.setattr(serial, 'Serial', fake_serial)
    monkeypatch.setattr(serial_backend.list_ports, 'comports', fake_comports)

    assert serial_backend.PYSERIAL.open(port='/dev/x') == 'the-port'
    assert serial_backend.PYSERIAL.comports() == []
    assert opened == ['/dev/x']
    assert enumerated == [True]
