# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The USB link to the firmware-backed motor board, and how it fails.

The EL-0940's motor controller is an internal USB device behind the
mainboard's own hub, powered by the board. So a pulled host cable leaves
its firmware running and homed, while a reboot loses everything the
firmware knew; either way the open port fails as a USB link does. What is
pinned: each of those, and each way a single reply can go wrong on the
wire (dropped, late, garbled), first at the port and then through the
production driver.
"""

import os
import re
import sys
import threading
import time

import psutil
import pytest
import serial

from drivers.motorboard import MotorBoard
from drivers.sim_wire.backend import MOTOR_DEVICE, MotorBoardSpec, SimWireBackend
from drivers.sim_wire.port import EmulatedPort

if not (sys.platform == 'darwin' or sys.platform.startswith('linux')):
    pytest.skip(
        'the firmware-backed simulator runs on macOS and Linux only', allow_module_level=True
    )


def _open(backend: SimWireBackend, timeout: float = 0.5) -> EmulatedPort:
    port = backend.open(port=MOTOR_DEVICE, baudrate=115200, timeout=timeout)
    deadline = time.monotonic() + 5
    while b'Firmware:' not in _exchange(port, b'INFO'):
        assert time.monotonic() < deadline, 'the board did not answer INFO'
    # INFO is several lines, and boot output can trail it: start on a quiet line.
    saved, port.timeout = port.timeout, 0.1
    while port.read(4096):
        pass
    port.timeout = saved
    return port


def _exchange(port: EmulatedPort, command: bytes) -> bytes:
    port.reset_input_buffer()
    port.write(command + b'\n')
    return port.readline().strip()


def _z_homed(port: EmulatedPort) -> str:
    return re.search(rb'Z homed: (\w+)', _exchange(port, b'FULLINFO')).group(1).decode()


@pytest.fixture
def backend():
    return SimWireBackend(MotorBoardSpec('LS850T', frozenset('XYZT')))


class TestTheLinkAtThePort:
    def test_an_unplugged_board_fails_the_port_and_vanishes_while_its_firmware_runs_on(
        self, backend
    ):
        port = _open(backend)
        assert _exchange(port, b'ZHOME') == b'Z home successful'
        backend.motor_board.unplug()
        with pytest.raises(serial.SerialException, match='unplugged'):
            port.write(b'INFO\n')
        port.close()
        assert backend.comports() == []
        with pytest.raises(serial.SerialException, match='unplugged'):
            backend.open(port=MOTOR_DEVICE, timeout=0.5)

        backend.motor_board.replug()
        assert [p.device for p in backend.comports()] == [MOTOR_DEVICE]
        port = _open(backend)
        try:
            assert _z_homed(port) == 'True'
        finally:
            port.close()

    def test_a_rebooted_board_fails_the_port_and_comes_back_knowing_nothing(self, backend):
        port = _open(backend)
        assert _exchange(port, b'ZHOME') == b'Z home successful'
        backend.motor_board.reboot()
        with pytest.raises(serial.SerialException, match='rebooted'):
            port.read(1)
        port.close()
        port = _open(backend)
        try:
            assert _z_homed(port) == 'False'
        finally:
            port.close()

    def test_a_dropped_reply_never_arrives_and_the_next_one_does(self, backend):
        port = _open(backend)
        try:
            backend.motor_board.drop_next_reply()
            assert _exchange(port, b'ZHOME') == b''
            assert _exchange(port, b'ZHOME') == b'Z home successful'
        finally:
            port.close()

    def test_a_late_reply_arrives_after_its_delay_and_not_before(self, backend):
        port = _open(backend, timeout=0.2)
        try:
            backend.motor_board.delay_next_reply(0.6)
            started = time.monotonic()
            assert _exchange(port, b'ZHOME') == b''
            port.timeout = 2.0
            assert port.readline().strip() == b'Z home successful'
            assert time.monotonic() - started >= 0.6
        finally:
            port.close()

    def test_a_garbled_reply_is_one_line_of_the_same_length_saying_something_else(self, backend):
        port = _open(backend)
        try:
            backend.motor_board.garble_next_reply()
            garbled = _exchange(port, b'ZHOME')
            assert len(garbled) == len(b'Z home successful')
            assert garbled != b'Z home successful'
            assert _exchange(port, b'ZHOME') == b'Z home successful'
        finally:
            port.close()

    def test_a_reboot_racing_a_reconnect_leaves_one_firmware_process(self, backend):
        # A driver reconnects the moment a reboot drops its port; the reboot
        # must not start a second firmware beside the one the reconnect found.
        _open(backend).close()
        stop = threading.Event()

        def reconnect_forever():
            while not stop.is_set():
                try:
                    backend.open(port=MOTOR_DEVICE, timeout=0.1).close()
                except serial.SerialException:
                    pass

        racer = threading.Thread(target=reconnect_forever)
        racer.start()
        try:
            for _ in range(20):
                backend.motor_board.reboot()
        finally:
            stop.set()
            racer.join()
        # By name: the launch's watcher subshell carries the runtime's path in
        # its arguments and shares the board's directory, but it is `sh`.
        firmware = [
            child
            for child in psutil.Process().children(recursive=True)
            if child.name().startswith('micropython')
            and child.cwd() == os.path.realpath(backend.motor_board._life.workdir)
        ]
        assert len(firmware) == 1

    def test_a_board_takes_one_connection_at_a_time(self, backend):
        # Its output goes to one port; a second would starve the first.
        port = _open(backend)
        try:
            with pytest.raises(serial.SerialException, match='already open'):
                backend.open(port=MOTOR_DEVICE, timeout=0.5)
        finally:
            port.close()
        _open(backend).close()

    def test_one_reply_fault_at_a_time(self, backend):
        backend.motor_board.drop_next_reply()
        with pytest.raises(ValueError, match='already armed'):
            backend.motor_board.garble_next_reply()
        with pytest.raises(ValueError, match='positive'):
            SimWireBackend(
                MotorBoardSpec('LS850T', frozenset('XYZT'))
            ).motor_board.delay_next_reply(0)


class TestTheLinkThroughTheDriver:
    def test_the_driver_reconnects_after_a_cable_pull_to_the_same_homed_firmware(self, backend):
        board = MotorBoard(backend=backend)
        try:
            assert board.home()
            backend.motor_board.unplug()
            # The driver meets the pulled cable on its next exchange and closes.
            assert not board.exchange_command('INFO')
            backend.motor_board.replug()
            # The next exchange reconnects: the same firmware, still homed.
            fullinfo = board.exchange_command('FULLINFO')
            assert re.search(r'Z homed: True', fullinfo)
        finally:
            board.disconnect()

    def test_a_dropped_reply_is_no_answer_to_the_driver(self, backend):
        board = MotorBoard(backend=backend)
        try:
            backend.motor_board.drop_next_reply()
            assert not board.exchange_command('ZHOME', timeout=0.5)
            assert board.exchange_command('ZHOME') == 'Z home successful'
        finally:
            board.disconnect()
