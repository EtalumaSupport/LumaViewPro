# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The emulated port: the real motor firmware in MicroPython, at the wire.

What is pinned here is what makes the port a serial link to a board rather
than a pipe to a process: the production driver connects through it with no
change, Ctrl-C and Ctrl-D do what they do on the board, and the process can
neither outlive its port nor fail quietly.
"""

import gc
import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest
import serial

from drivers.motorboard import MotorBoard
from drivers.sim_wire import backend as sim_backend
from drivers.sim_wire import port as sim_port
from drivers.sim_wire.backend import MotorBoardSpec, SimWireBackend

if not (sys.platform == 'darwin' or sys.platform.startswith('linux')):
    pytest.skip(
        'the firmware-backed simulator runs on macOS and Linux only', allow_module_level=True
    )

ALL_AXES = frozenset('XYZT')
MANIFEST = (sim_backend._PACKAGE / 'firmware' / 'MANIFEST').read_text()


def _open(dialect='3.0', timeout=0.5):
    backend = SimWireBackend(MotorBoardSpec('LS850T', ALL_AXES, dialect=dialect))
    port = backend.open(
        port=backend.comports()[0].device, baudrate=115200, timeout=timeout, write_timeout=1
    )
    _wait_for(lambda: b'Firmware:' in _exchange(port, b'INFO'), what='the board to answer INFO')
    _drain(port)
    return port


def _drain(port):
    # Boot output can trail the first reply; a test starts on a quiet line.
    saved, port.timeout = port.timeout, 0.1
    while port.read(4096):
        pass
    port.timeout = saved


def _wait_for(predicate, what, limit=5.0):
    deadline = time.monotonic() + limit
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f'timed out waiting for {what}')
        time.sleep(0.01)


def _exchange(port, command: bytes) -> bytes:
    port.reset_input_buffer()
    port.write(command + b'\n')
    return port.readline()


def _at_repl(port) -> bool:
    return port._board._tail == sim_port.REPL_PROMPT


def _firmware_pid(port) -> int:
    return port._board._life.process.proc.pid


@pytest.mark.parametrize('dialect', sim_backend.DIALECTS)
def test_the_production_driver_connects_through_the_emulator(dialect):
    spec = MotorBoardSpec('LS850T', ALL_AXES, dialect=dialect)
    board = MotorBoard(backend=SimWireBackend(spec))
    try:
        assert board.is_connected() and board.is_responsive()
        date = MANIFEST.split(f'motor-{dialect}.mpy')[1].split('FirmwareDate ')[1].split()[0]
        assert board.firmware_date == date
        assert sorted(board.detect_present_axes()) == sorted(ALL_AXES)
        assert board.motorconfig.model() == 'LS850T'
        # The unit config main ships; its Z agrees with LumaViewPro's defaults.
        assert board.motorconfig.usteps_per_mm('Z') == 170666
    finally:
        board.disconnect()


def test_the_axes_the_board_reports_are_the_axes_it_was_given():
    board = MotorBoard(backend=SimWireBackend(MotorBoardSpec('LS820', frozenset('Z'))))
    try:
        assert board.detect_present_axes() == ['Z']
    finally:
        board.disconnect()


# The runtime, not the firmware: prints, runs a long loop in C (where the VM
# never checks for an interrupt), then waits for input with nothing between.
_PENDING_CTRL_C = "import sys\nprint('ready')\nx = sum(range(20000000))\nsys.stdin.readline()\n"


@pytest.mark.parametrize('dialect', sim_backend.DIALECTS)
def test_a_ctrl_c_that_lands_before_the_read_fires_when_the_read_begins(dialect):
    # After each reply the firmware collects garbage and then blocks reading
    # the next command. A Ctrl-C landing in that gap is only scheduled, and
    # the runtime used to sit in the read with it pending until the next input
    # byte; on the board it fires at once. The loop in C makes the gap wide
    # enough that the signal always lands in it.
    proc = subprocess.Popen(
        [str(sim_backend.runtime_path(dialect)), '-c', _PENDING_CTRL_C],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        assert proc.stdout.readline() == b'ready\n'
        proc.send_signal(signal.SIGINT)
        # stdin stays open: the interrupt must fire with no input arriving.
        proc.wait(timeout=5)
        assert b'KeyboardInterrupt' in proc.stdout.read()
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        proc.stdin.close()
        proc.stdout.close()


@pytest.mark.parametrize('dialect', sim_backend.DIALECTS)
def test_ctrl_c_interrupts_the_running_firmware_into_the_repl(dialect):
    port = _open(dialect)
    try:
        _exchange(port, b'BOGUS')
        port.write(b'\x03')
        _wait_for(lambda: _at_repl(port), what='the REPL after Ctrl-C', limit=2.0)
        assert b'KeyboardInterrupt' in bytes(port._rx)
    finally:
        port.close()


def test_ctrl_d_is_a_soft_reset_at_the_repl_and_data_while_running():
    port = _open()
    try:
        port.write(b'\x04')
        assert b'not found' in _exchange(port, b'BOGUS')
        first = _firmware_pid(port)

        port.write(b'\x03')
        _wait_for(lambda: _at_repl(port), what='the REPL after Ctrl-C')
        # A driver reads the prompt before it sends Ctrl-D; the REPL is still
        # where the board is.
        _drain(port)
        port.write(b'\x04')
        assert _firmware_pid(port) != first
        _wait_for(
            lambda: b'Firmware:' in _exchange(port, b'INFO'), what='the firmware to boot again'
        )
    finally:
        port.close()


def test_ctrl_c_then_ctrl_d_in_one_write_is_a_soft_reset():
    # On the board the interrupt lands before the Ctrl-D does, so the Ctrl-D
    # meets the REPL; over a pipe a Ctrl-D ahead of the prompt would be end of
    # input and end the process.
    port = _open()
    try:
        first = _firmware_pid(port)
        port.write(b'\x03\x04')
        assert _firmware_pid(port) != first
        _wait_for(
            lambda: b'Firmware:' in _exchange(port, b'INFO'), what='the firmware to boot again'
        )
    finally:
        port.close()


def test_a_reopened_port_meets_the_same_running_homed_firmware():
    # The motor controller is powered by the board, not by the host cable: a
    # driver that closes its port and opens it again finds the firmware it
    # left, homing included.
    port = _open()
    board = port._board
    pid = _firmware_pid(port)
    assert _exchange(port, b'ZHOME').strip() == b'Z home successful'
    port.close()
    port = sim_port.EmulatedPort(board, port=sim_backend.MOTOR_DEVICE, timeout=0.5)
    try:
        assert _firmware_pid(port) == pid
        assert b'Z homed: True' in _exchange(port, b'FULLINFO')
    finally:
        port.close()


def test_the_board_process_ends_when_its_backend_is_released():
    backend = SimWireBackend(MotorBoardSpec('LS850T', ALL_AXES))
    port = backend.open(port=backend.comports()[0].device, baudrate=115200, timeout=0.5)
    pid = _firmware_pid(port)
    port.close()
    # Nothing holds the board now: not the port, not the backend.
    del port, backend
    gc.collect()
    _wait_for(lambda: not _alive(pid), what='the released board to end', limit=5.0)


def test_the_board_process_dies_with_the_interpreter_that_opened_it():
    # At stdin EOF the firmware's idle loop spins a whole core forever; a
    # host that dies without closing the port must not leave that behind.
    script = textwrap.dedent(
        """
        import sys, time
        from drivers.sim_wire.backend import MotorBoardSpec, SimWireBackend
        b = SimWireBackend(MotorBoardSpec('LS850T', frozenset('XYZT')))
        p = b.open(port=b.comports()[0].device, baudrate=115200, timeout=0.5)
        while True:  # the firmware is running, not still starting
            p.write(b'INFO\\n')
            if b'Firmware:' in p.readline():
                break
        print(b.motor_board._life.process.proc.pid, flush=True)
        time.sleep(60)
        """
    )
    host = subprocess.Popen(
        [sys.executable, '-c', script],
        cwd=sim_backend._REPO,
        env=dict(os.environ, PYTHONPATH=str(sim_backend._REPO)),
        stdout=subprocess.PIPE,
        text=True,
    )
    board_pid = int(host.stdout.readline())
    os.kill(host.pid, signal.SIGKILL)
    host.wait()
    _wait_for(lambda: not _alive(board_pid), what='the orphaned board to die', limit=5.0)


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_a_board_that_goes_away_raises_as_a_pulled_cable_does():
    port = _open()
    try:
        port._board._life.process.proc.kill()
        _wait_for(lambda: port._failure is not None, what='the reader to see the exit')
        port.reset_input_buffer()
        with pytest.raises(serial.SerialException, match='board process exited'):
            port.read(1)
        with pytest.raises(serial.SerialException, match='board process exited'):
            port.write(b'INFO\n')
    finally:
        port.close()


def test_output_nobody_reads_overflows_loudly(monkeypatch):
    monkeypatch.setattr(sim_port, 'RX_LIMIT_BYTES', 2048)
    port = _open()
    try:
        for _ in range(60):
            try:
                port.write(b'FULLINFO\n')
            except serial.SerialException:
                break
            time.sleep(0.005)
        _wait_for(lambda: port._failure is not None, what='the overflow')
        assert 'nothing is reading it' in port._failure
    finally:
        port.close()


def test_a_missing_runtime_is_refused_by_name(monkeypatch, tmp_path):
    monkeypatch.setattr(sim_backend, '_PACKAGE', tmp_path)
    with pytest.raises(serial.SerialException, match=r'build_sim_runtime\.sh'):
        sim_backend.runtime_path('3.0')


def test_the_repo_carries_compiled_firmware_and_no_firmware_source():
    firmware = sim_backend._PACKAGE / 'firmware'
    assert not list(firmware.rglob('*.py'))
    for mpy in firmware.glob('*.mpy'):
        data = mpy.read_bytes()
        assert data[:1] == b'M', f'{mpy.name} is not MicroPython bytecode'
        assert b'/Users/' not in data and b'/home/' not in data, f'{mpy.name} embeds a local path'
