# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A serial board's port search says what the OS listed, found or not.

A board missing at bring-up is reported as not detected. Without what the
OS did list, that cannot be told apart from a board that enumerated under
another VID:PID, as another device, or with no serial ports at all. The
search's log line names every listed port, and a found board names its
description and serial number, so a support bundle says which one it was.
"""

import logging

from serial.tools.list_ports_common import ListPortInfo

import drivers.serialboard as serialboard
from drivers.serialboard import SerialBoard
from tests.log_capture import records_of


def _port(device, vid, pid, description, serial_number):
    info = ListPortInfo(device)
    info.vid, info.pid = vid, pid
    info.description = description
    info.serial_number = serial_number
    return info


class _Listed:
    """The ports the OS lists; the subject is the search, so nothing is opened."""

    def __init__(self, *ports):
        self._ports = list(ports)

    def comports(self):
        return list(self._ports)


def _search(caplog, *ports):
    caplog.set_level(logging.INFO)
    board = SerialBoard(vid=0x0424, pid=0x704C, label='[LED Class ]', backend=_Listed(*ports))
    lines = [r.getMessage() for r in records_of(caplog, serialboard) if r.levelno == logging.INFO]
    return board, lines


def test_a_board_not_found_names_every_port_the_os_listed(caplog):
    board, lines = _search(
        caplog,
        _port('/dev/cu.usbmodem1', 0x2E8A, 0x0005, 'Board CDC', 'E6614'),
        _port('/dev/cu.Bluetooth', None, None, 'n/a', None),
    )

    assert not board.found
    assert lines == [
        '[LED Class ] No device with VID:PID 0424:704C; the OS listed 2 serial port(s): '
        "/dev/cu.usbmodem1 (VID:PID 2E8A:0005, 'Board CDC', SN E6614); "
        "/dev/cu.Bluetooth (VID:PID -, 'n/a', SN None)"
    ]


def test_no_ports_at_all_says_none(caplog):
    _board, lines = _search(caplog)

    assert lines == [
        '[LED Class ] No device with VID:PID 0424:704C; the OS listed 0 serial port(s): none'
    ]


def test_a_found_board_names_its_description_and_serial_number(caplog):
    board, lines = _search(
        caplog, _port('/dev/cu.usbmodem2', 0x0424, 0x704C, 'LED Controller', 'SN-0042')
    )

    assert board.found and board.port == '/dev/cu.usbmodem2'
    assert lines == [
        "[LED Class ] Found device at /dev/cu.usbmodem2 (VID:PID 0424:704C, 'LED Controller', SN SN-0042)"
    ]
