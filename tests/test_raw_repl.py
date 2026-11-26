# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A raw-REPL read that failed raises, never answering an empty or a guessed result.

The contract (``raw_repl.list_files``, ``raw_repl.read_file``): a listing or
a read whose code raised on the board raises ``BoardReportedError`` with the
board's traceback; one the board did not answer raises ``BoardNoReplyError``;
reads that disagree with no two alike raise ``BoardReplyUnreadableError``.
The board answers code that raised with empty stdout and the traceback on
stderr: read as stdout alone, a listing that raised would be a board with no
files, and a file that could not be opened an empty file, which the firmware
updater would back up and restore as the config.
"""

import time
from types import SimpleNamespace

import pytest

from _lumascope.drivers import raw_repl
from _lumascope.drivers.exceptions import (
    BoardNoReplyError,
    BoardReplyUnreadableError,
    BoardReportedError,
)

# raw_repl's own name for time, with sleeps that return at once; patching
# time.sleep itself would replace it for every thread in the process.
_NO_WAIT_TIME = SimpleNamespace(monotonic=time.monotonic, sleep=lambda _s: None)

_TRACEBACK = b'Traceback (most recent call last):\r\nOSError: [Errno 5] EIO\r\n'


class _Board:
    """A serial port whose board gives each exec the next of *replies*.

    A stand-in by design: the wire simulator has no raw REPL, so it cannot
    produce a board whose filesystem raises, goes silent or garbles a read
    (the simulator gap).
    """

    def __init__(self, *replies):
        self.timeout = 1
        self._replies = list(replies)
        self._pending = b''

    def write(self, data):
        # Each exec ends with Ctrl+D; the board's answer to it is then pending.
        if data.endswith(raw_repl.CTRL_D) and self._replies:
            self._pending = self._replies.pop(0)
        return len(data)

    def read(self, size):
        reply, self._pending = self._pending, b''
        return reply


def _raised():
    return b'OK\x04' + _TRACEBACK + b'\x04'


def _printed(stdout):
    return b'OK' + stdout + b'\x04\x04'


def test_a_listing_that_raised_on_the_board_raises_with_its_traceback():
    with pytest.raises(BoardReportedError) as refused:
        raw_repl.list_files(_Board(_raised()))
    assert refused.value.reason == 'board_reported_error'
    assert 'OSError: [Errno 5] EIO' in str(refused.value)


def test_a_listing_the_board_did_not_answer_raises():
    with pytest.raises(BoardNoReplyError) as refused:
        raw_repl.list_files(_Board(b''))
    assert refused.value.reason == 'board_no_reply'


def test_a_file_that_could_not_be_opened_raises_rather_than_reading_as_empty():
    with pytest.raises(BoardReportedError) as refused:
        raw_repl.read_file(_Board(_raised()), 'motorconfig.json', verify=False)
    assert refused.value.reason == 'board_reported_error'
    assert 'motorconfig.json' in str(refused.value)


def test_three_reads_that_differ_raise_rather_than_keeping_one(monkeypatch):
    monkeypatch.setattr(raw_repl, 'time', _NO_WAIT_TIME)
    board = _Board(_printed(b'YQ=='), _printed(b'Yg=='), _printed(b'Yw=='))
    with pytest.raises(BoardReplyUnreadableError) as refused:
        raw_repl.read_file(board, 'motorconfig.json')
    assert refused.value.reason == 'board_reply_unreadable'
