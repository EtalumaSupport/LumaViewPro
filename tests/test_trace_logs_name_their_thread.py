# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every trace log line names the thread that wrote it.

The instrument's commands are serialized onto lane workers, and which thread
sent a command is what separates a lane's own write from one made beside it.
camera.log always named its thread; serial.log (the motion and LED commands)
and api.log did not, so a support bundle could show which thread wrote the
camera and never which thread moved the stage or lit an LED.

conftest mocks lvp_logger wholesale (importing it opens the log files), so
each formatter's format string is read from its class and applied to a real
record.
"""

from __future__ import annotations

import ast
import logging
import threading

import pytest

from tests import ast_seams


def _format_string(class_name: str) -> str:
    cls = next(
        node
        for node in ast_seams.parse_module('lvp_logger.py').body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    for node in ast.walk(cls):
        if isinstance(node, ast.keyword) and node.arg == 'fmt':
            return ast.literal_eval(node.value)
    raise AssertionError(f'{class_name} sets no fmt')


@pytest.mark.parametrize('class_name', ['SerialFormatter', 'APIFormatter', 'CameraFormatter'])
def test_a_trace_line_names_its_thread(class_name):
    formatter = logging.Formatter(fmt=_format_string(class_name), datefmt='%H:%M:%S')
    lines = []

    def _write():
        record = logging.LogRecord('LVP.test', logging.INFO, __file__, 1, 'move_abs Z', None, None)
        lines.append(formatter.format(record))

    worker = threading.Thread(target=_write, name='IO_WORKER')
    worker.start()
    worker.join(5.0)

    assert lines and '[IO_WORKER] move_abs Z' in lines[0], lines
