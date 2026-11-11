# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""``--driver-log`` writes the drivers' own lines, at DEBUG, to its file, each once.

The IDS and FX2 bench procedures read that file. The drivers take their
loggers from src/_lumascope/lib/log_helpers by name, so the capture is on those named
loggers; a capture on anything the drivers do not write to keeps an empty
file through a whole bench run.

The capture is on ``LVP``, the main logger, and on each of the camera and
protocol lanes under it that does not propagate to it. A lane propagates in
the simulator suite, where the log files are declined, and not in a bench
run, where the files keep each lane out of the main log: a handler on
``LVP`` and on a lane that propagates to it writes the lane's lines twice,
and one on ``LVP`` alone misses a lane that does not.
"""

import collections
import logging
from types import SimpleNamespace

from _lumascope.lib.log_helpers import camera_logger, logger
from tests import conftest


def test_the_capture_holds_a_drivers_main_and_camera_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(conftest, 'bench_results_dir', lambda: tmp_path)
    config = SimpleNamespace()
    named = conftest._driver_loggers()
    levels = {lg: lg.level for lg in named}

    conftest._enable_driver_logging(config)
    try:
        logger.debug('a driver debug line')
        camera_logger.debug('a camera debug line')
    finally:
        conftest.pytest_unconfigure(config)
        for lg, level in levels.items():
            lg.setLevel(level)

    [captured] = tmp_path.glob('driver_test_*.log')
    text = captured.read_text()
    assert 'a driver debug line' in text
    assert 'a camera debug line' in text


def _written(monkeypatch, *, lanes_propagate):
    for lane in ('LVP.camera', 'LVP.protocol'):
        monkeypatch.setattr(logging.getLogger(lane), 'propagate', lanes_propagate)
    written = []
    handler = logging.Handler()
    handler.emit = lambda record: written.append(record.getMessage())
    loggers = conftest._driver_loggers()
    for driver_logger in loggers:
        driver_logger.addHandler(handler)
    try:
        logging.getLogger('LVP').warning('main line')
        logging.getLogger('LVP.camera').warning('camera line')
        logging.getLogger('LVP.protocol').warning('protocol line')
    finally:
        for driver_logger in loggers:
            driver_logger.removeHandler(handler)
    return collections.Counter(written)


def test_each_line_is_written_once_whether_the_lanes_propagate_or_not(monkeypatch):
    once = {'main line': 1, 'camera line': 1, 'protocol line': 1}
    assert _written(monkeypatch, lanes_propagate=True) == once, 'the simulator suite'
    assert _written(monkeypatch, lanes_propagate=False) == once, 'a bench run'
