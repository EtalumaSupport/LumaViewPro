# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""``--driver-log`` writes the drivers' own lines, at DEBUG, to its file.

The IDS and FX2 bench procedures read that file. The drivers take their
loggers from src/_lumascope/lib/log_helpers by name, so the capture is on those named
loggers; a capture on anything the drivers do not write to keeps an empty
file through a whole bench run.
"""

import logging
from types import SimpleNamespace

from _lumascope.lib.log_helpers import camera_logger, logger
from tests import conftest


def test_the_capture_holds_a_drivers_main_and_camera_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(conftest, 'bench_results_dir', lambda: tmp_path)
    config = SimpleNamespace()
    named = [logging.getLogger(name) for name in conftest._DRIVER_LOG_NAMES]
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
