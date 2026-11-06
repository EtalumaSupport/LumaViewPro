# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The one timestamp every LumaViewPro log line starts with.

Every record begins ``2026-10-29 17:10:57.320``: the full local date and
time, so one-line records from any two logs merge and sort as plain text; a
traceback's later lines carry no stamp and read with the line before them,
and the hour clocks repeat when daylight saving ends sorts out of order.
This module
imports only the standard library and does nothing at import, so the FX2
driver step (LocalSystem, at uninstall), the profiler (under sudo) and the
engineering plugin's tests can share the stamp without importing
``lvp_logger``, which loads the settings channel and whose start opens the
log files.
"""

from __future__ import annotations

import logging
import time

LOG_DATEFMT = '%Y-%m-%d %H:%M:%S'

# The bundle logs: main, errors, REST, metrics, protocol, GUI interactions.
MAIN_LAYOUT = '[%(levelname)s] [%(threadName)s] - %(filename)s - %(message)s'

# The trace logs (serial, camera, api) and the engineering plugin's copy of
# serial traffic. Which thread sent a command separates a lane's own write
# from one made beside it.
TRACE_LAYOUT = '[%(threadName)s] %(message)s'


def make_formatter(layout: str) -> logging.Formatter:
    """A formatter whose every record starts with the stamp, then ``layout``."""
    return logging.Formatter(f'%(asctime)s.%(msecs)03d {layout}', datefmt=LOG_DATEFMT)


def now_stamp() -> str:
    """The stamp for a line written by hand rather than through a handler."""
    now = time.time()
    return f'{time.strftime(LOG_DATEFMT, time.localtime(now))}.{int(now * 1000) % 1000:03d}'
