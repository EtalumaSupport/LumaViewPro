# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The memory profiler's summary log starts each line with the full date and time.

``summary.log`` is written by hand, not through a log handler, so it takes
the one stamp from ``modules.log_stamp`` and sorts with LumaViewPro's logs.
"""

import datetime
import re
import tracemalloc

from modules.memory_profiler import MemoryLeakProfiler

# Spelled here, not read from modules.log_stamp, so a wrong date format there
# fails this test instead of agreeing with itself.
_STAMP = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3}) Current: ')


def test_a_summary_line_starts_with_the_full_date_and_time(tmp_path, monkeypatch):
    tracemalloc.start(1)
    try:
        monkeypatch.setattr(MemoryLeakProfiler, 'LOG_DIR', str(tmp_path))
        monkeypatch.setattr(MemoryLeakProfiler, '_first_snapshot', tracemalloc.take_snapshot())
        MemoryLeakProfiler.take_report()
    finally:
        tracemalloc.stop()

    line = (tmp_path / 'summary.log').read_text().splitlines()[0]
    match = _STAMP.match(line)
    assert match, line
    stamped = datetime.datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S.%f')
    assert abs((datetime.datetime.now() - stamped).total_seconds()) < 60
