# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The host's wall time as LVP saves and publishes it: local, with its UTC offset.

Every time LVP saves or publishes as an instant is made here, so every one
states the offset in force at its instant and any two can be compared,
however many daylight-saving changes lie between them. A local time without
its offset names no instant in the hour a clock change repeats, and two of
them across a change differ by the wrong amount.

The wall clock itself is read in ``drivers.host_clock`` (``wall_ns``); this
module turns that integer into the saved form. A timezone has no place
beside the steady clock that module pairs the wall clock with, and the
callers that publish times (REST among them) import modules, not drivers.

``tests/guards/test_the_wall_clock_has_one_reader.py`` fails the build on a
read of the wall clock as a datetime anywhere else in production.
"""

import datetime

from drivers import host_clock


def local_time(wall_ns: int) -> datetime.datetime:
    """The instant ``wall_ns`` (nanoseconds since the epoch, as
    ``host_clock.wall_ns`` reads them) as host local time with the offset in
    force at that instant, to the nearest microsecond, a datetime's
    resolution.

    Made through UTC, so it names its instant in any hour, the repeated one
    included; a file's modification time is ``local_time(stat.st_mtime_ns)``.
    """
    seconds, microseconds = divmod((wall_ns + 500) // 1000, 1_000_000)
    utc = datetime.datetime.fromtimestamp(seconds, datetime.UTC)
    return utc.replace(microsecond=microseconds).astimezone()


def now() -> datetime.datetime:
    """The host's wall time now, local with its offset."""
    return local_time(host_clock.wall_ns())
