# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The free space a test sets, read where LVP reads it.

``common_utils.check_disk_space_ok`` reads free space through
``psutil.disk_usage``; patching that one reading sets the free space every
judge of it sees, the start's and the running checks' alike, whatever name a
module imported the probe under. A drive this full is not one the suite can
make on every host, so it is set here.
"""

from types import SimpleNamespace

import _lumascope.modules.common_utils as common_utils

PLENTY_MB = 1_000_000.0


def free_space_is(monkeypatch, free_mb: float) -> None:
    """Every free-space reading answers *free_mb*, whatever folder it is asked about."""
    free = free_mb * 1024**2
    total = max(free, 10**12)
    monkeypatch.setattr(
        common_utils.psutil,
        'disk_usage',
        lambda path: SimpleNamespace(
            total=total, used=total - free, free=free, percent=100 * (total - free) / total
        ),
    )


def free_space_unreadable(monkeypatch) -> None:
    """Every free-space reading raises, as a drive gone after its folder was read does."""

    def unreadable(path):
        raise OSError(f'the drive under {path} did not answer')

    monkeypatch.setattr(common_utils.psutil, 'disk_usage', unreadable)


def the_first_readings_find_room(count: int, answer):
    """A probe whose first *count* readings find room, and whose later ones *answer*.

    A disk that fills once work is under way: the judges read one probe, in
    the order the work reaches them -- a run's start, then its scan's check,
    then each write's.
    """
    readings = []

    def probe(path, required_mb):
        readings.append(path)
        if len(readings) <= count:
            return (True, PLENTY_MB)
        return answer(path, required_mb)

    return probe


def the_start_finds_room(answer):
    """A probe whose first reading, the start's, finds room, and whose later ones *answer*."""
    return the_first_readings_find_room(1, answer)
