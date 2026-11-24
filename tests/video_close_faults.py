# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A video whose close fails as a full disk fails it, at the PyAV boundary.

A stand-in by design: a disk that fills under a video, which the simulator
has no member to produce. The encode and the close are PyAV's own; the
container refuses with the ``ENOSPC`` PyAV raised on a real full volume, its
close only after it has really closed.
"""

from __future__ import annotations

import errno
import os

import _lumascope.modules.video_writer as video_writer_module


def _disk_full(path: str) -> OSError:
    # Named as PyAV names it, with the file.
    return OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), path)


class FailingVideos(list):
    """The paths of the videos told to fail, in the order opened, and how
    many packets their full disk refused."""

    packets_refused = 0


class _DiskFills:
    def __init__(self, path, container, failing: FailingVideos, after_packets: int | None):
        self._path = path
        self._container = container
        self._failing = failing
        self._room = after_packets

    def add_stream(self, *args, **kwargs):
        return self._container.add_stream(*args, **kwargs)

    def mux(self, packet):
        if self._room is not None:
            if self._room == 0:
                self._failing.packets_refused += 1
                raise _disk_full(self._path)
            self._room -= 1
        self._container.mux(packet)

    def close(self):
        self._container.close()
        raise _disk_full(self._path)


def fail_the_close(monkeypatch, *, of=lambda path: True, after_packets=None) -> FailingVideos:
    """Every video opened whose path *of* accepts fails at its close.

    With *after_packets*, its disk fills mid-stream: every packet after that
    many is refused too, so frames fail to encode before the close fails.
    """
    failing = FailingVideos()
    real_open = video_writer_module.av.open

    def _open(path, *args, **kwargs):
        container = real_open(path, *args, **kwargs)
        if not of(path):
            return container
        failing.append(path)
        return _DiskFills(path, container, failing, after_packets)

    monkeypatch.setattr(video_writer_module.av, 'open', _open)
    return failing


class _DiskRefuses:
    def __init__(self, path, container, failing: FailingVideos, refused: frozenset[int]):
        self._path = path
        self._container = container
        self._failing = failing
        self._refused = refused
        self._packets = 0

    def add_stream(self, *args, **kwargs):
        return self._container.add_stream(*args, **kwargs)

    def mux(self, packet):
        number = self._packets
        self._packets += 1
        if number in self._refused:
            self._failing.packets_refused += 1
            raise _disk_full(self._path)
        self._container.mux(packet)

    def close(self):
        self._container.close()


def refuse_packets(monkeypatch, packets: frozenset[int]) -> FailingVideos:
    """Every video opened has the packets numbered *packets* refused by a full disk.

    The disk fills for those packets and has room again after them (another
    program's file came and went), so the frames they carried fail to encode
    and the video still finishes.
    """
    failing = FailingVideos()
    real_open = video_writer_module.av.open

    def _open(path, *args, **kwargs):
        failing.append(path)
        return _DiskRefuses(path, real_open(path, *args, **kwargs), failing, packets)

    monkeypatch.setattr(video_writer_module.av, 'open', _open)
    return failing
