# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A protocol run folder whose video steps' frames are on disk, as a run saves them.

Each recording's frames sit in its own folder under the run, written through
the one recorded-frame writer so each states the time it was taken (or, for
an untimed recording, as pixels with no record), beside the recording's
``recording_manifest.json``; a still is one TIFF a scan. The run's protocol
and execution record name them as a run of this release does.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import pathlib

import numpy as np
import tifffile as tf

from _lumascope.modules.lumascope_api.frame_record import FrameTime, TimestampSource
from tests.frame_records import recorded_record, write_recorded
from tests.test_post_processing_stays_inside_its_folder import _COLUMNS, _step

_BASE = datetime.datetime(2026, 10, 7, 12, 0, 0).astimezone()


@dataclasses.dataclass(frozen=True)
class Recording:
    """One video step's recording, saved as frames.

    ``seconds`` are the frames' times from the recording's start, numbered in
    order (a lost frame is a gap in them); ``timed=False`` writes the frames
    with no stated time. ``measured_fps`` is the manifest's, None for none.
    """

    name: str
    seconds: tuple[float, ...]
    scan: int = 0
    well: str = 'A1'
    color: str = 'BF'
    label: str = ''
    custom: bool = False
    timed: bool = True
    measured_fps: float | None = None


@dataclasses.dataclass(frozen=True)
class Still:
    """One still step's capture in one scan."""

    name: str
    scan: int
    well: str = 'A1'
    color: str = 'BF'


def run_folder(scope, tmp_path: pathlib.Path, captures: list[Recording | Still]) -> pathlib.Path:
    """Write a run folder holding ``captures`` and return it."""
    folder = tmp_path / 'ProtocolData' / '20261007_120000'
    folder.mkdir(parents=True)
    steps, record, index = [], [], {}
    for seed, capture in enumerate(captures):
        if capture.name not in index:
            index[capture.name] = len(steps)
            step = _step(capture.name, '5000.0', '0', capture.well)
            step[5] = capture.color
            if isinstance(capture, Recording):
                step[16] = str(capture.custom)
                step[19] = 'video'
                step[23] = capture.label
            steps.append(step)
        at = _BASE + datetime.timedelta(minutes=capture.scan)
        if isinstance(capture, Recording):
            recorded = f'{capture.name}_{capture.scan:04d}_video'
            _write_recording(scope, folder / recorded, capture, at)
            count = len(capture.seconds)
        else:
            recorded = f'{capture.name}_{capture.scan:04d}.tiff'
            write_recorded(
                scope,
                np.full((64, 64), 100 + seed, dtype=np.uint8),
                folder / recorded,
                channel=capture.color,
                record=recorded_record(captured_at=at),
            )
            count = 1
        record.append(
            f'{recorded}\t{capture.name}\t{index[capture.name]}\t{capture.scan}\t'
            f'{at.strftime("%Y-%m-%d %H:%M:%S")}.000001\t{count}\t0.0\n'
        )
    (folder / 'run.tsv').write_text(
        'LumaViewPro Protocol\nVersion\t8\nPeriod\t1.0\nDuration\t0.5\n'
        'Labware\t96 well microplate\nCapture Root\t\n\nSteps\n'
        + '\t'.join(_COLUMNS)
        + '\n'
        + ''.join('\t'.join(step) + '\n' for step in steps)
    )
    (folder / 'protocol_record.tsv').write_text(
        'LumaViewPro Protocol Execution Record\nVersion\t3\nProtocol File\trun.tsv\n'
        'Filename\tStep Name\tStep Index\tScan Count\tTimestamp\tFrame Count\tDuration (s)\n'
        + ''.join(record)
    )
    return folder


def _write_recording(scope, folder: pathlib.Path, recording: Recording, start) -> None:
    folder.mkdir()
    for n, s in enumerate(recording.seconds):
        path = folder / f'{folder.name}_Frame_{n:04d}.tiff'
        pixels = np.full((64, 64), 40 + n, dtype=np.uint8)
        if not recording.timed:
            tf.imwrite(str(path), pixels)
            continue
        write_recorded(
            scope,
            pixels,
            path,
            channel=recording.color,
            record=recorded_record(
                captured_at=FrameTime(
                    at=start + datetime.timedelta(seconds=s),
                    source=TimestampSource.HOST_ARRIVAL,
                )
            ),
            frame_number=n,
        )
    manifest = {'channel': recording.color, 'false_color': False}
    if recording.measured_fps is not None:
        manifest['measured_fps'] = recording.measured_fps
    (folder / 'recording_manifest.json').write_text(json.dumps(manifest))
