# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A post-processing build whose output cannot be written stops there, and says so once.

The build loop's contract (``ProtocolPostProcessor._load_folder_inner``): an
``OSError`` out of a group's build or out of the record's append, or a video
the writer could not finish, ends the build as one ``PostProcessingFailedError``
naming the output it could not write, as the folder names it, in the system's
words, with what it had made before. The groups after it are not attempted:
on a full disk every one would fail the same way. A defect inside a group's
build is not a group failure: it ends the build as itself.
"""

from __future__ import annotations

import errno
import os
import pathlib

import pytest

from _lumascope.modules import composite_generation
from _lumascope.modules.channel_record import release_colors
from _lumascope.modules.composite_generation import CompositeGeneration
from _lumascope.modules.exceptions import PostProcessingFailedError
from _lumascope.modules.video_builder import VideoBuilder
from tests.installation_fixtures import shipped_grids
from tests.stock_channels import LED1
from tests.test_a_group_whose_frames_cannot_be_used_fails_alone import (
    _MODES_BIND,
    _hyperstack,
    _outputs,
    _run_folder,
    _z_stacks,
    _zproject,
)
from tests.video_close_faults import fail_the_close


def _a_disk_that_refuses(monkeypatch, refused) -> list[str]:
    """Every file taking a name *refused* answers True for is refused as a full disk.

    Returns the names refused, in order.
    """
    # a stand-in by design: a disk that fills partway through a build, which
    # no folder in a test produces; every saved file takes its name through
    # os.replace, so the refusal lands where a full disk's does
    real_replace = os.replace
    told = []

    def replace(src, dst, *args, **kwargs):
        name = pathlib.Path(dst).name
        if refused(name):
            told.append(name)
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), str(src))
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, 'replace', replace)
    return told


def _composite_folder(tmp_path) -> pathlib.Path:
    """Three positions, BF and LED 1 at each: three composites, A1 to A3."""
    return _run_folder(
        tmp_path,
        [
            {'name': f'{well}_{color}', 'well': well, 'x': 14.38 + 9 * w, 'color': color}
            for w, well in enumerate(('A1', 'A2', 'A3'))
            for color in ('BF', LED1)
        ],
    )


def _composite(folder):
    return CompositeGeneration(has_turret=False).load_folder(
        path=folder,
        tiling_grids=shipped_grids(),
        brightness_thresholds_percent={LED1: 50},
        channel_colors=release_colors(),
    )


def _video(folder):
    return VideoBuilder(has_turret=False).load_folder(
        path=folder,
        tiling_grids=shipped_grids(),
        frame_rate_fps=5,
        enable_timestamp_overlay=False,
        channel_colors=release_colors(),
    )


def _video_folder(tmp_path) -> pathlib.Path:
    return _run_folder(
        tmp_path,
        [
            {'name': f'{well}_BF', 'well': well, 'x': 14.38 + 9 * w, 'scan': scan}
            for w, well in enumerate(('A1', 'A2'))
            for scan in range(3)
        ],
    )


def test_a_composite_whose_second_output_cannot_be_written_stops_there_naming_it(
    tmp_path, monkeypatch
):
    folder = _composite_folder(tmp_path)
    refused = _a_disk_that_refuses(monkeypatch, lambda name: name.startswith('A2_Composite'))

    with pytest.raises(PostProcessingFailedError) as stopped:
        _composite(folder)

    told = str(stopped.value)
    assert len(refused) == 1, refused
    assert f'Composite/{refused[0]}' in told, told
    assert '.tmp' not in told and '[Errno' not in told, told
    assert 'No space left on device' in told, told
    # The cause is the whole answer: the person is never sent to the log.
    assert 'log' not in told, told
    assert stopped.value.reason == 'post_processing_incomplete'
    assert [pathlib.Path(p).name for p in stopped.value.produced_paths] == [
        name for name in _outputs(folder) if name.startswith('A1_Composite')
    ]
    assert not any(name.startswith('A3') for name in _outputs(folder)), 'the build went on'


# parametrize-justified: each case is a different builder over its own folder shape
@pytest.mark.parametrize('build', [_zproject, _hyperstack], ids=['z-projection', 'hyperstack'])
def test_a_stack_build_whose_output_cannot_be_written_stops_there_naming_it(
    tmp_path, monkeypatch, build
):
    folder = _z_stacks(tmp_path)
    refused = _a_disk_that_refuses(monkeypatch, lambda name: name.startswith('A2'))

    with pytest.raises(PostProcessingFailedError) as stopped:
        build(folder)

    told = str(stopped.value)
    assert len(refused) == 1, refused
    assert refused[0] in told and '.tmp' not in told and '[Errno' not in told, told
    assert 'No space left on device' in told, told
    assert [pathlib.Path(p).name[:2] for p in stopped.value.produced_paths] == ['A1']


def test_a_record_that_cannot_be_updated_is_told_beside_the_stop_not_in_its_place(
    tmp_path, monkeypatch
):
    # A1 and A2 built, then both deleted: A1 is built again, so the record's
    # row for it is replaced, which the build's end writes as a whole rewrite.
    folder = _composite_folder(tmp_path)
    _composite(folder)
    for made in (folder / 'Composite').glob('A[12]_*'):
        made.unlink()
    _a_disk_that_refuses(
        monkeypatch,
        lambda name: name.startswith('A2_Composite') or name == 'protocol_post_record.tsv',
    )

    with pytest.raises(PostProcessingFailedError) as stopped:
        _composite(folder)

    told = str(stopped.value)
    assert 'A2_Composite' in told, told
    assert 'The post-processing record could not be updated (No space left on device)' in told


@_MODES_BIND
def test_an_output_the_record_cannot_list_stops_the_build_naming_it_as_saved(tmp_path):
    # The projection writes the record; the hyperstack's first output is then
    # saved and its row cannot be appended to a record the system will not write.
    folder = _z_stacks(tmp_path)
    _zproject(folder)
    record = folder / 'protocol_post_record.tsv'
    record.chmod(0o400)
    try:
        with pytest.raises(PostProcessingFailedError) as stopped:
            _hyperstack(folder)
    finally:
        record.chmod(0o600)

    told = str(stopped.value)
    assert 'record could not list it (Permission denied)' in told, told
    (saved,) = stopped.value.produced_paths
    assert pathlib.Path(saved).name.startswith('A1') and pathlib.Path(saved).is_file()
    assert not any(n.startswith('A2') and 'hyperstack' in n for n in _outputs(folder))


def test_a_defect_in_a_composites_build_ends_the_build_as_itself(tmp_path, monkeypatch):
    folder = _composite_folder(tmp_path)

    def build_composite(**_kwargs):
        # a stand-in by design: a defect, which no input produces
        raise TypeError('a defect in the build')

    monkeypatch.setattr(composite_generation, 'build_composite', build_composite)

    with pytest.raises(TypeError, match='a defect in the build'):
        _composite(folder)


def test_a_video_that_cannot_be_finished_stops_the_build_and_leaves_no_file(tmp_path, monkeypatch):
    folder = _video_folder(tmp_path)
    failing = fail_the_close(monkeypatch, of=lambda path: 'A1' in pathlib.Path(path).name)

    with pytest.raises(PostProcessingFailedError) as stopped:
        _video(folder)

    assert len(failing) == 1
    told = str(stopped.value)
    assert 'could not be finished' in told and 'No space left on device' in told, told
    assert pathlib.Path(failing[0]).name not in told, 'the words named the temporary file'
    assert 'A1_BF' in told, told
    assert not pathlib.Path(failing[0]).exists()
    assert list(folder.rglob('*.mp4')) == [], 'the build went on past the stop'


def test_a_video_that_cannot_take_its_name_stops_the_build(tmp_path, monkeypatch):
    folder = _video_folder(tmp_path)
    refused = _a_disk_that_refuses(monkeypatch, lambda name: name.startswith('A1'))

    with pytest.raises(PostProcessingFailedError) as stopped:
        _video(folder)

    told = str(stopped.value)
    assert refused[0] in told and '[Errno' not in told, told
    assert list(folder.rglob('*.mp4')) == [], 'the build went on past the stop'
