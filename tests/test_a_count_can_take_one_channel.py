# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A cell count can be limited to one channel.

One Object Analysis method measured BF and fluorescence images alike, so a
run of both could not be counted with a method for each. ``count_cells``
takes a channel: only the images taken in it are counted, by the channel
the file states, else its run record's (a JPG states none), an old run's
name read as the channel it names today; every other image is left out and
named. A name that is no channel of the release is refused before the
folder is read.
"""

from __future__ import annotations

import datetime

import numpy as np
import pytest

from _lumascope.modules import image_utils
from _lumascope.modules.cell_count_method import CellCountMethod
from _lumascope.modules.channel_record import release_color, release_old_names
from _lumascope.modules.exceptions import ArgumentRefusedError, LeftOutReason
from _lumascope.modules.lumascope_api.frame_record import TimestampSource
from tests.count_runs import (
    copy_run,
    count_scope,  # noqa: F401 -- the module's runs, a fixture
    images_in,
    results_rows,
    session,  # noqa: F401 -- a fixture
)
from tests.stock_channels import LED0


def _count(session, folder, channel):
    return session.post_processing.count_cells(folder, method=CellCountMethod(), channel=channel)


def test_a_run_counted_for_one_channel_counts_its_images_alone_and_names_the_rest(
    count_scope, session, tmp_path
):
    folder = copy_run(count_scope, 'plate', tmp_path)

    result = _count(session, folder, 'BF')

    rows = results_rows(folder)
    assert rows and {row['channel'] for row in rows} == {'BF'}
    others = set(images_in(folder)) - {row['file'] for row in rows}
    assert others and all(LED0 in name for name in others)
    assert {(str(left.path), left.reason) for left in result.left_out} == {
        (name, LeftOutReason.OTHER_CHANNEL) for name in others
    }


def test_a_jpg_run_counted_for_one_channel_counts_by_its_record(count_scope, session, tmp_path):
    """A JPG states no channel; its record's is the one it was taken in."""
    folder = copy_run(count_scope, 'jpg', tmp_path)

    _count(session, folder, LED0)

    rows = results_rows(folder)
    assert len(rows) == 1 and LED0 in rows[0]['file']


def _still(path, channel):
    """A still that states *channel*, as an earlier release could save it."""
    taken = datetime.datetime(2026, 11, 8, 9, 0, tzinfo=datetime.UTC)
    image_utils.write_tiff(
        data=np.full((32, 32), 40, dtype=np.uint8),
        file_loc=path,
        metadata={
            'objective': {},
            'pixel_size_um': None,
            'channel': channel,
            'datetime': taken,
            'timestamp_iso': taken.isoformat(timespec='microseconds'),
            'timestamp_source': TimestampSource.CAMERA_FRAME_START.value,
        },
        ome=False,
        color=release_color('BF'),
        significant_bits=8,
        save_encoding='right_aligned',
        white_bits=None,
    )


def test_an_old_runs_name_is_counted_as_the_channel_it_names_today(session, tmp_path):
    old = release_old_names()[0]
    _still(tmp_path / 'old.tiff', old.name)
    _still(tmp_path / 'bf.tiff', 'BF')

    _count(session, tmp_path, old.stock_key)

    assert [row['file'] for row in results_rows(tmp_path)] == ['old.tiff']


def test_an_image_stating_no_channel_is_left_out_of_a_one_channel_count(session, tmp_path):
    from PIL import Image

    _still(tmp_path / 'bf.tiff', 'BF')
    Image.fromarray(np.full((32, 32), 40, dtype=np.uint8)).save(tmp_path / 'bare.png')

    result = _count(session, tmp_path, 'BF')

    assert [row['file'] for row in results_rows(tmp_path)] == ['bf.tiff']
    assert [(str(left.path), left.reason) for left in result.left_out] == [
        ('bare.png', LeftOutReason.OTHER_CHANNEL)
    ]


def test_a_name_that_is_no_channel_is_refused_before_the_folder_is_read(session, tmp_path):
    """Refused for the name, not for a folder that is not there."""
    with pytest.raises(ArgumentRefusedError) as refused:
        _count(session, tmp_path / 'nothing_here', 'bf')

    assert refused.value.reason == 'channel_unknown'
