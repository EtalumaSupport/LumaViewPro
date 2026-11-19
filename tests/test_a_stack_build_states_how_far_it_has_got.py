# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A stack build states how far it has got.

A recording's stack said nothing while it built (393 s on a slow PC, bundle
SN12075-2026-11-10-144813), so a person saw captures refused and no reason. A stack reads every plane three times, and
its percent counts every read, so it moves from the first; a build of
several stacks reports each one's percent as its share of the build's.
"""

from _lumascope.modules.channel_record import release_colors
from _lumascope.modules.protocol_post_processor import group_progress
from _lumascope.modules.stack_builder import StackBuilder
from tests.test_hyperstack_video_grouping import _stack_df, _write_frames


def test_a_stacks_percent_rises_from_its_first_read_to_100(tmp_path):
    planes = 4
    df = _stack_df([(f'A1_BF_{t:04}.tiff', t, 'BF', 0) for t in range(planes)])
    _write_frames(tmp_path, df)
    told = []

    result = StackBuilder.create_single_recording_stack(
        df=df,
        path=tmp_path,
        output_file_loc=tmp_path / 'stack.ome.tiff',
        save_encoding='right_aligned',
        channel_colors=release_colors(),
        on_progress=lambda percent, _status: told.append(percent),
    )

    assert result['status'], result['error']
    assert len(told) == 3 * planes, told
    assert told == sorted(set(told)), f'the percent must only rise: {told}'
    assert told[0] > 0 and told[-1] == 100, told


def test_a_build_of_several_stacks_never_falls_back_at_a_stacks_start():
    told = []

    def heard(percent, _status):
        told.append(percent)

    for current in (1, 2, 3):
        report = group_progress(heard, current_group=current, total_groups=3)
        for percent in (0, 50, 100):
            report(percent, None)

    assert told == sorted(told), f'the build fell back: {told}'
    assert (told[0], told[-1]) == (0, 100), told
    assert group_progress(None, current_group=1, total_groups=3) is None
