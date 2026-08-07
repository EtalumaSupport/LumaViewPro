# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests: post-processing accounts for every dropped input.

Bug shape: the shared load_folder funnel drops inputs at two sites with
no accounting -- single-image groups are skipped at DEBUG only, and
_filter_ignored_types removes derived-artifact rows (composites, videos,
stacks, already-processed outputs) before grouping. The completion and
empty-result messages are computed only from the survivors, so they
misattribute what happened: a folder of composites fed to the stitcher
claims "requires multiple tile positions" (bench evidence in the 2026-07-29
support bundle), and a well whose group collapsed to one image vanishes
from the count with no visible trace.

Contract under test: every dropped input is counted and named. The
success message carries the skip census; the empty-result message names
the real reason (what the filter excluded), not a structural guess; the
unattended-notification surface carries the same census.
"""

from unittest.mock import MagicMock

import pandas as pd

from modules.common_utils import PostFunction
from modules.protocol_post_processor import ProtocolPostProcessor
from modules.protocol_post_processing_result import PostProcResult
from modules.stitcher import Stitcher

from tests.test_capture_collision_policy import TILING_CONFIGS


class _FakePostProcessor(ProtocolPostProcessor):
    """Minimal concrete subclass driving the BASE class load_folder logic."""

    def __init__(self, drop_flag=None):
        super().__init__(post_function=PostFunction.HYPERSTACK, has_turret=False)
        self._drop_flag = drop_flag

    def _get_groups(self, df):
        return [(key, group) for key, group in df.groupby('GroupKey')]

    def _generate_filename(self, df, **kwargs):
        return df.iloc[0]['OutName']

    def _filter_ignored_types(self, df):
        if self._drop_flag is not None:
            df = df[df[self._drop_flag] == False]  # noqa: E712 -- pandas mask
        return df

    def _group_algorithm(self, path, df, **kwargs):
        return PostProcResult.ok(significant_bits=8)

    def _add_record(self, protocol_post_record, alg_metadata, root_path, **kwargs):
        pass


def _row(filepath, group_key, out_name, **flags):
    row = {'Filepath': filepath, 'GroupKey': group_key, 'OutName': out_name}
    row.update(dict.fromkeys(PostFunction.list_values(), False))
    for flag, value in flags.items():
        row[flag] = value
    return row


def _drive(processor, tmp_path, monkeypatch, images_df, popup=None):
    post_record = MagicMock()
    post_record.file_exists_in_records.return_value = False
    monkeypatch.setattr(
        processor._post_processing_helper,
        'load_folder',
        lambda **kwargs: {
            'status': True,
            'images_df': images_df,
            'root_path': tmp_path,
            'protocol_post_record': post_record,
            'protocol': None,
        },
    )
    return processor.load_folder(path=tmp_path, tiling_configs_file_loc=TILING_CONFIGS, popup=popup)


def _capture_notifications(monkeypatch):
    import modules.protocol_post_processor as ppp

    captured = []
    for method in ('notice', 'info', 'warning', 'error', 'critical'):
        monkeypatch.setattr(
            ppp.notifications,
            method,
            lambda category, title, message, _m=method, **kw: captured.append((_m, title, message)),
        )
    return captured


def _mixed_group_df():
    """One 2-image group (eligible) + one 1-image group (skipped today)."""
    return pd.DataFrame(
        [
            _row('g0_f0.tiff', 0, 'out0'),
            _row('g0_f1.tiff', 0, 'out0'),
            _row('g1_f0.tiff', 1, 'out1'),
        ]
    )


def test_success_message_carries_single_image_skip_census(tmp_path, monkeypatch):
    result = _drive(
        _FakePostProcessor(), tmp_path, monkeypatch, _mixed_group_df(), popup=MagicMock()
    )

    assert result['status'] is True
    assert result['new_count'] == 1
    message = result['message']
    assert message.startswith('Success.'), 'census must append, never prepend'
    assert 'single' in message.lower() and '1' in message, (
        f'success message must name the skipped single-image group; got: {message!r}'
    )


def test_unattended_completion_carries_the_census(tmp_path, monkeypatch):
    captured = _capture_notifications(monkeypatch)
    result = _drive(_FakePostProcessor(), tmp_path, monkeypatch, _mixed_group_df(), popup=None)

    assert result['status'] is True
    completions = [c for c in captured if c[0] == 'notice' and c[1].endswith('Saved')]
    assert len(completions) == 1
    assert 'single' in completions[0][2].lower(), (
        f'unattended completion must carry the skip census; got: {completions[0][2]!r}'
    )


def test_stitcher_empty_result_names_the_composite_exclusion(tmp_path, monkeypatch):
    """A composite-only folder must be told composites are excluded --
    not that the folder lacks multi-tile structure (it has it)."""
    columns = {
        'Scan Count': 0,
        'Z-Slice': 0,
        'Well': 'A1',
        'Color': 'Composite',
        'Objective': 'obj',
        'Tile Group ID': 0,
        'Custom Step': False,
        'Raw': True,
    }
    rows = []
    for tile in range(4):
        row = dict(columns)
        row['Filepath'] = f'composite_t{tile}.tiff'
        row.update(dict.fromkeys(PostFunction.list_values(), False))
        row[PostFunction.COMPOSITE.value] = True
        rows.append(row)
    df = pd.DataFrame(rows)

    stitcher = Stitcher(has_turret=False)
    result = _drive(stitcher, tmp_path, monkeypatch, df, popup=MagicMock())

    assert result['status'] is False
    message = result['message']
    assert 'composite' in message.lower(), (
        f'empty-result message must name the composite exclusion; got: {message!r}'
    )
    assert 'tile positions' not in message.lower(), (
        'the multi-tile structural guess misattributes a filter exclusion'
    )


def test_empty_result_census_is_generic_across_subclasses(tmp_path, monkeypatch):
    """The census lives in the BASE funnel: any subclass filter's drops are
    named from the dropped rows' own PostFunction flags."""
    df = pd.DataFrame(
        [
            _row('v0.tiff', 0, 'out0', **{PostFunction.VIDEO.value: True}),
            _row('v1.tiff', 0, 'out0', **{PostFunction.VIDEO.value: True}),
        ]
    )
    processor = _FakePostProcessor(drop_flag=PostFunction.VIDEO.value)
    result = _drive(processor, tmp_path, monkeypatch, df, popup=MagicMock())

    assert result['status'] is False
    assert 'video' in result['message'].lower(), (
        f'empty-result message must name the dropped category; got: {result["message"]!r}'
    )
