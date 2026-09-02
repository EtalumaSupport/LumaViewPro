# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression for issue #407: the protocol-end hyperstack save was silent.

Bug shape: ProtocolPostProcessor.load_folder had exactly one lifecycle
surface -- the popup the attended (Post Processing tab) callers pass.
The one unattended caller (protocol-end hyperstack, popup=None) got a
log line at INFO (invisible to normal users: the popup bridge subscribes
at NOTICE+) and its returned status dict was discarded, so neither
completion nor failure ever reached the screen.

Contract under test: when popup is None, load_folder itself announces
start (once the group count is known) and completion/failure at exit --
for every subclass. Attended runs (popup passed) are unchanged.
"""

from unittest.mock import MagicMock

import pandas as pd

from modules.common_utils import PostFunction
from modules.protocol_post_processor import ProtocolPostProcessor
from modules.protocol_post_processing_result import PostProcResult

from tests.test_capture_collision_policy import TILING_CONFIGS


class _FakePostProcessor(ProtocolPostProcessor):
    """Minimal concrete subclass driving the BASE class load_folder logic."""

    def __init__(self, fail_groups=False):
        super().__init__(post_function=PostFunction.HYPERSTACK, has_turret=False)
        self._fail_groups = fail_groups

    def _get_groups(self, df):
        return [(key, group) for key, group in df.groupby('GroupKey')]

    def _generate_filename(self, df, **kwargs):
        return df.iloc[0]['OutName']

    def _filter_ignored_types(self, df):
        return df

    def _group_algorithm(self, path, df, **kwargs):
        if self._fail_groups:
            return PostProcResult.failed(error='synthetic group failure')
        return PostProcResult.ok(significant_bits=8)

    def _add_record(self, protocol_post_record, alg_metadata, root_path, **kwargs):
        pass


def _images_df():
    rows = []
    for i in range(2):  # one group, two frames -> eligible
        row = {'Filepath': f'g0_f{i}.tiff', 'GroupKey': 0, 'OutName': 'out1'}
        row.update(dict.fromkeys(PostFunction.list_values(), False))
        rows.append(row)
    return pd.DataFrame(rows)


def _drive(processor, tmp_path, monkeypatch, popup=None, **load_kwargs):
    post_record = MagicMock()
    post_record.file_exists_in_records.return_value = False
    monkeypatch.setattr(
        processor._post_processing_helper,
        'load_folder',
        lambda **kwargs: {
            'status': True,
            'images_df': _images_df(),
            'root_path': tmp_path,
            'protocol_post_record': post_record,
            'protocol': None,
        },
    )
    return processor.load_folder(
        path=tmp_path, tiling_configs_file_loc=TILING_CONFIGS, popup=popup, **load_kwargs
    )


def _capture_notifications(monkeypatch):
    """Record (method, title, message) for every bus emission."""
    import modules.protocol_post_processor as ppp

    captured = []
    for method in ('notice', 'info', 'warning', 'error', 'critical'):
        monkeypatch.setattr(
            ppp.notifications,
            method,
            lambda category, title, message, _m=method, **kw: captured.append((_m, title, message)),
        )
    return captured


def test_unattended_run_announces_start_and_completion(tmp_path, monkeypatch):
    captured = _capture_notifications(monkeypatch)
    result = _drive(_FakePostProcessor(), tmp_path, monkeypatch, popup=None)

    assert result['status'] is True
    assert captured, 'unattended load_folder must notify (pre-fix: silent)'
    assert captured[0][0] == 'notice'
    assert captured[0][1].startswith('Saving')
    assert '1 ' in captured[0][2], 'start notice must carry the group count'
    assert captured[-1][0] == 'notice'
    assert captured[-1][1].endswith('Saved')
    assert str(tmp_path) in captured[-1][2], 'completion must name the output root'


def test_attended_run_stays_popup_only(tmp_path, monkeypatch):
    captured = _capture_notifications(monkeypatch)
    popup = MagicMock()
    result = _drive(_FakePostProcessor(), tmp_path, monkeypatch, popup=popup)

    assert result['status'] is True
    assert captured == [], 'popup callers must not get a second surface'


def test_unattended_all_groups_failed_is_loud(tmp_path, monkeypatch):
    captured = _capture_notifications(monkeypatch)
    result = _drive(_FakePostProcessor(fail_groups=True), tmp_path, monkeypatch, popup=None)

    assert result['status'] is False
    errors = [c for c in captured if c[0] == 'error']
    assert len(errors) == 1, f'exactly one failure notification expected; saw {captured}'
    assert errors[0][1].endswith('Save Failed')
    assert result['message'] in errors[0][2]


def test_caller_owned_surface_gets_no_lifecycle_notices(tmp_path, monkeypatch):
    # A run kind that settles its own outcome (the composite merge) owns
    # the surface: neither the start notice nor the result notice may
    # speak for it, on success or on failure.
    captured = _capture_notifications(monkeypatch)
    ok = _drive(_FakePostProcessor(), tmp_path, monkeypatch, popup=None, announce=False)
    failed = _drive(_FakePostProcessor(fail_groups=True), tmp_path, monkeypatch, announce=False)

    assert ok['status'] is True and failed['status'] is False
    assert captured == [], f'announce=False must silence the loader; saw {captured}'
