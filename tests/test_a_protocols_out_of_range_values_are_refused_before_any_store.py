# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A protocol's out-of-range values are refused before anything is stored or moved.

A protocol whose BF Layer Settings row held Exposure 0, Sum 0 and Gain -3
loaded, and ``apply_layer_settings`` stored all three in the live
settings, values no write of them could store; going to a step stored the
step's own the same way, once its moves had started. The load still takes
the file, so a person can fix it in the app, and tells the values in its
notice; applying the row, or going to such a step, is refused whole by the
settings' layer ranges, naming each value, before any store or move.
"""

import pytest

from modules.exceptions import ProtocolRunRefusedError, ProtocolStepsInvalidNotice
from modules.notification_center import notifications
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


@pytest.fixture(scope='module')
def session(tmp_path_factory):
    s = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path_factory.mktemp('live'))), simulate=True
    )
    s.run_application_session()
    s.set_layer_acquire('BF', 'image')
    yield s
    s.shutdown()


def _edited(session, tmp_path, block_row, cells):
    """``session``'s new protocol, saved, with ``cells`` written over one row of it.

    ``block_row`` picks the row: the header row's first cell, and the cell
    the row starts with.
    """
    saved = session.save_protocol(session.new_protocol(), tmp_path / 'good.tsv')
    lines = saved.read_text().split('\n')
    header_start, row_start = block_row
    header_at = next(k for k, line in enumerate(lines) if line.startswith(f'{header_start}\t'))
    header = lines[header_at].split('\t')
    for k in range(header_at + 1, len(lines)):
        row = lines[k].split('\t')
        if row[0] == row_start:
            for column, value in cells.items():
                row[header.index(column)] = value
            lines[k] = '\t'.join(row)
            break
    edited = tmp_path / 'bad.tsv'
    edited.write_text('\n'.join(lines))
    return edited


def test_the_load_tells_the_layer_settings_values_and_the_apply_refuses_them_all_storing_none(
    session, tmp_path, monkeypatch
):
    edited = _edited(
        session, tmp_path, ('Layer', 'BF'), {'Exposure': '0', 'Sum': '0', 'Gain': '-3'}
    )
    heard = []
    monkeypatch.setattr(
        notifications, 'report_outcome', lambda outcome, **kw: heard.append(outcome)
    )

    protocol = session.load_protocol(str(edited))

    (notice,) = [n for n in heard if isinstance(n, ProtocolStepsInvalidNotice)]
    for words in (
        'BF Exposure: it must be above 0',
        'BF Gain: it must be at least 0',
        'BF Sum: it must be a whole number of at least 1',
    ):
        assert words in str(notice), str(notice)

    before = session.get_settings_snapshot()
    with pytest.raises(ProtocolRunRefusedError) as refused:
        session.apply_layer_settings(protocol)
    assert refused.value.reason == 'values_out_of_range'
    for words in ('BF Exposure:', 'BF Gain:', 'BF Sum:'):
        assert words in str(refused.value), str(refused.value)
    assert session.get_settings_snapshot() == before


def test_going_to_a_step_holding_an_out_of_range_value_starts_no_move_and_stores_nothing(
    session, tmp_path
):
    edited = _edited(
        session, tmp_path, ('Name', session.new_protocol().step(idx=0)['Name']), {'Exposure': '0'}
    )
    protocol = session.load_protocol(str(edited))
    # Away from the step, so going to it would move X.
    session.scope.motion.move_relative('X', 1000)
    before = session.get_settings_snapshot()
    position = session.scope.motion.get_target_position('X')

    with pytest.raises(ProtocolRunRefusedError) as refused:
        session.start_go_to_step(protocol, 0)

    assert refused.value.reason == 'values_out_of_range'
    assert 'Step 1' in str(refused.value) and 'Exposure:' in str(refused.value)
    assert session.get_settings_snapshot() == before
    assert session.scope.motion.get_target_position('X') == position
