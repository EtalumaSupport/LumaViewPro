# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A protocol's out-of-range values are refused before anything is stored or moved.

A protocol whose BF Layer Settings row held Exposure 0, Sum 0 and Gain -3
loaded, and ``apply_channel_settings`` stored all three in the live
settings, values no write of them could store; going to a step stored the
step's own the same way, once its moves had started. Each is now judged by
the settings' layer ranges first: such a file is refused at load, naming
each value, with no notice beside the refusal; a protocol built in memory
is refused by the apply; going to such a step is refused before any move.
A step's values stay lenient at load: the run refuses them.
"""

import pytest

from modules.exceptions import ProtocolRunRefusedError
from modules.notification_center import notifications
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings
from tests.test_validate_steps import _make_protocol, _valid_step


@pytest.fixture(scope='module')
def session(tmp_path_factory):
    s = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path_factory.mktemp('live'))), simulate=True
    )
    s.run_application_session()
    s.set_channel_acquire('BF', 'image')
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


def test_a_file_whose_layer_settings_are_out_of_range_is_refused_at_load_naming_each(
    session, tmp_path, monkeypatch
):
    edited = _edited(
        session, tmp_path, ('Layer', 'BF'), {'Exposure': '0', 'Sum': '0', 'Gain': '-3'}
    )
    heard = []
    monkeypatch.setattr(
        notifications, 'report_outcome', lambda outcome, **kw: heard.append(outcome)
    )
    before = session.get_settings_snapshot()

    with pytest.raises(ProtocolRunRefusedError) as refused:
        session.load_protocol(str(edited))

    assert refused.value.reason == 'values_out_of_range'
    for words in (
        'BF Exposure: it must be above 0',
        'BF Gain: it must be at least 0',
        'BF Sum: it must be a whole number of at least 1',
    ):
        assert words in str(refused.value), str(refused.value)
    # Raised, and nothing told beside it: one answer to one load.
    assert heard == []
    assert session.get_settings_snapshot() == before


def test_a_protocol_built_in_memory_is_refused_by_the_apply_storing_nothing(session):
    # Built from steps alone, its Layer Settings are inferred from its first
    # BF step, Exposure 0 among them.
    protocol = _make_protocol([_valid_step(Color='BF', Exposure=0)])
    before = session.get_settings_snapshot()

    with pytest.raises(ProtocolRunRefusedError) as refused:
        session.apply_channel_settings(protocol)

    assert refused.value.reason == 'values_out_of_range'
    assert 'BF Exposure:' in str(refused.value), str(refused.value)
    assert session.get_settings_snapshot() == before


def test_going_to_a_step_holding_an_out_of_range_value_starts_no_move_and_stores_nothing(
    session, tmp_path
):
    edited = _edited(
        session,
        tmp_path,
        ('Name', session.new_protocol().step_row(step_index=0)['Name']),
        {'Exposure': '0'},
    )
    protocol = session.load_protocol(str(edited))
    # Away from the step, so going to it would move X.
    session.scope.motion.move_xy_by(dx_um=1000)
    before = session.get_settings_snapshot()
    position = session.scope.motion.get_target_positions()['X']

    with pytest.raises(ProtocolRunRefusedError) as refused:
        session.start_go_to_step(protocol, 0)

    assert refused.value.reason == 'values_out_of_range'
    assert 'Step 1' in str(refused.value) and 'Exposure:' in str(refused.value)
    assert session.get_settings_snapshot() == before
    assert session.scope.motion.get_target_positions()['X'] == position
