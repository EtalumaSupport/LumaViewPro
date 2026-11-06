# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An Update of a z-stack slice is its channel's in the stack, and its place is the stack's.

A z-stack of a channel is one acquisition at several heights, so its slices
share one light and camera: Update on any slice gives every slice of that
channel in the stack the new settings, and the other channels keep theirs.
A run visits a stack as one place, so the X, Y and objective Update writes
are every slice's, every channel; each slice keeps its own Z and label.

Every Update of a stacked slice made since steps recorded their Add was
refused as a duplicate channel, because the slices of a channel share their
Add and their channel; a stack from a file before that column was not
refused, and its Update left one slice apart from its stack.
"""

from __future__ import annotations

import pytest

import _lumascope.modules.config_helpers as config_helpers
from _lumascope.modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings
from tests.stock_channels import LED0

CHANNEL_COLUMNS = ('Exposure', 'Gain', 'Illumination')
PLACE_COLUMNS = ('X', 'Y', 'Objective', 'Turret Slot')


@pytest.fixture
def session():
    settings = complete_settings(objective_id='10x Oly')
    settings['protocol']['labware'] = 'Center Plate'
    session = ScopeSession.create(settings, simulate=True)
    try:
        home_sim_scope(session.scope)
        for layer in config_helpers.get_channel_configs(session.settings):
            session.settings[layer]['acquire'] = None
        for layer, focus in (('BF', 5000.0), (LED0, 5020.0)):
            session.set_channel_acquire(layer, 'image')
            session.save_channel_focus(layer, focus)
        session.update_settings('zstack.step_size', 50.0)
        session.update_settings('zstack.position', 'center')
        session.update_settings('zstack.range', 100.0)
        yield session
    finally:
        session.shutdown()


def _slices(protocol, color):
    steps = protocol.steps().reset_index(drop=True)
    return steps[steps['Color'] == color]


def _stack(session):
    protocol = session.new_protocol(use_zstacking=True)
    assert len(_slices(protocol, 'BF')) == 3 and len(_slices(protocol, LED0)) == 3
    return protocol


def _update_a_middle_bf_slice(session, protocol, *, nudge_um=0.0):
    edited = int(_slices(protocol, 'BF').index[1])
    session.go_to_step(protocol, edited)
    if nudge_um:
        session.scope.motion.move_xy_by(dx_um=nudge_um, dy_um=nudge_um)
    session.update_settings('BF.exposure_ms', 7.0)
    session.update_settings('BF.gain_db', 3.0)
    session.update_settings('BF.illumination_ma', 40.0)
    session.update_step(protocol, edited, channel='BF')
    return edited


def _without_channel_groups(session, protocol, tmp_path):
    """``protocol`` as a file saved before steps recorded their Add reads it."""
    path = tmp_path / 'before_adds.tsv'
    protocol.to_file(path)
    lines = path.read_text().splitlines()
    header = lines.index('Steps') + 1
    drop = lines[header].split('\t').index('Channel Group ID')
    rewritten = []
    for i, line in enumerate(lines):
        if line.startswith('Version'):
            line = 'Version\t8'
        elif i >= header and line:
            cells = line.split('\t')
            del cells[drop]
            line = '\t'.join(cells)
        rewritten.append(line)
    path.write_text('\n'.join(rewritten) + '\n')
    loaded = session.load_protocol(path)
    assert set(loaded.steps()['Channel Group ID']) == {-1}
    return loaded


@pytest.mark.parametrize('recorded_adds', [True, False], ids=['since_adds', 'before_adds'])
def test_an_update_of_a_slice_gives_its_channel_in_the_stack_the_new_settings(
    session, tmp_path, recorded_adds
):
    # parametrize-justified: two file eras, each its own path to the edit (the
    # duplicate check runs on one and is skipped on the other).
    protocol = _stack(session)
    if not recorded_adds:
        protocol = _without_channel_groups(session, protocol, tmp_path)
    led0_before = _slices(protocol, LED0)
    bf_before = _slices(protocol, 'BF')

    _update_a_middle_bf_slice(session, protocol)

    bf = _slices(protocol, 'BF')
    assert list(bf['Exposure']) == [7.0] * 3
    assert list(bf['Gain']) == [3.0] * 3
    assert list(bf['Illumination']) == [40.0] * 3
    assert list(bf['Z']) == list(bf_before['Z'])
    assert list(bf['Name']) == list(bf_before['Name'])
    led0 = _slices(protocol, LED0)
    for column in (*CHANNEL_COLUMNS, 'Z'):
        assert list(led0[column]) == list(led0_before[column]), column


def test_an_update_of_a_slice_at_a_nudged_stage_moves_the_whole_stack(session):
    protocol = _stack(session)
    z_before = list(protocol.steps()['Z'])
    x_before = protocol.step_row(step_index=0)['X']

    _update_a_middle_bf_slice(session, protocol, nudge_um=500.0)

    steps = protocol.steps()
    for column in PLACE_COLUMNS:
        assert steps[column].nunique() == 1, column
    # A plate axis runs opposite its stage axis (`stage_to_plate`), so the move's size is the claim.
    assert abs(steps['X'].iloc[0] - x_before) == pytest.approx(0.5)
    assert list(steps['Z']) == z_before
