# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An old protocol, loaded on a scope, runs the LEDs it was made with.

A protocol written before the fluorescence keys were named for their
excitation holds ``Blue`` / ``Green`` / ``Red`` in its ``Color`` cells, its
``Stim_Config`` keys and its ``Layer Settings`` rows. Loaded on a scope, the
unit is known, so each old name is renamed to the unit's key for its LED
(``ChannelIdentity.unit_key``): on a stock unit ``Ex405`` / ``Ex488`` /
``Ex589``, on a unit whose LEDs 0 / 1 / 2 are 488 / 550 / 640 its own
``Ex488`` / ``Ex550`` / ``Ex640``, and on a unit with no LED there the stock
key, which the scope's refusals then name. A name nothing knows is refused
as unknown, not as missing hardware. The run refuses an enabled
stimulation on a channel the unit lacks.

Each test runs on the shipped release, whose keys are named for their
excitation.
"""

from __future__ import annotations

import csv
import io
import json
import pathlib
import threading
import types

import pytest

from modules.exceptions import ProtocolRunRefusedError
from modules.path_utils import read_installation_file
from modules.protocol import Protocol, ProtocolFormatError
from modules.run_events import RunEvents
from modules.scope_session import ScopeSession
from tests.installation_fixtures import SHIPPED_DATA, non_stock_block, write_installation
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings
from tests.test_a_run_needs_every_axis_position import COMPLETION_TIMEOUT
from tests.test_going_to_a_step_is_the_sessions_move import _protocol, _step

# The old names, the stock keys they read as, in LED order.
_OLD = ('Blue', 'Green', 'Red')
_STOCK = ('Ex405', 'Ex488', 'Ex589')
_PLATE = '96 well microplate'
# Each stock key as an earlier release wrote it.
_AS_EARLIER = types.MappingProxyType(dict(zip(_STOCK, _OLD, strict=True)))
# The channels an earlier release knew: its stimulation cells name these alone.
_EARLIER_RELEASE = {'BF', 'PC', 'DF', *_STOCK, 'Lumi'}


def _transmitted_rows() -> list[dict]:
    """The shipped LS850's transmitted rows, which a unit's own block carries too."""
    rows = read_installation_file(SHIPPED_DATA / 'scopes.json')['Models']['LS850']['Layers']
    return [row for row in rows if row['key_name'] in ('BF', 'PC', 'DF')]


def _session(root: pathlib.Path, monkeypatch, *, microscope: str, non_stock: bool = False):
    board_block = non_stock_block(_transmitted_rows()) if non_stock else None
    write_installation(root, board_block=board_block)
    settings = complete_settings(
        microscope=microscope,
        live_folder=str(root / 'live'),
        stage_offset={'x': 0.0, 'y': 0.0},
        objective_id='10x Oly',
    )
    session = ScopeSession.create(settings, source_path=str(root), simulate=True)
    if session.scope.motor_connected:
        home_sim_scope(session.scope)
    return session


def _old_file(
    session,
    path: pathlib.Path,
    colors,
    *,
    rename=_AS_EARLIER,
    stimulated=(),
    added_stimulation=None,
):
    """A protocol on ``colors`` as an earlier release saved it.

    Written by ``session`` (whose release is in excitation keys) with a
    step and a Layer Settings row per colour, each step carrying a
    stimulation entry per channel that release knew, disabled but for
    ``stimulated``; then put in the earlier release's words: version 6
    (no Label and no Auto_Named, both recovered from the Name), each key in
    ``rename`` written as its old name in the Color, Name, Stim_Config and
    Layer Settings cells. ``added_stimulation`` entries are written into each
    step's Stim_Config as they are, after the rename.
    """
    objective = session.scope.runtime_state.get_current_objective_id()
    stimulation = {
        key: {**session.settings[key]['stim_config'], 'enabled': key in stimulated}
        for key in (*_EARLIER_RELEASE, *stimulated)
        if session.settings[key].get('stim_config') is not None
    }
    steps = [
        _step(60.0, 40.0, 5000.0, objective)
        | {'Name': f'A1_{color}', 'Color': color, 'Stim_Config': stimulation}
        for color in colors
    ]
    for color in colors:
        session.set_channel_acquire(color, 'image')
    saved = session.save_protocol(_protocol(_PLATE, *steps), path)

    rows = list(csv.reader(io.StringIO(saved.read_text()), delimiter='\t'))
    header = next(row for row in rows if row[:1] == ['Name'])
    steps_from = rows.index(header)
    kept = [column for column in header if column in Protocol.COLUMNS[6]]
    old_rows = []
    for at, row in enumerate(rows):
        if row[:1] == ['Version']:
            row = ['Version', '6']
        elif row[:1] and row[0] in rename:
            row = [rename[row[0]], *row[1:]]
        elif at > steps_from and row:
            cells = dict(zip(header, row, strict=True))
            stim = {rename.get(k, k): v for k, v in json.loads(cells['Stim_Config']).items()}
            cells['Stim_Config'] = json.dumps(stim | (added_stimulation or {}))
            cells['Color'] = rename.get(cells['Color'], cells['Color'])
            cells['Name'] = f'A1_{cells["Color"]}'
            row = [cells[column] for column in kept]
        elif at == steps_from:
            row = kept
        old_rows.append(row)
    out = io.StringIO()
    csv.writer(out, delimiter='\t', lineterminator='\n').writerows(old_rows)
    saved.write_text(out.getvalue())
    return saved


def _lit_by_going_to_each_step(session, protocol) -> list[int]:
    """The LED address each step lights when gone to with the preview on."""
    session.settings['protocol_led_on'] = True
    lit = []
    for index in range(protocol.num_steps()):
        session.go_to_step(protocol, index)
        lit.extend(
            r.address for r in session.scope.illumination.get_led_states() if r.state.name == 'LIT'
        )
    return lit


# parametrize-justified: each unit is its own installation and bring-up.
@pytest.mark.parametrize(
    ('microscope', 'non_stock', 'keys'),
    [
        pytest.param('LS850', False, ('Ex405', 'Ex488', 'Ex589'), id='stock'),
        pytest.param('LS850', True, ('Ex488', 'Ex550', 'Ex640'), id='non-stock'),
    ],
)
def test_an_old_protocol_loads_in_the_units_keys_and_lights_its_leds(
    tmp_path, monkeypatch, microscope, non_stock, keys
):
    """Steps, stimulation, Layer Settings and Names are in the unit's keys,
    and going to each step lights LED 0, 1 and 2, as the old file did."""
    maker = _session(tmp_path / 'maker', monkeypatch, microscope='LS850')
    try:
        old = _old_file(maker, tmp_path / 'old.tsv', _STOCK)
    finally:
        maker.shutdown()
    assert 'Blue' in old.read_text() and 'Ex405' not in old.read_text()
    session = _session(tmp_path / 'unit', monkeypatch, microscope=microscope, non_stock=non_stock)
    try:
        protocol = session.load_protocol(str(old))
        steps = protocol.steps()
        colors = list(steps['Color'])
        stim_keys = {k for cell in steps['Stim_Config'] for k in cell}
        settings_keys = set(protocol.channel_settings())
        names = list(steps['Name'])
        lit = _lit_by_going_to_each_step(session, protocol)
    finally:
        session.shutdown()

    assert colors == list(keys)
    assert stim_keys & {*_OLD, *_STOCK, 'Ex550', 'Ex640'} == set(keys)
    assert settings_keys == set(keys)
    assert not any(name in n for n in names for name in _OLD), names
    assert all(key in n for key, n in zip(keys, names, strict=True)), names
    assert lit == [0, 1, 2]


def test_on_a_unit_without_the_led_an_old_name_is_its_stock_key(tmp_path, monkeypatch):
    """On an LS560, which has LED 1 alone, an old ``Green`` step loads as its
    ``Ex488``, keeping the disabled ``Blue`` and ``Red`` stimulation as the
    stock ``Ex405`` and ``Ex589``, and goes to its step on LED 1; an old
    ``Blue`` step is refused as a layer the scope lacks; a ``Bleu`` step as a
    channel nothing knows."""
    maker = _session(tmp_path / 'maker', monkeypatch, microscope='LS850')
    try:
        green = _old_file(maker, tmp_path / 'green.tsv', ('Ex488',))
        blue = _old_file(maker, tmp_path / 'blue.tsv', ('Ex405',))
        bleu = _old_file(maker, tmp_path / 'bleu.tsv', ('Ex405',), rename={'Ex405': 'Bleu'})
    finally:
        maker.shutdown()
    session = _session(tmp_path / 'unit', monkeypatch, microscope='LS560')
    try:
        protocol = session.load_protocol(str(green))
        with pytest.raises(ProtocolRunRefusedError) as absent:
            session.load_protocol(str(blue))
        with pytest.raises(ProtocolRunRefusedError) as unknown:
            session.load_protocol(str(bleu))
        step = protocol.steps().iloc[0]
        lit = _lit_by_going_to_each_step(session, protocol)
    finally:
        session.shutdown()

    assert step['Color'] == 'Ex488'
    assert {k for k, v in step['Stim_Config'].items() if not v['enabled']} >= {'Ex405', 'Ex589'}
    assert lit == [1]
    assert absent.value.reason == 'layer_not_on_scope'
    assert (unknown.value.reason, 'Bleu' in str(unknown.value)) == ('channel_unknown', True)


def test_a_file_naming_one_channel_twice_is_refused(tmp_path, monkeypatch):
    """A hand-edited file with a stimulation entry under both ``Blue`` and
    ``Ex405`` holds two values for one LED; the load refuses it, naming
    both, rather than keep either."""
    session = _session(tmp_path / 'unit', monkeypatch, microscope='LS850')
    try:
        old = _old_file(
            session,
            tmp_path / 'old.tsv',
            ('Ex405',),
            added_stimulation={'Ex405': session.settings['Ex405']['stim_config']},
        )
        with pytest.raises(ProtocolFormatError) as refused:
            session.load_protocol(str(old))
    finally:
        session.shutdown()

    assert 'Blue' in str(refused.value) and 'Ex405' in str(refused.value)


def _run(session, protocol, tmp_path) -> threading.Event:
    done = threading.Event()
    session.create_protocol_runner().start_single_scan(
        protocol=protocol,
        sequence_name='stimulation',
        parent_dir=str(tmp_path / 'runs'),
        events=RunEvents(files_written=lambda run_dir, files: done.set()),
    )
    return done


def test_an_enabled_stimulation_on_a_channel_the_unit_lacks_refuses_the_run(tmp_path, monkeypatch):
    """A stock unit lacks ``Ex550``: a step stimulating it loads, and is
    refused when the run is prepared, as a layer the scope lacks; the same
    step with that stimulation disabled runs."""
    session = _session(tmp_path / 'unit', monkeypatch, microscope='LS850')
    try:
        stimulating = session.load_protocol(
            str(
                _old_file(
                    session, tmp_path / 'on.tsv', ('Ex405',), rename={}, stimulated=('Ex550',)
                )
            )
        )
        disabled = session.load_protocol(
            str(_old_file(session, tmp_path / 'off.tsv', ('Ex405',), rename={}))
        )
        with pytest.raises(ProtocolRunRefusedError) as refused:
            _run(session, stimulating, tmp_path)
        ran = _run(session, disabled, tmp_path).wait(COMPLETION_TIMEOUT)
    finally:
        session.shutdown()

    assert (refused.value.reason, 'Ex550' in str(refused.value)) == ('layer_not_on_scope', True)
    assert ran, 'the run with the stimulation disabled never wrote its files'


def test_a_stimulation_entry_that_is_not_a_config_refuses_the_run_naming_it(tmp_path, monkeypatch):
    """A hand-edited Stim_Config entry holding a bare number is no
    stimulation the run can judge: the run is refused, naming the step and
    the entry, before anything starts."""
    session = _session(tmp_path / 'unit', monkeypatch, microscope='LS850')
    try:
        protocol = session.load_protocol(
            str(
                _old_file(
                    session,
                    tmp_path / 'bare.tsv',
                    ('Ex405',),
                    rename={},
                    added_stimulation={'Ex550': 5},
                )
            )
        )
        with pytest.raises(ProtocolFormatError) as refused:
            _run(session, protocol, tmp_path)
    finally:
        session.shutdown()

    assert 'Ex550' in str(refused.value) and 'not a config' in str(refused.value)
