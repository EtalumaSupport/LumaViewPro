# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Save and the layer-settings restore are Session members, as a pair.

A protocol file carries a Layer Settings block: each acquiring layer's
acquire mode, illumination, gain, auto-gain, exposure, false colour, sum
and stimulation switch. The GUI composed the block from the store on Save
and put it back into the store on Load, so a script could neither save a
protocol the GUI would restore nor restore one the GUI had saved.
``ScopeSession.save_protocol`` writes the block and ``apply_layer_settings``
puts it back; ``load_protocol`` keeps its contract, the plate only.

The block is typed where it is read. A cell the loader cannot type is
refused at load, naming the file, like a step cell; a block with no
'Layer' column is refused rather than replaced by an inference from the
steps. A file with no block (v5) has its layer settings inferred from its
steps, typed the same way.
"""

import pytest

from modules.exceptions import ProtocolNotSavedError
from modules.protocol import ProtocolFormatError
from tests.scope_fakes import home_sim_scope
from tests.test_labware_name_resolution import _STEP_HEADER, _STEP_ROW, _protocol_file
from tests.test_loading_a_protocol_puts_the_scope_on_its_plate import (  # noqa: F401 -- pytest fixture
    session,
)

PLATE = '6 well microplate'
BLOCK_HEADER = (
    'Layer\tAcquire\tIllumination\tGain\tAuto_Gain\tExposure\tFalse_Color\tSum\tStim_Enabled\n'
)


def _file_with_block(tmp_path, *rows, header=BLOCK_HEADER):
    tsv = tmp_path / 'with_block.tsv'
    tsv.write_text(
        'LumaViewPro Protocol\n'
        'Version\t5\n'
        'Period\t30.0\n'
        'Duration\t24.0\n'
        f'Labware\t{PLATE}\n'
        'Capture Root\t\n'
        '\n'
        'Layer Settings\n' + header + ''.join(rows) + '\n'
        'Steps\n' + _STEP_HEADER + _STEP_ROW
    )
    return tsv


def _set_layer(session, layer, **values):
    with session.settings_lock:
        session.settings[layer].update(values)


def _layer_fields(session, layer):
    keys = (
        'acquire',
        'illumination_ma',
        'gain_db',
        'auto_gain',
        'exposure_ms',
        'false_color',
        'sum',
    )
    return {key: session.settings[layer][key] for key in keys}


def _only_bf_acquiring(session):
    for layer in ('BF', 'PC', 'DF', 'Red', 'Green', 'Blue', 'Lumi'):
        if layer in session.settings:
            _set_layer(session, layer, acquire=None)
    _set_layer(
        session,
        'BF',
        acquire='image',
        illumination_ma=37.0,
        gain_db=2.5,
        auto_gain=False,
        exposure_ms=12.0,
        false_color=False,
        sum=2,
    )


class TestSaveThenRestore:
    @pytest.fixture(autouse=True)
    def _homed(self, session):
        # New images an unfocused layer at the current Z, which needs a Z
        # the scope knows.
        home_sim_scope(session.scope)

    def test_a_saved_protocol_restores_its_steps_schedule_and_layer_settings(
        self, session, tmp_path
    ):
        _only_bf_acquiring(session)
        saved_bf = _layer_fields(session, 'BF')
        protocol = session.new_protocol()
        written = session.save_protocol(protocol, tmp_path / 'plate')

        _set_layer(session, 'BF', illumination_ma=99.0, sum=1)
        _set_layer(session, 'Green', acquire='image')
        loaded = session.load_protocol(written)
        session.apply_layer_settings(loaded)

        assert loaded.num_steps() == protocol.num_steps()
        assert (loaded.period(), loaded.duration()) == (protocol.period(), protocol.duration())
        assert _layer_fields(session, 'BF') == saved_bf
        assert session.settings['Green']['acquire'] is None, (
            'a layer the protocol does not name is cleared'
        )

    def test_the_load_alone_leaves_the_layer_controls_alone(self, session, tmp_path):
        _only_bf_acquiring(session)
        written = session.save_protocol(session.new_protocol(), tmp_path / 'plate')
        _set_layer(session, 'BF', illumination_ma=99.0)
        _set_layer(session, 'Green', acquire='image')

        session.load_protocol(written)

        assert session.settings['BF']['illumination_ma'] == 99.0
        assert session.settings['Green']['acquire'] == 'image'

    def test_save_adds_the_extension_and_returns_the_path_written(self, session, tmp_path):
        _only_bf_acquiring(session)

        written = session.save_protocol(session.new_protocol(), tmp_path / 'plate')

        assert written == tmp_path / 'plate.tsv'
        assert written.exists()

    def test_save_does_not_change_what_the_gui_opens_at_start_up(self, session, tmp_path):
        _only_bf_acquiring(session)
        remembered = session.settings['protocol']['filepath']

        session.save_protocol(session.new_protocol(), tmp_path / 'scratch')

        assert session.settings['protocol']['filepath'] == remembered

    def test_a_failed_write_raises(self, session, tmp_path):
        _only_bf_acquiring(session)

        with pytest.raises(ProtocolNotSavedError):
            session.save_protocol(session.new_protocol(), tmp_path / 'no such folder' / 'plate')

    def test_a_layer_this_scope_does_not_have_is_logged_and_dropped(
        self, session, tmp_path, monkeypatch
    ):
        import modules.scope_session as scope_session

        warned = []
        monkeypatch.setattr(
            scope_session.logger, 'warning', lambda msg, *a, **k: warned.append(msg)
        )
        tsv = _file_with_block(
            tmp_path,
            'BF\timage\t5.0\t0.0\tFalse\t2.0\tFalse\t1\t\n',
            'Infrared\timage\t5.0\t0.0\tFalse\t2.0\tFalse\t1\t\n',
        )

        session.apply_layer_settings(session.load_protocol(tsv))

        assert 'Infrared' not in session.settings
        assert session.settings['BF']['acquire'] == 'image'
        assert [m for m in warned if 'Infrared' in m], 'the dropped layer is named in the log'


class TestTheBlockIsTypedAtLoad:
    def test_a_saved_block_comes_back_typed(self, session, tmp_path):
        tsv = _file_with_block(tmp_path, 'Red\timage\t350.0\t20.0\tTrue\t200.0\tTrue\t1\tFalse\n')

        row = session.load_protocol(tsv).layer_settings()['Red']

        assert row == {
            'Layer': 'Red',
            'Acquire': 'image',
            'Illumination': 350.0,
            'Gain': 20.0,
            'Auto_Gain': True,
            'Exposure': 200.0,
            'False_Color': True,
            'Sum': 1,
            'Stim_Enabled': False,
        }

    def test_a_blank_cell_is_none(self, session, tmp_path):
        tsv = _file_with_block(tmp_path, 'BF\timage\t5.0\t0.0\tFalse\t2.0\tFalse\t1\t\n')

        assert session.load_protocol(tsv).layer_settings()['BF']['Stim_Enabled'] is None

    @pytest.mark.parametrize(
        'row',
        [
            'BF\timage\tbright\t0.0\tFalse\t2.0\tFalse\t1\t\n',
            'BF\timage\t5.0\t0.0\tmaybe\t2.0\tFalse\t1\t\n',
            'BF\timage\t5.0\t0.0\tFalse\t2.0\tFalse\t1.5\t\n',
            'BF\tstill\t5.0\t0.0\tFalse\t2.0\tFalse\t1\t\n',
        ],
        ids=['illumination', 'auto_gain', 'sum', 'acquire'],
    )
    def test_a_cell_the_loader_cannot_type_is_refused_naming_the_file(self, session, tmp_path, row):
        tsv = _file_with_block(tmp_path, row)

        with pytest.raises(ProtocolFormatError) as refused:
            session.load_protocol(tsv)

        assert refused.value.file == tsv

    def test_a_block_with_no_layer_column_is_refused_not_inferred(self, session, tmp_path):
        tsv = _file_with_block(
            tmp_path,
            'BF\timage\t5.0\t0.0\tFalse\t2.0\tFalse\t1\t\n',
            header=BLOCK_HEADER.replace('Layer\t', 'Name\t'),
        )

        with pytest.raises(ProtocolFormatError):
            session.load_protocol(tsv)

    def test_a_block_with_no_rows_is_refused(self, session, tmp_path):
        with pytest.raises(ProtocolFormatError):
            session.load_protocol(_file_with_block(tmp_path))

    def test_a_row_that_names_no_layer_is_refused(self, session, tmp_path):
        tsv = _file_with_block(tmp_path, '\timage\t5.0\t0.0\tFalse\t2.0\tFalse\t1\t\n')

        with pytest.raises(ProtocolFormatError):
            session.load_protocol(tsv)

    def test_a_file_with_no_block_is_inferred_typed(self, session, tmp_path):
        row = session.load_protocol(_protocol_file(tmp_path, PLATE)).layer_settings()['BF']

        assert row == {
            'Layer': 'BF',
            'Acquire': 'image',
            'Illumination': 100.0,
            'Gain': 0.0,
            'Auto_Gain': False,
            'Exposure': 40.0,
            'False_Color': False,
            'Sum': 1,
            'Stim_Enabled': None,
        }


def _v5_file_whose_step_has(tmp_path, column, value):
    columns = _STEP_HEADER.rstrip('\n').split('\t')
    row = _STEP_ROW.rstrip('\n').split('\t')
    row[columns.index(column)] = value
    tsv = tmp_path / 'v5_bad_step.tsv'
    tsv.write_text(
        'LumaViewPro Protocol\n'
        'Version\t5\n'
        'Period\t30.0\n'
        'Duration\t24.0\n'
        f'Labware\t{PLATE}\n'
        '\n'
        'Steps\n' + _STEP_HEADER + '\t'.join(row) + '\n'
    )
    return tsv


class TestAFileWithNoBlockAndABadStep:
    """A step cell of the wrong type refuses the file, with a block or
    without: no layer row is inferred from a cell nothing could read."""

    @pytest.mark.parametrize(
        'column, value',
        [
            ('Illumination', 'abc'),
            ('Gain', 'abc'),
            ('Exposure', 'abc'),
            ('Sum', '1.5'),
            ('Auto_Gain', 'maybe'),
            ('False_Color', 'maybe'),
            ('Acquire', 'Image'),
        ],
    )
    def test_it_is_refused_naming_the_step_and_the_column(self, session, tmp_path, column, value):
        with pytest.raises(ProtocolFormatError) as refused:
            session.load_protocol(_v5_file_whose_step_has(tmp_path, column, value))

        assert 'step 1' in str(refused.value)
        assert column in str(refused.value)
