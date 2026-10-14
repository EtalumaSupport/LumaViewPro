# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Resolver truth table for the layer identity record.

Pins the resolution contract: wholesale block-or-model (never a field
merge), motor-first model precedence with the configured model only
where hardware reports none, the explicit override, per-row loud skip
(scope survives), catalogue-positional ids, and the source tags that
make degraded states visible instead of silent.
"""

import json

import pytest

from lvp_logger import logger as _mock_logger

import modules.common_utils as common_utils
from modules.layer_record import (
    UNRESOLVED,
    LayerIdentity,
    LayerRecord,
    GREY,
    format_display_color,
    load_layer_catalogue,
    output_color,
    parse_display_color,
    release_color,
    resolve_layer_identity,
    shown_color,
)
from modules.exceptions import ArgumentRefusedError, InstallationFileError
from modules.path_utils import read_installation_file, resolve_data_file
from modules.layer_record import release_colors

# The shipped catalogue's records, as scopes.json states them.
CATALOGUE = read_installation_file(resolve_data_file('scopes.json'))['LayerOrder']


def row(key, display=None, channel=None, nm=None, **extra):
    d = {
        'key_name': key,
        'display_name': display if display is not None else key,
        'led_channel': channel,
        'excitation_nm': nm,
    }
    d.update(extra)
    return d


def write_scopes(tmp_path, models, catalogue=CATALOGUE):
    data = {'LayerOrder': catalogue, 'Models': models}
    path = tmp_path / 'scopes_fixture.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    return str(path)


LS850T_ROWS = [
    row('BF', 'BF', 3),
    row('PC', 'PC', 4),
    row('DF', 'DF', 5),
    row('Blue', 'Blue', 0, 405),
    row('Green', 'Green', 1, 488),
    row('Red', 'Red', 2, 589),
    row('Lumi', 'Lumi', None),
]

LS620_ROWS = [
    row('BF', 'BF-Phase', 3),
    row('Blue', 'Blue', 0, 405),
    row('Green', 'Green', 1, 488),
    row('Red', 'Red', 2, 589),
]

LS560_ROWS = [
    row('BF', 'PC-BF', 3),
    row('Green', 'Green', 1, 488),
]


def errors_logged():
    return ' | '.join(str(c) for c in _mock_logger.error.call_args_list)


def warnings_logged():
    return ' | '.join(str(c) for c in _mock_logger.warning.call_args_list)


REPORTED = []


def resolve(data_file, **kwargs):
    """The identity resolved from ``data_file``; what it reported is in ``REPORTED``."""
    _mock_logger.reset_mock()
    REPORTED.clear()
    defaults = {
        'board_block': None,
        'board_config_read_ok': True,
        'motor_model': None,
        'configured_model': None,
    }
    defaults.update(kwargs)
    data = read_installation_file(data_file)
    return resolve_layer_identity(
        models=data['Models'],
        catalogue=load_layer_catalogue(data, data_file),
        report=REPORTED.append,
        **defaults,
    )


def channels(rows):
    """A board's Channels block holding ``rows``, each keyed by its key name."""
    return {r['key_name']: {k: v for k, v in r.items() if k != 'key_name'} for r in rows}


class TestCatalogue:
    def test_ids_are_catalogue_positions(self, tmp_path):
        path = write_scopes(tmp_path, {'LS850T': {'Layers': LS850T_ROWS, 'Filterset': 'Stock'}})
        identity = resolve(path, motor_model='LS850T')
        assert [(r.key_name, r.id) for r in identity.layers] == [
            ('BF', 0),
            ('PC', 1),
            ('DF', 2),
            ('Blue', 3),
            ('Green', 4),
            ('Red', 5),
            ('Lumi', 6),
        ]

    def test_layers_sort_by_id_whatever_the_row_order(self, tmp_path):
        path = write_scopes(tmp_path, {'M': {'Layers': list(reversed(LS850T_ROWS))}})
        identity = resolve(path, motor_model='M')
        assert [r.id for r in identity.layers] == sorted(r.id for r in identity.layers)

    def test_missing_layer_order_refuses_naming_the_file(self, tmp_path):
        path = tmp_path / 'scopes_fixture.json'
        path.write_text(json.dumps({'LS850T': {'Layers': LS850T_ROWS}}), encoding='utf-8')
        with pytest.raises(InstallationFileError, match='LayerOrder') as refused:
            load_layer_catalogue(read_installation_file(path), path)
        assert refused.value.file_path == path

    def test_the_shipped_catalogue_answers_each_channel_set(self):
        """Every channel set the program uses, read from the shipped catalogue.

        Each set is what a consumer is handed (the composite's transmitted
        base, the false-colour layers, the LED layers, the layers a scale bar
        is drawn dark on), so the stock seven answer these whatever shape the
        catalogue's entries take. DF is transmitted with a dark background.
        """
        assert common_utils.get_layers() == ['BF', 'PC', 'DF', 'Blue', 'Green', 'Red', 'Lumi']
        assert common_utils.get_transmitted_layers() == ['BF', 'PC', 'DF']
        assert common_utils.get_fluorescence_layers() == ['Blue', 'Green', 'Red']
        assert common_utils.get_luminescence_layers() == ['Lumi']
        assert common_utils.get_bright_background_layers() == ['BF', 'PC']
        assert common_utils.get_image_layers() == ['Blue', 'Green', 'Red', 'Lumi']
        assert common_utils.get_layers_with_led() == ['BF', 'PC', 'DF', 'Blue', 'Green', 'Red']


def _refuses(colour):
    try:
        parse_display_color(colour)
    except ValueError:
        return True
    return False


class TestCatalogueRecords:
    """Each catalogue record is whole, or the file is refused naming it."""

    @staticmethod
    def _refused(tmp_path, catalogue):
        path = tmp_path / 'scopes_fixture.json'
        path.write_text(json.dumps({'LayerOrder': catalogue, 'Models': {}}), encoding='utf-8')
        with pytest.raises(InstallationFileError) as refused:
            load_layer_catalogue(read_installation_file(path), path)
        assert refused.value.file_path == path
        return str(refused.value)

    def test_a_record_missing_its_modality_is_refused(self, tmp_path):
        bf = {k: v for k, v in CATALOGUE[0].items() if k != 'modality'}
        assert "missing ['modality']" in self._refused(tmp_path, [bf])

    def test_a_record_of_the_wrong_kind_is_refused_naming_the_field(self, tmp_path):
        cases = {
            'modality': {'modality': 'phase'},
            'background': {'background': 'grey'},
            'manual_max_exposure_ms': {'manual_max_exposure_ms': 0},
            'display_color': {'display_color': [1, 0, 1]},
            'step_order': {'step_order': 1.5},
        }
        wrong = {
            field: self._refused(tmp_path, [{**CATALOGUE[0], **change}])
            for field, change in cases.items()
        }
        assert [field for field, why in wrong.items() if field not in why] == []

    def test_a_colour_is_its_red_green_and_blue_weights(self):
        assert parse_display_color('#ff8000') == (1.0, 128 / 255, 0.0)
        refused = [c for c in ('#FF8000', 'ff8000', '#ff800', [1, 0, 1]) if _refuses(c)]
        assert refused == ['#FF8000', 'ff8000', '#ff800', [1, 0, 1]]

    def test_a_colour_not_at_full_brightness_is_refused(self):
        # A baked plane is read back by its brightest plane, which holds the
        # mono whole only when that component is full.
        refused = [c for c in ('#800080', '#7f0001', '#000000', '#fe0000') if _refuses(c)]
        assert refused == ['#800080', '#7f0001', '#000000', '#fe0000']

    def test_a_colour_is_written_as_the_text_it_was_read_from(self):
        texts = ('#ff8000', '#ffffff', '#00ff00', '#0080ff', '#ff00ff')
        assert [format_display_color(parse_display_color(t)) for t in texts] == list(texts)

    def test_a_repeated_key_or_step_order_is_refused(self, tmp_path):
        assert 'key' in self._refused(tmp_path, [CATALOGUE[0], CATALOGUE[0]])
        moved = {**CATALOGUE[1], 'step_order': CATALOGUE[0]['step_order']}
        assert 'step_order' in self._refused(tmp_path, [CATALOGUE[0], moved])

    def test_a_name_is_no_longer_a_record(self, tmp_path):
        assert 'not an object' in self._refused(tmp_path, ['BF', 'PC'])


class TestTheColourRead:
    """A layer's colour, the colour an image named for it is written in, and
    the colour it is shown in, each read from the release catalogue."""

    def test_a_layer_reads_its_release_colour(self):
        assert release_color('Green') == (0.0, 1.0, 0.0)
        assert release_color('BF') == GREY

    def test_a_composite_is_written_grey_and_a_layer_in_its_colour(self):
        assert output_color('Composite', colors=release_colors()) == GREY
        assert output_color('Red', colors=release_colors()) == (1.0, 0.0, 0.0)

    def test_false_colour_off_shows_grey(self):
        assert shown_color('Blue', True, colors=release_colors()) == (0.0, 0.0, 1.0)
        assert shown_color('Blue', False, colors=release_colors()) == GREY

    def test_a_name_that_is_no_layer_has_no_colour(self):
        refused = []
        for read in (
            release_color,
            lambda name: output_color(name, colors=release_colors()),
            lambda name: shown_color(name, False, colors=release_colors()),
        ):
            with pytest.raises(ArgumentRefusedError) as raised:
                read('FarRed')
            refused.append(raised.value.reason)
        assert refused == ['layer_unknown'] * 3


class TestPrecedence:
    def test_block_present_wins_wholesale(self, tmp_path):
        path = write_scopes(tmp_path, {'LS850T': {'Layers': LS850T_ROWS, 'Filterset': 'Stock'}})
        block = {'Channels': channels(LS560_ROWS), 'Filterset': 'Custom-X'}
        identity = resolve(path, board_block=block, motor_model='LS850T')
        assert identity.source == 'motorconfig'
        assert identity.filterset == 'Custom-X'
        assert [r.key_name for r in identity.layers] == ['BF', 'Green']
        assert identity.find('Red') is None

    def test_no_block_resolves_the_motor_model(self, tmp_path):
        path = write_scopes(tmp_path, {'LS850T': {'Layers': LS850T_ROWS, 'Filterset': 'Stock'}})
        identity = resolve(path, motor_model='LS850T')
        assert identity.source == 'scopes'
        assert identity.filterset == 'Stock'
        assert len(identity.layers) == 7

    def test_motor_model_outranks_configured(self, tmp_path):
        path = write_scopes(
            tmp_path,
            {'LS850T': {'Layers': LS850T_ROWS}, 'LS560': {'Layers': LS560_ROWS}},
        )
        identity = resolve(path, motor_model='LS850T', configured_model='LS560')
        assert len(identity.layers) == 7

    def test_configured_model_serves_hardware_that_reports_none(self, tmp_path):
        path = write_scopes(tmp_path, {'LS560': {'Layers': LS560_ROWS}})
        identity = resolve(path, configured_model='LS560')
        assert identity.source == 'scopes'
        assert [r.key_name for r in identity.layers] == ['BF', 'Green']

    def test_motor_model_without_entry_goes_unresolved_not_configured(self, tmp_path):
        path = write_scopes(tmp_path, {'LS560': {'Layers': LS560_ROWS}})
        identity = resolve(path, motor_model='LS9999', configured_model='LS560')
        assert identity.source == 'unresolved'
        assert identity.layers == ()
        assert identity.model == 'LS9999'
        assert 'LS9999' in errors_logged()

    def test_nothing_resolvable_is_the_unresolved_snapshot(self, tmp_path):
        path = write_scopes(tmp_path, {})
        identity = resolve(path)
        assert identity == UNRESOLVED
        assert identity.layers == ()
        assert identity.source == 'unresolved'

    def test_failed_board_read_is_loud_then_resolves_the_model(self, tmp_path):
        path = write_scopes(tmp_path, {'LS850T': {'Layers': LS850T_ROWS}})
        identity = resolve(path, board_config_read_ok=False, motor_model='LS850T')
        assert identity.source == 'scopes'
        assert len(identity.layers) == 7
        assert 'could not be read' in errors_logged()

    def test_absent_block_with_clean_read_is_quiet(self, tmp_path):
        path = write_scopes(tmp_path, {'LS850T': {'Layers': LS850T_ROWS}})
        resolve(path, motor_model='LS850T')
        assert 'could not be read' not in errors_logged()


class TestOverride:
    def test_override_beats_block_and_motor_model(self, tmp_path):
        path = write_scopes(
            tmp_path,
            {'LS850T': {'Layers': LS850T_ROWS}, 'LS560': {'Layers': LS560_ROWS}},
        )
        block = {'Channels': channels(LS850T_ROWS), 'Filterset': 'Stock'}
        identity = resolve(path, board_block=block, motor_model='LS850T', override_model='LS560')
        assert [r.key_name for r in identity.layers] == ['BF', 'Green']
        assert 'override' in warnings_logged()

    def test_override_of_unknown_model_is_loud_and_unresolved(self, tmp_path):
        path = write_scopes(tmp_path, {'LS850T': {'Layers': LS850T_ROWS}})
        identity = resolve(path, motor_model='LS850T', override_model='LS9999')
        assert identity.source == 'unresolved'
        assert identity.layers == ()
        assert identity.model == 'LS9999'
        assert 'LS9999' in errors_logged()


class TestRowParsing:
    def test_multi_channel_row_is_skipped_loudly_and_scope_survives(self, tmp_path):
        rows = [row('BF', 'BF', 3), dict(row('Green', 'Green'), led_channel=[1, 2])]
        path = write_scopes(tmp_path, {'M': {'Layers': rows}})
        identity = resolve(path, motor_model='M')
        assert [r.key_name for r in identity.layers] == ['BF']
        assert 'multiple' in errors_logged()

    def test_uncatalogued_key_is_skipped_loudly(self, tmp_path):
        rows = [row('BF', 'BF', 3), row('Skylight', 'Skylight', 6)]
        path = write_scopes(tmp_path, {'M': {'Layers': rows}})
        identity = resolve(path, motor_model='M')
        assert [r.key_name for r in identity.layers] == ['BF']
        assert 'Skylight' in errors_logged()

    def test_a_key_that_is_not_a_name_is_skipped_loudly(self, tmp_path):
        rows = [row('BF', 'BF', 3), row(['Green'], 'Green', 1, 488)]
        path = write_scopes(tmp_path, {'M': {'Layers': rows}})
        identity = resolve(path, motor_model='M')
        assert [r.key_name for r in identity.layers] == ['BF']
        assert "['Green']" in errors_logged()

    def test_missing_field_is_skipped_loudly(self, tmp_path):
        rows = [row('BF', 'BF', 3), {'key_name': 'Green', 'display_name': 'Green'}]
        path = write_scopes(tmp_path, {'M': {'Layers': rows}})
        identity = resolve(path, motor_model='M')
        assert [r.key_name for r in identity.layers] == ['BF']
        assert 'missing' in errors_logged()

    def test_every_row_unusable_is_a_zero_layer_identity_not_a_fallback(self, tmp_path):
        path = write_scopes(tmp_path, {'LS850T': {'Layers': LS850T_ROWS, 'Filterset': 'Stock'}})
        block = {
            'Channels': channels([dict(row('BF', 'BF'), led_channel=[1, 2])]),
            'Filterset': 'Broken',
        }
        identity = resolve(path, board_block=block, motor_model='LS850T')
        assert identity.source == 'motorconfig'
        assert identity.layers == ()
        assert identity.filterset == 'Broken'

    def test_extra_fields_are_tolerated(self, tmp_path):
        rows = [row('BF', 'BF', 3, None, future_field='anything')]
        path = write_scopes(tmp_path, {'M': {'Layers': rows}})
        identity = resolve(path, motor_model='M')
        assert identity.find('BF') is not None

    def test_ledless_layer_is_a_valid_record(self, tmp_path):
        path = write_scopes(tmp_path, {'M': {'Layers': [row('Lumi', 'Lumi', None)]}})
        identity = resolve(path, motor_model='M')
        lumi = identity.find('Lumi')
        assert lumi is not None
        assert lumi.led_channel == ()
        assert lumi.excitation_nm is None

    def test_a_fluorescence_row_with_no_excitation_is_skipped_loudly(self, tmp_path):
        rows = [row('BF', 'BF', 3), row('Green', 'Green', 1, None)]
        path = write_scopes(tmp_path, {'M': {'Layers': rows}})
        identity = resolve(path, motor_model='M')
        assert [r.key_name for r in identity.layers] == ['BF']
        assert 'no excitation_nm' in errors_logged()

    def test_a_fluorescence_layer_is_labelled_by_its_excitation(self, tmp_path):
        path = write_scopes(tmp_path, {'LS620': {'Layers': LS620_ROWS}})
        identity = resolve(path, motor_model='LS620')
        assert [r.label for r in identity.layers] == ['BF-Phase', '405 nm', '488 nm', '589 nm']

    def test_a_record_carries_its_catalogue_fields(self, tmp_path):
        path = write_scopes(tmp_path, {'LS850T': {'Layers': LS850T_ROWS}})
        identity = resolve(path, motor_model='LS850T')
        assert [
            (r.key_name, r.modality, r.background, r.display_color) for r in identity.layers
        ] == [
            (layer['key'], layer['modality'], layer['background'], layer['display_color'])
            for layer in CATALOGUE
        ]

    def test_single_channel_and_excitation_normalise(self, tmp_path):
        path = write_scopes(tmp_path, {'M': {'Layers': [row('Green', 'Green', 1, 488)]}})
        green = resolve(path, motor_model='M').find('Green')
        assert green.led_channel == (1,)
        assert green.excitation_nm == pytest.approx(488.0)
        assert isinstance(green.excitation_nm, float)


class TestTheBoardsChannelBlock:
    """A board's Channels block is the unit's channels, read as the model's rows are."""

    def test_a_fluorescence_channel_is_its_excitation_and_any_other_its_name(self, tmp_path):
        path = write_scopes(tmp_path, {'LS850T': {'Layers': LS850T_ROWS, 'Filterset': 'Stock'}})
        block = {
            'Channels': {
                'BF': {'led_channel': 3, 'display_name': 'Brightfield'},
                'Blue': {'led_channel': 6, 'excitation_nm': 445},
            },
            'Filterset': 'Custom-445',
        }
        identity = resolve(path, board_block=block, motor_model='LS850T')
        assert (identity.source, identity.filterset) == ('motorconfig', 'Custom-445')
        assert [(r.key_name, r.label, r.display_name, r.led_channel) for r in identity.layers] == [
            ('BF', 'Brightfield', 'Brightfield', (3,)),
            ('Blue', '445 nm', None, (6,)),
        ]
        assert REPORTED == []

    def test_a_rows_colour_is_the_units_and_without_one_the_releases(self, tmp_path):
        path = write_scopes(tmp_path, {'LS850T': {'Layers': LS850T_ROWS}})
        block = {
            'Channels': {
                'Blue': {'led_channel': 0, 'excitation_nm': 405, 'display_color': '#00ffff'},
                'Green': {'led_channel': 1, 'excitation_nm': 488},
            }
        }
        identity = resolve(path, board_block=block, motor_model='LS850T')
        assert identity.find('Blue').display_color == '#00ffff'
        assert identity.find('Green').display_color == format_display_color(release_color('Green'))

    def test_the_rows_it_cannot_read_are_left_out_and_reported_in_one_notice(self, tmp_path):
        path = write_scopes(tmp_path, {'LS850T': {'Layers': LS850T_ROWS}})
        block = {
            'Channels': {
                'BF': {'led_channel': 3, 'display_name': 'BF'},
                'Skylight': {'led_channel': 7, 'display_name': 'Skylight'},
                'Green': {'led_channel': 1},
                'Red': {'led_channel': 2, 'excitation_nm': 589, 'display_color': '#800000'},
                'PC': {'led_channel': 4},
            }
        }
        identity = resolve(path, board_block=block, motor_model='LS850T')
        assert [r.key_name for r in identity.layers] == ['BF']
        assert [n.reason for n in REPORTED] == ['channel_rows_skipped']
        assert set(REPORTED[0].skipped) == {'Skylight', 'Green', 'Red', 'PC'}

    def test_two_channels_on_one_led_address_leave_the_unit_no_channels(self, tmp_path):
        path = write_scopes(tmp_path, {'LS850T': {'Layers': LS850T_ROWS}})
        block = {
            'Channels': {
                'BF': {'led_channel': 3, 'display_name': 'BF'},
                'Blue': {'led_channel': 3, 'excitation_nm': 405},
                'Green': {'led_channel': 1, 'excitation_nm': 488},
            }
        }
        identity = resolve(path, board_block=block, motor_model='LS850T')
        assert (identity.source, identity.layers) == ('motorconfig', ())
        assert [n.reason for n in REPORTED] == ['channel_block_unusable']
        assert 'BF and Blue' in str(REPORTED[0])

    def test_a_block_that_is_not_a_mapping_leaves_the_unit_no_channels(self, tmp_path):
        path = write_scopes(tmp_path, {'LS850T': {'Layers': LS850T_ROWS}})
        identity = resolve(path, board_block={'Channels': LS850T_ROWS}, motor_model='LS850T')
        assert identity.layers == ()
        assert [n.reason for n in REPORTED] == ['channel_block_unusable']

    @pytest.mark.parametrize(
        'block',
        [{'Layers': LS560_ROWS, 'Filterset': 'Custom-X'}, {'Filterset': 'Custom-X'}],
        ids=['old Layers block', 'Filterset alone'],
    )
    def test_a_board_without_a_channels_block_takes_its_models_and_says_so(self, tmp_path, block):
        path = write_scopes(tmp_path, {'LS850T': {'Layers': LS850T_ROWS, 'Filterset': 'Stock'}})
        identity = resolve(path, board_block=block, motor_model='LS850T')
        assert (identity.source, identity.filterset, len(identity.layers)) == ('scopes', 'Stock', 7)
        assert [n.reason for n in REPORTED] == ['channel_block_superseded']

    def test_a_scope_whose_board_has_the_old_block_tells_its_outcome_listener(self, tmp_path):
        from modules.scope_session import ScopeSession
        from tests.settings_fixtures import complete_settings

        session = ScopeSession.create(
            complete_settings(microscope='LS850'), simulate=True, warn_pre_release=False
        )
        heard = []
        session.add_outcome_listener(heard.append)
        try:
            session.scope._motion_driver.motorconfig.update_from_board(
                {'Layers': LS560_ROWS, 'Filterset': 'Custom-X'}
            )
            identity = session.scope.refresh_layer_identity()
        finally:
            session.shutdown()
        assert (identity.source, identity.model) == ('scopes', 'LS850')
        assert [getattr(n, 'reason', None) for n in heard] == ['channel_block_superseded']

    def test_two_model_rows_on_one_led_address_leave_the_model_no_layers(self, tmp_path):
        rows = [row('BF', 'BF', 3), row('PC', 'PC', 3)]
        path = write_scopes(tmp_path, {'M': {'Layers': rows}})
        identity = resolve(path, motor_model='M')
        assert identity.layers == ()
        assert 'BF and PC both name LED address 3' in errors_logged()


class TestSnapshotSemantics:
    def test_records_are_immutable(self):
        record = LayerRecord(
            id=0,
            key_name='BF',
            display_name='BF',
            led_channel=(3,),
            excitation_nm=None,
            modality='transmitted',
            background='bright',
            display_color='#ffffff',
            label='BF',
        )
        with pytest.raises(AttributeError):
            record.display_name = 'PC-BF'
        identity = LayerIdentity(layers=(record,), filterset='', source='scopes', model=None)
        with pytest.raises(AttributeError):
            identity.filterset = 'X'

    def test_find_maps_key_name_exactly_once(self, tmp_path):
        path = write_scopes(tmp_path, {'LS560': {'Layers': LS560_ROWS}})
        identity = resolve(path, configured_model='LS560')
        assert identity.find('BF').display_name == 'PC-BF'
        assert identity.find('PC-BF') is None
