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

from modules.layer_record import (
    UNRESOLVED,
    LayerIdentity,
    LayerRecord,
    load_layer_catalogue,
    load_scopes_data,
    resolve_layer_identity,
)

CATALOGUE = ['BF', 'PC', 'DF', 'Blue', 'Green', 'Red', 'Lumi']


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

LS560_ROWS = [
    row('BF', 'PC-BF', 3),
    row('Green', 'Green', 1, 488),
]


def errors_logged():
    return ' | '.join(str(c) for c in _mock_logger.error.call_args_list)


def warnings_logged():
    return ' | '.join(str(c) for c in _mock_logger.warning.call_args_list)


def resolve(data_file, **kwargs):
    _mock_logger.reset_mock()
    defaults = {
        'board_block': None,
        'board_config_read_ok': True,
        'motor_model': None,
        'configured_model': None,
    }
    defaults.update(kwargs)
    return resolve_layer_identity(data_file=data_file, **defaults)


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

    def test_missing_layer_order_is_loud_and_empty(self, tmp_path):
        path = tmp_path / 'scopes_fixture.json'
        path.write_text(json.dumps({'LS850T': {'Layers': LS850T_ROWS}}), encoding='utf-8')
        _mock_logger.reset_mock()
        catalogue = load_layer_catalogue(load_scopes_data(str(path)))
        assert catalogue == ()
        assert 'LayerOrder' in errors_logged()

    def test_unreadable_file_is_loud_and_empty(self, tmp_path):
        _mock_logger.reset_mock()
        data = load_scopes_data(str(tmp_path / 'missing.json'))
        assert data == {}
        assert 'unreadable' in errors_logged()


class TestPrecedence:
    def test_block_present_wins_wholesale(self, tmp_path):
        path = write_scopes(tmp_path, {'LS850T': {'Layers': LS850T_ROWS, 'Filterset': 'Stock'}})
        block = {'Layers': LS560_ROWS, 'Filterset': 'Custom-X'}
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
        assert identity == UNRESOLVED
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
        block = {'Layers': LS850T_ROWS, 'Filterset': 'Stock'}
        identity = resolve(path, board_block=block, motor_model='LS850T', override_model='LS560')
        assert [r.key_name for r in identity.layers] == ['BF', 'Green']
        assert 'override' in warnings_logged()

    def test_override_of_unknown_model_is_loud_and_unresolved(self, tmp_path):
        path = write_scopes(tmp_path, {'LS850T': {'Layers': LS850T_ROWS}})
        identity = resolve(path, motor_model='LS850T', override_model='LS9999')
        assert identity == UNRESOLVED
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

    def test_missing_field_is_skipped_loudly(self, tmp_path):
        rows = [row('BF', 'BF', 3), {'key_name': 'Green', 'display_name': 'Green'}]
        path = write_scopes(tmp_path, {'M': {'Layers': rows}})
        identity = resolve(path, motor_model='M')
        assert [r.key_name for r in identity.layers] == ['BF']
        assert 'missing' in errors_logged()

    def test_every_row_unusable_is_a_zero_layer_identity_not_a_fallback(self, tmp_path):
        path = write_scopes(tmp_path, {'LS850T': {'Layers': LS850T_ROWS, 'Filterset': 'Stock'}})
        block = {'Layers': [dict(row('BF', 'BF'), led_channel=[1, 2])], 'Filterset': 'Broken'}
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

    def test_single_channel_and_excitation_normalise(self, tmp_path):
        path = write_scopes(tmp_path, {'M': {'Layers': [row('Green', 'Green', 1, 488)]}})
        green = resolve(path, motor_model='M').find('Green')
        assert green.led_channel == (1,)
        assert green.excitation_nm == pytest.approx(488.0)
        assert isinstance(green.excitation_nm, float)


class TestSnapshotSemantics:
    def test_records_are_immutable(self):
        record = LayerRecord(
            id=0, key_name='BF', display_name='BF', led_channel=(3,), excitation_nm=None
        )
        with pytest.raises(AttributeError):
            record.display_name = 'PC-BF'
        identity = LayerIdentity(layers=(record,), filterset='', source='scopes')
        with pytest.raises(AttributeError):
            identity.filterset = 'X'

    def test_find_maps_key_name_exactly_once(self, tmp_path):
        path = write_scopes(tmp_path, {'LS560': {'Layers': LS560_ROWS}})
        identity = resolve(path, configured_model='LS560')
        assert identity.find('BF').display_name == 'PC-BF'
        assert identity.find('PC-BF') is None
