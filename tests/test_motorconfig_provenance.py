# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Board-config provenance: a failed READ is a recorded fact, not an empty dict.

Pins the two halves that make no-block distinguishable from
cannot-read-the-block: `get_config` raises a typed error instead of
returning the same `{}` a silent board produces, and the loader converts
that into a queryable flag on the motorconfig while the scope stays up
on defaults. Also pins the wholesale handling of the LED/filterset block
(never deep-merged) and guards the defaults file against ever growing
one -- a defaults-side block would let a partial board block silently
field-merge into hardware that does not exist.
"""

import json
import pathlib

import pytest

from drivers.exceptions import ConfigReadError
from drivers.motorboard import MotorBoard
from drivers.motorconfig import MotorConfig
from lvp_logger import logger as _mock_logger
from modules.path_utils import read_installation_file
from tests.stock_channels import LED2
from tests.wire_boards import motor_on_the_wire


def make_config(tmp_path, defaults=None):
    path = tmp_path / 'defaults.json'
    path.write_text(json.dumps(defaults if defaults is not None else {}), encoding='utf-8')
    return MotorConfig(read_installation_file(path))


def bare_board(monkeypatch, response):
    """A MotorBoard with no serial machinery: only what get_config touches."""
    board = MotorBoard.__new__(MotorBoard)
    monkeypatch.setattr(board, 'exchange_command', lambda cmd, **kw: response, raising=False)
    return board


class TestGetConfig:
    """The firmware's own CONFIG replies, on the firmware-backed simulator: the
    3.0 firmware answers JSON, the field firmware a Python dict repr, and a
    lost or garbled reply is the wire's own fault. ``{}`` and a bare number
    are answers no emulated firmware gives, so those two cases script the
    exchange."""

    @pytest.mark.firmware_tier
    def test_valid_json_parses(self):
        board, _sim = motor_on_the_wire(dialect='3.0')
        assert board.exchange_command('CONFIG').startswith('{"')
        assert board.get_config()['Serial Number'] == 'SIM-0000'

    @pytest.mark.firmware_tier
    def test_legacy_dict_repr_parses(self):
        """The field firmware, the one the shipped scopes run, answers CONFIG
        with a Python dict repr."""
        board, _sim = motor_on_the_wire(dialect='field')
        assert board.exchange_command('CONFIG').startswith("{'")
        assert board.get_config()['Serial Number'] == 'SIM-0000'

    def test_empty_mapping_is_a_valid_answer(self, monkeypatch):
        board = bare_board(monkeypatch, '{}')
        assert board.get_config() == {}

    @pytest.mark.firmware_tier
    @pytest.mark.slow
    def test_no_answer_raises(self):
        """Slow: the lost reply is waited out for the read's own bound."""
        board, sim = motor_on_the_wire()
        sim.drop_next_reply()
        with pytest.raises(ConfigReadError):
            board.get_config()

    @pytest.mark.firmware_tier
    def test_unparseable_payload_raises(self):
        board, sim = motor_on_the_wire()
        sim.garble_next_reply()
        with pytest.raises(ConfigReadError):
            board.get_config()

    def test_non_mapping_payload_raises(self, monkeypatch):
        board = bare_board(monkeypatch, '42')
        with pytest.raises(ConfigReadError):
            board.get_config()


class TestLoaderRecordsProvenance:
    def _board_with_config(self, tmp_path, monkeypatch, response):
        board = MotorBoard.__new__(MotorBoard)
        board.motorconfig = make_config(tmp_path)
        monkeypatch.setattr(board, 'exchange_command', lambda cmd, **kw: response, raising=False)
        monkeypatch.setattr(board, '_rebuild_cached_values', lambda: None, raising=False)
        return board

    def test_failed_read_sets_the_flag_and_survives(self, tmp_path, monkeypatch):
        board = self._board_with_config(tmp_path, monkeypatch, 'garbage %%')
        _mock_logger.reset_mock()
        board._load_board_config()
        assert board.motorconfig.board_config_read_ok is False
        assert any('unreadable' in str(c) for c in _mock_logger.warning.call_args_list)

    def test_clean_read_leaves_the_flag_true(self, tmp_path, monkeypatch):
        board = self._board_with_config(tmp_path, monkeypatch, '{"Serial Number": "SN1"}')
        board._load_board_config()
        assert board.motorconfig.board_config_read_ok is True
        assert board.motorconfig.serial_number() == 'SN1'

    def test_empty_config_is_not_a_failed_read(self, tmp_path, monkeypatch):
        board = self._board_with_config(tmp_path, monkeypatch, '{}')
        board._load_board_config()
        assert board.motorconfig.board_config_read_ok is True

    def test_merge_failure_records_unreliable_state(self, tmp_path, monkeypatch):
        board = self._board_with_config(tmp_path, monkeypatch, '{"Serial Number": "SN1"}')

        def boom():
            raise RuntimeError('cache rebuild failed')

        monkeypatch.setattr(board, '_rebuild_cached_values', boom, raising=False)
        board._load_board_config()
        assert board.motorconfig.board_config_read_ok is False


class TestFlagExistsEverywhere:
    def test_bare_new_construction_carries_the_flag(self):
        config = MotorConfig.__new__(MotorConfig)
        config._config = {}
        assert config.board_config_read_ok is True

    def test_marking_one_instance_does_not_poison_the_class(self, tmp_path):
        first = make_config(tmp_path)
        second = make_config(tmp_path)
        first.mark_board_read_failed()
        assert first.board_config_read_ok is False
        assert second.board_config_read_ok is True


class TestWholesaleLedBlock:
    def test_board_block_replaces_a_defaults_block_whole(self, tmp_path):
        config = make_config(
            tmp_path,
            defaults={'Channels': {'BF': {'led_channel': 3}, LED2: {}}, 'Filterset': 'Old'},
        )
        config.update_from_board({'Channels': {'BF': {'led_channel': 4}}, 'Filterset': 'New'})
        block = config.led_block()
        assert block == {'Channels': {'BF': {'led_channel': 4}}, 'Filterset': 'New'}

    def test_absent_block_is_none_not_empty(self, tmp_path):
        config = make_config(tmp_path)
        config.update_from_board({'Serial Number': 'SN1'})
        assert config.led_block() is None

    def test_the_block_is_the_keys_the_board_holds(self, tmp_path):
        # The identity resolver decides what each key means: a Channels block
        # with no Filterset, and the old Layers block, are handed over as held.
        config = make_config(tmp_path)
        config.update_from_board({'Channels': {}, 'Layers': []})
        assert config.led_block() == {'Channels': {}, 'Layers': []}


def test_defaults_file_never_carries_an_led_block():
    """The shipped defaults must not gain the LED/filterset keys.

    A defaults-side block would deep-merge under a board's partial block,
    producing a layer list no physical filterset matches. The block's only
    legitimate homes are the board's own config and the model rows.
    """
    with open(pathlib.Path('data/motorconfig_defaults.json'), encoding='utf-8') as f:
        defaults = json.load(f)
    assert 'Channels' not in defaults
    assert 'Layers' not in defaults
    assert 'Filterset' not in defaults
