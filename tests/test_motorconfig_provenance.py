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


def make_config(tmp_path, defaults=None):
    path = tmp_path / 'defaults.json'
    path.write_text(json.dumps(defaults if defaults is not None else {}), encoding='utf-8')
    return MotorConfig(defaults_file=path)


def bare_board(monkeypatch, response):
    """A MotorBoard with no serial machinery: only what get_config touches."""
    board = MotorBoard.__new__(MotorBoard)
    monkeypatch.setattr(board, 'exchange_command', lambda cmd, **kw: response, raising=False)
    return board


class TestGetConfig:
    def test_valid_json_parses(self, monkeypatch):
        board = bare_board(monkeypatch, '{"Serial Number": "SN1"}')
        assert board.get_config() == {'Serial Number': 'SN1'}

    def test_legacy_dict_repr_parses(self, monkeypatch):
        board = bare_board(monkeypatch, "{'Serial Number': 'SN1'}")
        assert board.get_config() == {'Serial Number': 'SN1'}

    def test_empty_mapping_is_a_valid_answer(self, monkeypatch):
        board = bare_board(monkeypatch, '{}')
        assert board.get_config() == {}

    def test_no_answer_raises(self, monkeypatch):
        board = bare_board(monkeypatch, None)
        with pytest.raises(ConfigReadError):
            board.get_config()

    def test_unparseable_payload_raises(self, monkeypatch):
        board = bare_board(monkeypatch, 'garbage %% not json')
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
            defaults={'Layers': {'hypothetical': 'dict-shaped'}, 'Filterset': 'Old'},
        )
        config.update_from_board({'Layers': [{'key_name': 'BF'}], 'Filterset': 'New'})
        block = config.led_block()
        assert block == {'Layers': [{'key_name': 'BF'}], 'Filterset': 'New'}

    def test_absent_block_is_none_not_empty(self, tmp_path):
        config = make_config(tmp_path)
        config.update_from_board({'Serial Number': 'SN1'})
        assert config.led_block() is None

    def test_block_without_filterset_gets_an_empty_string(self, tmp_path):
        config = make_config(tmp_path)
        config.update_from_board({'Layers': []})
        assert config.led_block() == {'Layers': [], 'Filterset': ''}


def test_defaults_file_never_carries_an_led_block():
    """The shipped defaults must not gain the LED/filterset keys.

    A defaults-side block would deep-merge under a board's partial block,
    producing a layer list no physical filterset matches. The block's only
    legitimate homes are the board's own config and the model rows.
    """
    with open(pathlib.Path('data/motorconfig_defaults.json'), encoding='utf-8') as f:
        defaults = json.load(f)
    assert 'Layers' not in defaults
    assert 'Filterset' not in defaults
