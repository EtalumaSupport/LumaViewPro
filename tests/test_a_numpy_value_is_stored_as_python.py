# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A numpy value is made Python's, and a closed set's member its string, by the owner that takes it into the settings, a protocol or a method file.

Two JSON encoders compensated for numpy values reaching a file (#818). The
owners now convert where a value enters what they store: the settings writer
and the dict a session is handed, a protocol's config cells, and the
cell-count method's save. Nothing else changes and nothing is refused: a
``timedelta``, an array a caller owns, a mapping's int keys and NaN stay as
they are. A ``StrEnum`` member is stored as its plain string, so the
settings' exact-type kind check reads it as the string it is.
"""

import datetime
import enum
import json
import math

import numpy as np
import pytest

from modules import post_processing
from modules.python_values import python_values
from tests.settings_fixtures import complete_settings
from modules.protocol import Protocol
from tests.scope_fakes import stock_unit
from tests.test_run_refusal_contract import (
    _build_real_protocol,
    _make_single_step_protocol,
)
from tests.installation_fixtures import shipped_grids


def test_numpy_scalars_at_any_depth_become_python_and_nothing_else_changes():
    buffer = np.zeros(3)
    when = datetime.timedelta(seconds=5)
    value = {
        'a': np.int64(1),
        'b': [np.float32(0.5), {'c': np.bool_(True)}],
        2: math.nan,
        'when': when,
        'buffer': buffer,
    }
    inner = value['b']

    returned = python_values(value)

    assert returned is value and value['b'] is inner, 'a container was replaced, not converted'
    assert value['a'] == 1 and type(value['a']) is int
    assert value['b'][0] == 0.5 and type(value['b'][0]) is float
    assert value['b'][1]['c'] is True
    assert math.isnan(value[2]) and 2 in value
    assert value['when'] is when
    assert value['buffer'] is buffer
    assert python_values(np.float32(1.5)) == 1.5 and type(python_values(np.float32(1.5))) is float


class _Word(enum.StrEnum):
    VIDEO = 'video'
    CENTER = 'center'


def test_a_str_enum_member_at_any_depth_becomes_its_plain_string():
    value = {'a': _Word.VIDEO, 'b': [_Word.CENTER]}

    python_values(value)

    assert (type(value['a']), value['a']) == (str, 'video')
    assert (type(value['b'][0]), value['b'][0]) == (str, 'center')


@pytest.fixture
def sim_session(tmp_path):
    from modules.scope_session import ScopeSession

    session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    try:
        yield session
    finally:
        session.shutdown()


class TestTheSettings:
    def test_a_written_numpy_scalar_is_stored_as_python(self, sim_session):
        sim_session.update_settings('video.max_fps', np.int64(25))

        stored = sim_session.get_setting('video.max_fps')
        assert (type(stored), stored) == (int, 25)

    def test_a_handed_dict_is_stored_as_python_in_the_same_object(self, tmp_path):
        from modules.scope_session import ScopeSession

        handed = complete_settings(live_folder=str(tmp_path))
        handed['BF']['gain'] = np.float32(2.0)
        handed['video']['max_fps'] = np.int64(10)

        session = ScopeSession.create(
            handed, simulate=True, settings_file=str(tmp_path / 'saved.json')
        )
        try:
            assert session.settings is handed
            assert type(handed['BF']['gain']) is float
            assert type(handed['video']['max_fps']) is int
        finally:
            session.shutdown()

        # Written by plain json, with no encoder behind it.
        saved = json.loads((tmp_path / 'saved.json').read_text())
        assert saved['video']['max_fps'] == 10


class TestAProtocolsConfigCells:
    @staticmethod
    def _protocol_with_numpy_config():
        row = _make_single_step_protocol().steps().iloc[0].to_dict()
        video = {'duration': np.int64(1), 'fps': np.float32(5.0)}
        row['Video Config'] = video
        return _build_real_protocol([row]), video

    def test_a_config_cell_reads_back_as_python_and_the_callers_dict_is_kept(self):
        protocol, video = self._protocol_with_numpy_config()

        cell = protocol.step_row(0)['Video Config']

        assert cell == {'duration': 1, 'fps': 5.0}
        assert type(cell['duration']) is int and type(cell['fps']) is float
        assert type(video['duration']) is np.int64, "the caller's dict was changed"

    def test_the_protocol_is_written_and_read_back(self, tmp_path):
        protocol, _ = self._protocol_with_numpy_config()
        path = tmp_path / 'p.tsv'

        protocol.to_file(path)

        reopened = Protocol.from_file(path, shipped_grids(), unit=stock_unit())
        assert reopened.step_row(0)['Video Config'] == {'duration': 1, 'fps': 5.0}


def test_a_method_holding_numpy_is_saved_as_python(tmp_path):
    from modules.cell_count_method import CellCountFilters, CellCountMethod, FilterBounds

    method = CellCountMethod(
        threshold_pct=np.int64(20),
        filters=CellCountFilters(area_um2=FilterBounds(np.float32(10.0), np.int64(500))),
    )
    path = tmp_path / 'method.json'

    post_processing.save_cell_count_method(method, path)

    saved = json.loads(path.read_text())
    assert saved['filters']['area'] == {'min': 10.0, 'max': 500}
    assert saved['segmentation']['parameters']['threshold'] == 20
    assert type(method.filters.area_um2.max) is np.int64, "the caller's method was changed"
