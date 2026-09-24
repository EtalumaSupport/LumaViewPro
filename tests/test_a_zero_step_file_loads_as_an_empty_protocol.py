# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A protocol file that parses but carries no steps loads as a real, empty Protocol.

The reader used to abandon its own parse at zero rows and hand back False
where a Protocol was promised, so a fresh install -- whose shipped default
protocol has no steps -- either crashed on the bool or silently adopted an
empty protocol built from the UI's settings, discarding the file's own
labware, period and duration. The shipped file IS the fixture here.

Finishing the parse exposes a second defect at zero rows: the recovered
``Label`` and ``Auto_Named`` columns were assigned from empty lists, which
pandas types float64. ``Label`` then raised at its ``.str`` consumer;
``Auto_Named`` was silently the wrong type. The per-row ``apply`` calls that
parse the config columns have the zero-row twin of the same defect: pandas
invokes the function once on a phantom all-NaN row to infer a result type,
so an empty file logged two ERRORs for a step that does not exist. The
dtype assertions pin every per-row column, because only a dtype assertion
catches the silent ones, and the log assertion pins the phantom row.
"""

from __future__ import annotations

import datetime
import pathlib
from unittest.mock import patch

import pandas as pd

import modules.protocol as protocol_module
from modules.protocol import Protocol

REPO = pathlib.Path(__file__).resolve().parent.parent
TILING_CONFIGS = REPO / 'data' / 'tiling.json'
SHIPPED_DEFAULT = REPO / 'data' / 'new_default_protocol.tsv'
STEPPED_EXAMPLE = REPO / 'data' / 'example_protocol.tsv'


def _load(path: pathlib.Path) -> Protocol:
    return Protocol.from_file(file_path=path, tiling_configs_file_loc=TILING_CONFIGS)


def test_the_shipped_default_loads_as_an_empty_protocol_carrying_the_files_own_facts():
    protocol = _load(SHIPPED_DEFAULT)

    assert isinstance(protocol, Protocol)
    assert protocol.num_steps() == 0
    assert protocol.period() == datetime.timedelta(minutes=20)
    assert protocol.duration() == datetime.timedelta(days=2)
    assert protocol.labware() == '96 well microplate'
    assert protocol.capture_root() == ''
    assert protocol.validate_steps() == []


def test_the_per_row_columns_keep_their_type_at_zero_rows():
    steps = _load(SHIPPED_DEFAULT).steps()

    assert len(steps) == 0
    assert pd.api.types.is_string_dtype(steps['Label'])
    assert steps['Auto_Named'].dtype == bool
    assert pd.api.types.is_string_dtype(steps['Name'])
    assert steps['Video Config'].dtype == object
    assert steps['Stim_Config'].dtype == object


def test_a_zero_step_load_logs_nothing_for_a_step_that_does_not_exist():
    with patch.object(protocol_module, 'logger') as logger:
        _load(SHIPPED_DEFAULT)

    assert logger.error.call_args_list == []
    assert logger.warning.call_args_list == []


def test_a_stepped_file_is_unaffected():
    steps = _load(STEPPED_EXAMPLE).steps()

    assert len(steps) == 96
    assert pd.api.types.is_string_dtype(steps['Label'])
    assert steps['Auto_Named'].dtype == bool
    assert steps['Label'].head(5).tolist() == ['', '', '', '', '']
    assert steps['Auto_Named'].head(5).tolist() == [False] * 5
