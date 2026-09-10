# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The steps frame carries its full column schema at every row count.

Bug history
-----------
Bench 2026-07-28: applying tiling raised
``[UI] apply_tiling failed: "None of [Index(['X', 'Y'], dtype='str')] are
in the [columns]"`` and surfaced as an error popup.

The expansions replace the steps frame with
``pd.DataFrame.from_dict(rows)``. pandas gives that frame NO columns at
all when ``rows`` is empty -- which is what every input step being
skipped produces -- so a consumer's ``df[['X', 'Y']]`` raised KeyError
instead of returning an empty selection.

The schema is restored in ``_set_steps``, the single place the frame is
replaced, rather than in each consumer: a protocol with no steps stays a
queryable protocol for tiling, z-stacking, delete, insert and load alike.

These tests pin the CONTRACT (schema at any row count), not the one
crashing caller.
"""

from __future__ import annotations

import pandas as pd

from modules.protocol import Protocol
from tests.test_protocol_roundtrip import _build_protocol, _make_step


# center reference at Z=5000, range 100, step 20 -> slices 4950..5050
_ZSTACK = {'range': 100.0, 'step_size': 20.0, 'z_reference': 'center'}

_SCHEMA_COLUMNS = list(Protocol._create_empty_steps_df().columns)


def _proto():
    # z_slice=-1 marks a not-yet-stacked step so apply_zstacking expands it.
    return _build_protocol([_make_step(name='A1_BF', z=5000.0, z_slice=-1)])


def test_zstacking_every_slice_out_of_range_keeps_the_schema():
    """The production path that empties the row list still yields a
    frame every consumer can query."""
    proto = _proto()
    # No slice can land inside [0, 100] when the stack spans 4950..5050.
    axes_config = {'Z': {'limits': {'min': 0.0, 'max': 100.0}}}

    proto.apply_zstacking(zstack_params=_ZSTACK, axes_config=axes_config)

    steps = proto.steps()
    assert len(steps) == 0
    assert list(steps.columns) == _SCHEMA_COLUMNS
    # The selection that raised KeyError at the bench.
    assert len(steps[['X', 'Y']]) == 0


def test_empty_replacement_frame_regains_schema_and_dtypes():
    """Assigning a column-less frame is not a way to lose the schema."""
    proto = _proto()
    schema = Protocol._create_empty_steps_df()

    proto._set_steps(pd.DataFrame.from_dict([]))

    steps = proto.steps()
    assert list(steps.columns) == _SCHEMA_COLUMNS
    assert steps.empty
    # An object-typed X column would silently re-admit the numpy/native
    # scalar confusion the step accessors exist to prevent.
    for column in _SCHEMA_COLUMNS:
        assert steps[column].dtype == schema[column].dtype, column


def test_populated_frame_is_passed_through_untouched():
    """The schema restore must not rewrite or reorder real rows."""
    proto = _proto()
    axes_config = {'Z': {'limits': {'min': 0.0, 'max': 10000.0}}}

    proto.apply_zstacking(zstack_params=_ZSTACK, axes_config=axes_config)

    steps = proto.steps()
    assert len(steps) == 6
    assert list(steps.columns) == _SCHEMA_COLUMNS
    assert steps['Z'].tolist() == [4950.0, 4970.0, 4990.0, 5010.0, 5030.0, 5050.0]
