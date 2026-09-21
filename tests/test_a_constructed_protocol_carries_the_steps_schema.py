# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A protocol constructed from a config carries the full steps schema at any row count.

The steps setter already substitutes the canonical typed empty frame for
any empty frame, so a protocol with no steps stays queryable after every
mutation. Construction assigned the caller's config directly and skipped
the setter, so a caller-supplied empty frame -- an L2 caller's
``create_protocol(config=...)``, or the reader's own zero-row parse --
was stored as given: ``df[['X', 'Y']]`` raised KeyError on a frame with
no columns, and a zero-row parse kept the CSV reader's untyped columns.
"""

from __future__ import annotations

import pathlib

import pandas as pd

from modules.protocol import Protocol

REPO = pathlib.Path(__file__).resolve().parent.parent
TILING_CONFIGS = REPO / 'data' / 'tiling.json'
SHIPPED_DEFAULT = REPO / 'data' / 'new_default_protocol.tsv'


def test_a_column_less_frame_from_a_caller_is_queryable():
    protocol = Protocol(tiling_configs_file_loc=TILING_CONFIGS, config={'steps': pd.DataFrame()})

    assert protocol.num_steps() == 0
    assert protocol.steps()[['X', 'Y']].shape == (0, 2)


def test_a_zero_row_parse_is_stored_with_the_canonical_schema():
    steps = Protocol.from_file(
        file_path=SHIPPED_DEFAULT, tiling_configs_file_loc=TILING_CONFIGS
    ).steps()
    canonical = Protocol._create_empty_steps_df()

    assert len(steps) == 0
    assert dict(steps.dtypes) == dict(canonical.dtypes)
