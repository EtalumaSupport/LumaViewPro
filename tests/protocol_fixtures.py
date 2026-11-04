# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A protocol built or read the way a test with no scope builds one.

A scope builds its protocols on what it read at bring-up: its tiling grids
and its unit. A test that builds one without a scope hands in the shipped
grids and a stock unit instead, here, so a new argument a protocol needs is
added in one place rather than at every site
(``tests/guards/test_a_test_builds_a_protocol_through_its_fixtures.py``).
"""

from modules.protocol import Protocol
from tests.installation_fixtures import shipped_grids
from tests.scope_fakes import stock_unit


def protocol_of(config=None, **kwargs) -> Protocol:
    """A protocol over ``config`` on the shipped tiling grids."""
    return Protocol(tiling_grids=shipped_grids(), config=config, **kwargs)


def protocol_from_file(file_path, unit=None, **kwargs) -> Protocol:
    """The protocol in ``file_path``, read on the shipped grids for ``unit``, a stock LS850 when None."""
    return Protocol.from_file(
        file_path=file_path,
        tiling_grids=shipped_grids(),
        unit=stock_unit() if unit is None else unit,
        **kwargs,
    )
