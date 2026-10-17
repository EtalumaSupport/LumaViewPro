# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run converts its plate positions with the stage offset it was prepared at.

The run reads the scope's offset once, when it is prepared, as a frozen
``StageOffset``: a change to the setting while it runs moves none of its
steps, and the next run is prepared with the change.
"""

from modules.coord_transformations import StageOffset
from tests.scope_fakes import bind_settings_like_a_session
from tests.test_a_run_cannot_start_where_it_cannot_save import _prepare
from tests.test_protocol_execution import (  # noqa: F401 -- pytest fixtures
    executor,
    executors,
    scope,
)


def test_a_run_holds_the_offset_it_was_prepared_at_and_the_next_reads_the_change(
    executor, scope, tmp_path
):
    bind_settings_like_a_session(scope, stage_offset={'x': 100.0, 'y': 50.0})
    first = _prepare(executor, tmp_path, str(tmp_path))

    bind_settings_like_a_session(scope, stage_offset={'x': 200.0, 'y': 50.0})
    second = _prepare(executor, tmp_path, str(tmp_path))

    assert first.stage_offset == StageOffset(x_um=100.0, y_um=50.0)
    assert second.stage_offset == StageOffset(x_um=200.0, y_um=50.0)
