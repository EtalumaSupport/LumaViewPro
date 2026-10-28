# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A real PostProcessingAPI on its own started lane, for a test that builds without a Session."""

import contextlib
import pathlib
from collections.abc import Iterator

from modules.layer_record import release_colors
from modules.post_processing_api import PostProcessingAPI
from modules.sequential_io_executor import SequentialIOExecutor
from tests.installation_fixtures import shipped_grids

REPO = pathlib.Path(__file__).resolve().parents[1]


@contextlib.contextmanager
def a_post_processing_api() -> Iterator[PostProcessingAPI]:
    """The session's builds on a lane of their own, shut down on exit.

    No scope holds anything and no run's files drain, so a caller's build is
    never refused for them.
    """
    lane = SequentialIOExecutor(name='POSTPROC_TEST')
    lane.start()
    try:
        yield PostProcessingAPI(
            lane=lane,
            tiling_grids=shipped_grids(),
            has_turret=lambda: False,
            settings_snapshot=dict,
            channel_colors=release_colors,
            scope_holder=lambda: None,
            files_draining=lambda: False,
        )
    finally:
        lane.shutdown()
