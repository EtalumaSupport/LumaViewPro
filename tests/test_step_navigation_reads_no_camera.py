# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Going to a step does not read the camera on the GUI thread to write a log line.

`go_to_step` runs on the Kivy thread. Its debug trace of the camera's gain
and exposure used to ask the camera itself, a hardware round trip that
can wait behind a camera lane busy with a grab, for a line nobody reads
unless debug logging is on. The trace takes the imaging API's cached
values, which is what the camera last reported.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from tests.test_a_refused_step_navigation_changes_nothing import (  # noqa: F401 -- nav_env is a fixture
    ON_TURRET,
    _navigate,
    nav_env,
)


def test_going_to_a_step_asks_the_camera_nothing(nav_env):
    imaging = SimpleNamespace(
        active_cached=True,
        gain_db_cached=4.0,
        exposure_ms_cached=25.0,
        get_gain_db=MagicMock(return_value=4.0),
        get_exposure_ms=MagicMock(return_value=25.0),
    )
    nav_env.scope.imaging = imaging

    _navigate(ON_TURRET)

    assert nav_env.move_absolute.call_count > 0, 'the navigation never reached its moves'
    assert imaging.get_gain_db.call_count == 0, 'go_to_step read the gain from the camera'
    assert imaging.get_exposure_ms.call_count == 0, 'go_to_step read the exposure from the camera'
