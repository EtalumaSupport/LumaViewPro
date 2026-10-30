# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Going to a step does not read the camera on the GUI thread to write a log line.

`go_to_step` runs on the Kivy thread. Its debug trace of the camera's gain
and exposure used to ask the camera itself, a hardware round trip that
can wait behind a camera lane busy with a grab, for a line nobody reads
unless debug logging is on. The trace went with the go-to's move into the
Session; the GUI's part reads the camera not at all.
"""

from __future__ import annotations

import threading

from modules.scope_session import ScopeSession
from tests.test_a_refused_step_navigation_changes_nothing import (  # noqa: F401 -- nav_env is a fixture
    ON_TURRET,
    _navigate,
    nav_env,
    pytestmark,
)


def test_going_to_a_step_asks_the_camera_nothing(nav_env, monkeypatch):
    """The camera's own reads, outside the Session's go-to, are the GUI's: none."""
    in_session = threading.local()
    real_go = ScopeSession.start_go_to_step

    def _start_go_to_step(self, protocol, step_index):
        in_session.depth = getattr(in_session, 'depth', 0) + 1
        try:
            return real_go(self, protocol, step_index)
        finally:
            in_session.depth -= 1

    monkeypatch.setattr(ScopeSession, 'start_go_to_step', _start_go_to_step)
    camera = nav_env.session.scope.camera
    asked = []
    for member in ('read_gain_db', 'read_exposure_ms'):
        real_read = getattr(camera, member)

        def _read(*args, _real=real_read, _member=member, **kwargs):
            if not getattr(in_session, 'depth', 0):
                asked.append(_member)
            return _real(*args, **kwargs)

        monkeypatch.setattr(camera, member, _read)

    _navigate(nav_env, ON_TURRET)

    assert nav_env.went == [(1, None)], 'the navigation never reached its move'
    assert asked == [], 'go_to_step read the camera itself'
