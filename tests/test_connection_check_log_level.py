# Copyright Etaluma, Inc.
"""Asking which parts are missing writes nothing to the log.

``Lumascope.unconnected_parts()`` is a read: the GUI's bring-up poll asks it
about every 100 ms, the run loop every 30 s, and a REST client as often as it
likes, so a line per call floods the log. Whoever acts on the answer says
which part is missing: the run's ``not_connected`` refusal, and a mid-run
loss's ending, which names each part gone and is logged once by the reporter.

Each test reads every record logged on its own thread while the read runs,
from any module and at any level.
"""

import threading
from contextlib import contextmanager

import pytest

from modules.lumascope_api import _lumascope as lumascope_mod
from modules.lumascope_api.bring_up import Part
from tests.scope_fakes import build_scope


@contextmanager
def _lines_logged_on_this_thread(caplog):
    """Collect every record this thread logs inside the block."""
    caplog.set_level(1)
    caplog.set_level(1, logger=lumascope_mod.logger.name)
    caplog.clear()
    lines = []
    yield lines
    me = threading.get_ident()
    lines.extend(
        f'{r.levelname} {r.name}: {r.getMessage()}' for r in caplog.records if r.thread == me
    )


@pytest.fixture
def sim_scope():
    scope = build_scope(simulate=True)
    yield scope
    scope.disconnect()


def test_a_scope_with_every_part_is_asked_without_a_line(sim_scope, caplog):
    with _lines_logged_on_this_thread(caplog) as lines:
        missing = sim_scope.unconnected_parts()

    assert missing == ()
    assert lines == [], f'the read logged: {lines}'


def test_a_missing_camera_is_answered_without_a_line(sim_scope, caplog):
    sim_scope.simulation.remove_camera()

    with _lines_logged_on_this_thread(caplog) as lines:
        missing = sim_scope.unconnected_parts()

    assert missing == (Part.CAMERA,)
    assert lines == [], f'the read logged: {lines}'
