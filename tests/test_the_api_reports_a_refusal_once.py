# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An API funnel reports its refusal once; a second report of it shows nothing.

A refusal is reported by the funnel that raises it -- logged once, shown at
most once -- and then raised, so the caller who waits gets it. That caller
may be a GUI boundary that hands everything it catches to the same
reporter. The show-once mark travels on the exception object, so that
second report is a no-op; a funnel that posts a warning of its own instead
of going through the reporter leaves no mark, and the boundary shows the
refusal twice.

Also here: the Session wires the run's return to IDLE to its run-state
listeners.
"""

import pytest

from modules.exceptions import AxisStateUnknownError, ProtocolRunRefusedError
from modules.notification_center import Severity
from modules.protocol import Protocol
from tests.test_a_zstack_with_no_range_is_refused import (  # noqa: F401
    _TILING,
    _standalone_config,
    sim_scope,
)


def _report_again(error):
    from modules import notification_center

    notification_center.notifications.report_outcome(error, solicited=True, category='UI:TEST')


def test_the_zstack_builders_refusal_is_shown_once_however_often_it_is_reported(
    sim_scope, centre_posts
):
    with pytest.raises(ProtocolRunRefusedError) as refusal:
        Protocol.from_config(
            input_config=_standalone_config({'range': 0.0, 'step_size': 5.0}),
            tiling_configs_file_loc=_TILING,
            capabilities=sim_scope.capabilities,
            objective_helper=sim_scope.objective_helper,
            wellplate_loader=sim_scope.wellplate_loader,
        )
    _report_again(refusal.value)

    assert [(n.title, n.severity) for n in centre_posts] == [
        ('Z-Stack Not Configured', Severity.WARNING)
    ]


def test_the_motion_refusal_is_shown_once_however_often_it_is_reported(sim_scope, centre_posts):
    with pytest.raises(AxisStateUnknownError) as refusal:
        sim_scope.motion.refuse_unknown_positions(('Z',), recording=True, then='save the focus')
    _report_again(refusal.value)

    assert [(n.title, n.severity) for n in centre_posts] == [('Scope Not Homed', Severity.WARNING)]


def test_the_sessions_listeners_hear_a_run_return_to_idle(tmp_path):
    from modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), microscope='LS850T'), simulate=True
    )
    try:
        heard = []
        session.add_run_state_listener(lambda: heard.append(True))
        heard.clear()  # the level republish on registering

        session.sequenced_capture_runner._on_run_idle()

        assert heard == [True], 'the idle edge must reach the run-state listeners'
    finally:
        session.shutdown()
