# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An outcome is shown once means delivered once.

An unsolicited report can be suppressed after the reporter decides to show
it: on shutdown, on an unattended run's mute, or inside the dedup window.
That suppressed post must not spend the object's one show, or the person's
own later request for the same outcome shows nothing at all.
"""

from __future__ import annotations

import pytest

from modules.exceptions import MoveNotCompletedError
from modules.notification_center import NotificationCenter, Severity


@pytest.fixture
def centre():
    c = NotificationCenter(dedup_window_s=10.0)
    c.shown = []
    c.add_listener((lambda n: n.shown and c.shown.append(n)), min_severity=Severity.INFO)
    return c


def _fault():
    return MoveNotCompletedError('Z', 'timed_out')


def test_a_report_muted_by_an_unattended_run_leaves_it_to_the_request(centre):
    fault = _fault()
    centre.open_run_scope(attended=False)
    centre.report_outcome(fault, solicited=False, category='Motion')
    assert centre.shown == []

    centre.report_outcome(fault, solicited=True, category='UI:MOVE_Z')

    assert [n.message for n in centre.shown] == [str(fault)]


def test_a_report_inside_the_dedup_window_leaves_it_to_the_request(centre):
    earlier, fault = _fault(), _fault()
    centre.report_outcome(earlier, solicited=False, category='Motion')
    centre.report_outcome(fault, solicited=False, category='Motion')
    assert len(centre.shown) == 1

    centre.report_outcome(fault, solicited=True, category='Motion')

    assert len(centre.shown) == 2


def test_a_report_during_shutdown_is_not_spent(centre):
    fault = _fault()
    centre.set_shutting_down(True)
    centre.report_outcome(fault, solicited=False, category='Motion')
    centre.set_shutting_down(False)

    centre.report_outcome(fault, solicited=True, category='UI:MOVE_Z')

    assert len(centre.shown) == 1


def test_a_delivered_show_is_still_spent(centre):
    fault = _fault()
    centre.report_outcome(fault, solicited=False, category='Motion')
    centre.report_outcome(fault, solicited=True, category='UI:MOVE_Z')

    assert len(centre.shown) == 1
