# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run-state edge whose redraw raises is reported there, and the app keeps running.

The GUI hears each run-state edge by scheduling ``publish_run_state`` on the
clock. A raise out of a clock callback reaches Kivy's main loop, whose crash
guard re-raises anything a plugin did not cause, and the application closes.
No person asked for that redraw and nothing waits on it, so its fault is
reported where it stops, unsolicited, and the next edge draws again.
"""

import ast

from modules.notification_center import notifications
from tests.ast_seams import find_def


class TestRunUnasked:
    def test_a_draw_that_raises_is_reported_unasked_and_not_raised(self, monkeypatch):
        from ui.ui_helpers import run_unasked

        reports = []
        monkeypatch.setattr(
            notifications, 'report_outcome', lambda exception, **kw: reports.append((exception, kw))
        )
        failure = RuntimeError('a widget failed to draw')

        def _draw():
            raise failure

        run_unasked(_draw, 'RUN_STATE')

        assert [e for e, _ in reports] == [failure]
        assert reports[0][1]['solicited'] is False, 'no person asked for a run-state redraw'

    def test_each_edge_draws_again_after_a_failed_one(self, monkeypatch):
        from ui.ui_helpers import run_unasked

        reports = []
        monkeypatch.setattr(
            notifications, 'report_outcome', lambda exception, **kw: reports.append(exception)
        )
        drawn = []

        def _draw():
            drawn.append(True)
            if len(drawn) == 1:
                raise RuntimeError('the first edge failed to draw')

        assert run_unasked(_draw, 'RUN_STATE') is False
        assert run_unasked(_draw, 'RUN_STATE') is True

        assert len(drawn) == 2
        assert len(reports) == 1


def test_the_apps_run_state_listener_draws_through_run_unasked():
    on_start = find_def('lumaviewpro.py', 'on_start', class_name='LumaViewProApp')
    assert on_start is not None, 'LumaViewProApp.on_start is gone'
    listeners = [
        node
        for node in ast.walk(on_start)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'add_run_state_listener'
    ]
    assert len(listeners) == 1, 'the App registers one run-state listener'
    registered = ast.unparse(listeners[0])
    assert 'publish_run_state' in registered
    assert 'run_unasked' in registered, (
        'publish_run_state runs bare on the clock: a raise in it closes the application'
    )
