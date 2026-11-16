# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A repeating GUI callback that raises is reported, and its schedule goes on.

The contract is ``schedule_ui_interval``'s (ui/ui_helpers.py), the one way
the GUI makes a repeating schedule. Kivy ends a repeating schedule whose
callback raised, whatever the application's exception handler answers, so
a readout drawn by a bare one would freeze at its first fault and look
live. Through the routine, a raise is reported unasked under the
schedule's label and the next tick runs; a callback that answers False
still ends its own schedule; cancelling the handle ends it.

The suite stubs Kivy, and the claim is about Kivy's own Clock, so the real
Kivy runs in a child process, once, under a handler that passes every
raise as the application's does, and each test reads its case from that
run. The reports are heard as a client hears them, by a listener on the
notification centre.

A window needs a display. Linux CI has none, so there the tests are
skipped; the suite run before every push runs on a desktop and runs them.
"""

import json
import os
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent

_SCRIPT = r"""
import json, os, sys
os.environ['KIVY_NO_ARGS'] = '1'
os.environ['KIVY_NO_CONSOLELOG'] = '1'
sys.path[:0] = [os.path.join(sys.argv[1], 'src'), sys.argv[1]]
from kivy.base import ExceptionHandler, ExceptionManager
from kivy.clock import Clock
from _lumascope.modules.notification_center import Severity, notifications
from ui.ui_helpers import schedule_ui_interval


class _Passes(ExceptionHandler):
    def handle_exception(self, inst):
        return ExceptionManager.PASS


ExceptionManager.add_handler(_Passes())
posts = []
notifications.add_listener(lambda n: posts.append(n.category), min_severity=Severity.DEBUG)


def ticked(name, ticks, *, raise_on=None, answer_false_on=None, cancel_after=None):
    runs = []

    def callback(dt):
        runs.append(dt)
        if len(runs) == raise_on:
            raise ValueError('a tick that raises')
        if len(runs) == answer_false_on:
            return False

    event = schedule_ui_interval(callback, 0, name)
    for _ in range(ticks):
        Clock.tick()
        if cancel_after is not None and len(runs) == cancel_after:
            event.cancel()
    return {
        'runs': len(runs),
        'scheduled': bool(event.is_triggered),
        'posts': posts.count(f'UI:{name}'),
    }


cases = {
    'raises': ticked('RAISES', 6, raise_on=2),
    'answers_false': ticked('ANSWERS_FALSE', 6, answer_false_on=3),
    'cancelled': ticked('CANCELLED', 6, cancel_after=2),
}
print('CASES ' + json.dumps(cases))
"""


@pytest.fixture(scope='module')
def cases(tmp_path_factory):
    # Kivy reads its config from and writes its logs and icons into KIVY_HOME,
    # so the child gets one of its own rather than the shared ~/.kivy.
    result = subprocess.run(
        [sys.executable, '-c', _SCRIPT, str(REPO)],
        capture_output=True,
        text=True,
        env={**os.environ, 'KIVY_HOME': str(tmp_path_factory.mktemp('kivy_home'))},
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    [line] = [ln for ln in result.stdout.splitlines() if ln.startswith('CASES ')]
    return json.loads(line[len('CASES ') :])


@pytest.mark.needs_display
def test_a_tick_that_raises_is_reported_once_and_the_schedule_runs_on(cases):
    assert cases['raises'] == {'runs': 6, 'scheduled': True, 'posts': 1}


@pytest.mark.needs_display
def test_a_callback_that_answers_false_ends_its_own_schedule(cases):
    assert cases['answers_false'] == {'runs': 3, 'scheduled': False, 'posts': 0}


@pytest.mark.needs_display
def test_cancelling_the_handle_ends_the_schedule(cases):
    assert cases['cancelled'] == {'runs': 2, 'scheduled': False, 'posts': 0}
