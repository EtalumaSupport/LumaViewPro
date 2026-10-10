# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A button tapped straight off a refused field edit does not act; off a taken one, it does.

Kivy commits a focused field when a touch elsewhere ends, and a Button's
on_release runs after that commit, in the same frame. So the refusal is
already reported when the button's handler asks the boundary whether this
input carried one. This is the order measured with real touches; a stubbed
widget tree cannot show it, so a real (hidden) window runs in a child
process, through the real boundary and the protocol's real range rule.

A window needs a display. Linux CI has none, so there the test is skipped;
the suite run before every push runs on a desktop and runs it.
"""

import os
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent

_SCRIPT = r"""
import os, sys
os.environ['KIVY_NO_ARGS'] = '1'
os.environ['KIVY_NO_CONSOLELOG'] = '1'
sys.path.insert(0, sys.argv[1])
from kivy.config import Config
Config.set('graphics', 'window_state', 'hidden')
from kivy.base import EventLoop
from kivy.core.window import Window
from kivy.tests.common import UnitTestTouch
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.textinput import TextInput
from kivy.uix.togglebutton import ToggleButton
from modules.notification_center import notifications
from modules.protocol import schedule_from_units
from ui.ui_helpers import refused_in_this_input, run_reported

EventLoop.ensure_window()
log = []
notifications.report_outcome = lambda exc, **kw: log.append('reported ' + type(exc).__name__)
held = {'period': 10.0}

def commit(field, focused):
    if not focused:
        def edit():
            schedule_from_units('period', field.text)
            held['period'] = float(field.text)
        run_reported(edit, None, 'PROTOCOL_PERIOD')

def run(*_):
    log.append('not started' if refused_in_this_input() else 'started at %s' % held['period'])

root = FloatLayout(size=(800, 600))
field = TextInput(multiline=False, size_hint=(None, None), size=(200, 40), pos=(10, 500))
field.bind(focus=commit)
button = ToggleButton(text='Run', size_hint=(None, None), size=(100, 40), pos=(300, 500))
button.bind(on_release=run)
root.add_widget(field)
root.add_widget(button)
Window.add_widget(root)

for typed in ('0.005', '20'):
    log.clear()
    for _ in range(3):
        EventLoop.idle()
    field.focus = True
    field.text = typed
    touch = UnitTestTouch(350, 520)
    touch.touch_down()
    touch.touch_up()
    print('%s -> %s' % (typed, log))
"""


@pytest.mark.skipif(
    sys.platform.startswith('linux') and not os.environ.get('DISPLAY'),
    reason='a real touch needs a window, and this Linux host has no display',
)
def test_run_off_a_refused_period_does_not_start_and_off_a_taken_one_does():
    result = subprocess.run(
        [sys.executable, '-c', _SCRIPT, str(REPO)],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    lines = [line for line in result.stdout.splitlines() if '->' in line]
    assert lines == [
        "0.005 -> ['reported ProtocolScheduleRefusedError', 'not started']",
        "20 -> ['started at 20.0']",
    ]
