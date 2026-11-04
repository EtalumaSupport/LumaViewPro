# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A CommitInput commits a person's edit or Enter, and nothing else.

The contract is ``CommitInput``'s (ui/ui_helpers.py): a click in and out
with nothing typed commits nothing; an edit left, or an Enter whatever the
box holds, commits once; and the box is never written from code while it
has focus -- a write then is held, shown if the person leaves the box
unedited, dropped if they leave it edited.

The suite stubs Kivy, and the class rests on Kivy's own focus, validate and
text machinery (its private ``_get_text`` / ``_set_text`` among them), so a
stub cannot show any of it: the real Kivy runs in a child process, once,
and each test reads its case from that run. A person's keys arrive as
Kivy's keyboard handlers receive them (``keyboard_on_textinput``,
``_key_down`` for Enter), the box built from a kv rule as the app's are.

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
sys.path.insert(0, sys.argv[1])
from kivy.config import Config
Config.set('graphics', 'window_state', 'hidden')
from kivy.lang import Builder
from kivy.factory import Factory
from ui.ui_helpers import CommitInput  # registers the kv class

Builder.load_string('''
<Probe@BoxLayout>:
    Slider:
        id: slider
        min: 0
        max: 2000
        value: 3
    CommitInput:
        id: box
        multiline: False
        input_filter: 'float'
        text: str(int(slider.value))
        on_commit: app_commits.append(self.text)
''')

cases = {}

def fresh():
    commits = []
    import builtins
    builtins.app_commits = commits
    probe = Factory.Probe()
    return probe.ids.box, probe.ids.slider, commits

def type_keys(box, keys):
    for key in keys:
        box.keyboard_on_textinput(None, key)

def enter(box):
    box._key_down((None, None, 'enter', 1))

box, slider, commits = fresh()
box.focus = True
box.focus = False
cases['click_in_and_out'] = {'commits': commits, 'text': box.text}

box, slider, commits = fresh()
box.focus = True
type_keys(box, '7')
box.focus = False
cases['edit_then_click_away'] = {'commits': commits}

box, slider, commits = fresh()
box.focus = True
enter(box)
cases['enter_unchanged'] = {'commits': commits, 'focus': box.focus}

box, slider, commits = fresh()
box.focus = True
type_keys(box, '9')
box.do_undo()
box.focus = False
cases['edit_undone'] = {'commits': commits}

box, slider, commits = fresh()
box.focus = True
type_keys(box, '7')
box.focus = False
box.focus = True
box.focus = False
cases['click_in_and_out_after_a_commit'] = {'commits': commits}

box, slider, commits = fresh()
box.focus = True
slider.value = 7
while_focused = box.text
box.focus = False
cases['written_while_focused_left_unedited'] = {
    'while_focused': while_focused, 'after': box.text, 'commits': commits}

box, slider, commits = fresh()
slider.value = 999
box.focus = True
box.select_all()
slider.value = 1000
type_keys(box, '5')
box.focus = False
cases['written_between_select_all_and_a_key'] = {'commits': commits, 'text': box.text}

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
def test_a_click_in_and_out_commits_nothing(cases):
    assert cases['click_in_and_out'] == {'commits': [], 'text': '3'}


@pytest.mark.needs_display
def test_an_edit_left_commits_once(cases):
    assert cases['edit_then_click_away']['commits'] == ['37']


@pytest.mark.needs_display
def test_enter_commits_once_whatever_the_box_holds(cases):
    assert cases['enter_unchanged'] == {'commits': ['3'], 'focus': False}


@pytest.mark.needs_display
def test_an_edit_undone_is_no_edit(cases):
    assert cases['edit_undone']['commits'] == []


@pytest.mark.needs_display
def test_a_committed_box_clicked_in_and_out_again_commits_nothing_more(cases):
    assert cases['click_in_and_out_after_a_commit']['commits'] == ['37']


@pytest.mark.needs_display
def test_a_write_while_focused_waits_and_shows_once_the_box_is_left_unedited(cases):
    assert cases['written_while_focused_left_unedited'] == {
        'while_focused': '3',
        'after': '7',
        'commits': [],
    }


@pytest.mark.needs_display
def test_a_write_between_select_all_and_a_key_cannot_mix_into_the_entry(cases):
    # Written into the focused box, Kivy would keep the old selection over the
    # longer new text, and the key would leave '50' -- a number nobody typed.
    assert cases['written_between_select_all_and_a_key'] == {'commits': ['5'], 'text': '5'}
