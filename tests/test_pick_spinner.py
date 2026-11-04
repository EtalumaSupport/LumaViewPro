# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A spinner's record is the person's pick, written from the pick.

A plain Spinner's text-change event fires for the app writing the spinner as
much as for a person choosing, so a record written from it said a person
picked what the app restored at startup. ``PickSpinner`` writes
``SELECT <log_name> <option>`` from the dropdown's pick, the one path a
person's choice takes, before anything acts on it.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from ui.pick_spinner import PickSpinner


@pytest.fixture
def record(caplog):
    caplog.set_level(logging.INFO, logger='LVP.gui_interactions')
    return lambda: [r.getMessage() for r in caplog.records if r.name == 'LVP.gui_interactions']


def _spinner(**props):
    # a stand-in by design: Kivy is stubbed in the test process; the subject is the pick path
    fields = {'text': '10x Oly', 'is_open': True, 'log_name': '', 'text_follows_pick': False}
    spinner = SimpleNamespace(**{**fields, **props}, dispatched=[])
    spinner.dispatch = lambda event, value: spinner.dispatched.append((event, value))
    return spinner


def test_a_pick_is_reported_and_not_taken_as_the_text(record):
    spinner = _spinner()
    PickSpinner._on_dropdown_select(spinner, None, '20x Oly')
    assert spinner.dispatched == [('on_pick', '20x Oly')]
    assert spinner.text == '10x Oly'
    assert spinner.is_open is False
    assert record() == []


def test_a_named_pick_is_recorded_before_anything_acts_on_it(record):
    spinner = _spinner(log_name='OBJECTIVE')
    recorded_when_dispatched = []
    spinner.dispatch = lambda event, value: recorded_when_dispatched.append(list(record()))
    PickSpinner._on_dropdown_select(spinner, None, '20x Oly')
    assert recorded_when_dispatched == [['SELECT OBJECTIVE 20x Oly']]


def test_a_spinner_whose_text_follows_shows_the_pick_after_recording_it(record):
    spinner = _spinner(log_name='BINNING', text_follows_pick=True)
    PickSpinner._on_dropdown_select(spinner, None, '2x2')
    assert spinner.text == '2x2'
    assert record() == ['SELECT BINNING 2x2']
