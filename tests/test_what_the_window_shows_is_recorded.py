# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""What the window shows is in the GUI record, as it changes.

The GUI record held what a person did -- presses, picks, popup answers -- but
not what the window then showed: the controls greyed for a home, the homing
banner, the title's "Homing, please wait...". Whether they appeared could
only be asked of whoever was watching. Each is now a line when it changes,
and only then, so a writer that runs on every edge does not repeat it.
"""

from __future__ import annotations

import logging

import pytest

from modules import gui_logger


@pytest.fixture
def record(caplog):
    gui_logger._shown.clear()
    caplog.set_level(logging.INFO, logger='LVP.gui_interactions')

    def lines():
        return [r.getMessage() for r in caplog.records if r.name == 'LVP.gui_interactions']

    yield lines
    gui_logger._shown.clear()


def test_a_change_is_recorded_once(record):
    gui_logger.display('HOMING_BANNER', True)
    gui_logger.display('HOMING_BANNER', True)
    gui_logger.display('HOMING_BANNER', False)

    assert record() == ['DISPLAY HOMING_BANNER True', 'DISPLAY HOMING_BANNER False']


def test_each_name_is_its_own_record(record):
    gui_logger.display('CONTROLS_LOCKED', 'True by=A home')
    gui_logger.display('TITLE_EVENT', 'Homing, please wait...')
    gui_logger.display('CONTROLS_LOCKED', 'True by=A home')

    assert record() == [
        'DISPLAY CONTROLS_LOCKED True by=A home',
        'DISPLAY TITLE_EVENT Homing, please wait...',
    ]


def test_a_number_that_names_what_is_shown_is_recorded_each_time_it_changes(record):
    # A step pointer and a well are numbers that ARE what is shown: a run's
    # walk through its steps, or the stage along a row, is a change each time.
    for step in range(1, 4):
        gui_logger.display('STEP_POINTER', f'{step}/3')
    for well in ('B3', 'B4'):
        gui_logger.display('TARGET_WELL', well)

    assert record() == [
        'DISPLAY STEP_POINTER 1/3',
        'DISPLAY STEP_POINTER 2/3',
        'DISPLAY STEP_POINTER 3/3',
        'DISPLAY TARGET_WELL B3',
        'DISPLAY TARGET_WELL B4',
    ]


def test_progress_is_recorded_once_per_stage_its_caller_names(record):
    for elapsed in range(1, 4):
        gui_logger.display(
            'TITLE_EVENT', f'Recording Manual Video: {elapsed}s', stage='Recording Manual Video'
        )
    gui_logger.display('TITLE_EVENT', None)

    assert record() == [
        'DISPLAY TITLE_EVENT Recording Manual Video: 1s',
        'DISPLAY TITLE_EVENT None',
    ]


def test_a_video_step_of_another_length_is_another_stage(record):
    from types import SimpleNamespace

    from ui import ui_helpers

    def recording(elapsed_s, total_s):
        # a stand-in by design: the run's video progress record, a plain value
        return SimpleNamespace(phase='recording', elapsed_s=elapsed_s, total_s=total_s)

    ui_helpers.show_video_progress(recording(1, 30))
    ui_helpers.show_video_progress(recording(2, 30))
    ui_helpers.show_video_progress(recording(1, 60))
    ui_helpers.show_video_progress(SimpleNamespace(phase='ended'))

    assert record() == [
        'DISPLAY TITLE_EVENT Recording Video... 1s / 30s',
        'DISPLAY TITLE_EVENT Recording Video... 1s / 60s',
        'DISPLAY TITLE_EVENT None',
    ]
