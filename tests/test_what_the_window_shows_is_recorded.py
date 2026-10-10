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


def test_a_count_is_progress_and_its_stage_is_recorded_once(record):
    for elapsed in range(1, 4):
        gui_logger.display('TITLE_EVENT', f'Recording Manual Video: {elapsed}s')
    gui_logger.display('TITLE_EVENT', None)

    assert record() == [
        'DISPLAY TITLE_EVENT Recording Manual Video: 1s',
        'DISPLAY TITLE_EVENT None',
    ]
