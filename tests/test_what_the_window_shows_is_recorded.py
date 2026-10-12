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


def _bound_popup():
    """A progress popup whose title and words record as Kivy's bindings would."""
    from ui.progress_popup import CustomPopup

    class _Popup:
        # a stand-in by design: Kivy is stubbed in the test process; each
        # property fires its binding on a change only, as a Kivy property does
        progress = 0
        _showing_outcome = False
        _title = ''
        _text = ''
        _record_title = CustomPopup._record_title
        _record_words = CustomPopup._record_words
        show_outcome = getattr(CustomPopup, 'show_outcome', None)

        @property
        def title(self):
            return self._title

        @title.setter
        def title(self, value):
            if value != self._title:
                self._title = value
                self._record_title(self, value)

        @property
        def text(self):
            return self._text

        @text.setter
        def text(self, value):
            if value != self._text:
                self._text = value
                self._record_words(self, value)

        def dismiss(self):
            pass

    return _Popup()


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


def test_a_dialog_records_its_close_on_leaving_the_screen(record):
    from types import SimpleNamespace

    from ui.notification_popup import _log_dialog_closed

    # a stand-in by design: Kivy calls this on each edge of the dialog's
    # open state; the subject is what each edge records
    popup = SimpleNamespace(title='Stitch')
    _log_dialog_closed(popup, popup, True)
    _log_dialog_closed(popup, popup, False)

    assert record() == ['DIALOG_CLOSED | Stitch']


def test_a_progress_popup_records_its_title_and_the_first_words_under_it(record):
    # a body names the popup, then writes its words, then its progress
    popup = _bound_popup()
    popup.title = 'Stitch'
    popup.text = 'Stitching tiles...'
    popup.text = '3/10: A1.tiff'

    assert record() == [
        'DISPLAY PROGRESS_POPUP Stitch',
        'DISPLAY PROGRESS_POPUP_WORDS Stitch | Stitching tiles...',
    ]


@pytest.mark.parametrize(
    'progress, expected',
    [
        (
            [],
            [
                'DISPLAY PROGRESS_POPUP Enhance',
                'DISPLAY PROGRESS_POPUP_WORDS Enhance | Enhance complete.',
            ],
        ),
        (
            ['Enhancing 1/3'],
            [
                'DISPLAY PROGRESS_POPUP Enhance',
                'DISPLAY PROGRESS_POPUP_WORDS Enhance | Enhancing 1/3',
                'DISPLAY PROGRESS_POPUP_WORDS Enhance | Enhance complete.',
            ],
        ),
    ],
    ids=['outcome-first', 'progress-then-outcome'],
)
def test_a_builds_outcome_is_recorded_once_whatever_came_before_it(
    record, monkeypatch, progress, expected
):
    """The popup's words are recorded in one place, the outcome as its own stage."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import ui.post_processing as post_processing

    # parametrize-justified: the two cases are different behaviours of one
    # record -- the outcome as the first words, and after progress words

    def _run_inline(call, redraw, label, **kw):
        call()
        redraw()

    monkeypatch.setattr(post_processing, 'submit_reported', _run_inline)
    monkeypatch.setattr(post_processing._app_ctx, 'ctx', MagicMock())

    def _build(on_progress):
        for words in progress:
            on_progress(50, words)
        return SimpleNamespace(message='Enhance complete.')

    popup = _bound_popup()
    popup.title = 'Enhance'
    post_processing._run_build(_build, popup, 'ENHANCE')

    assert record() == expected


def test_a_background_builds_outcome_lands_behind_its_progress(record, monkeypatch):
    """The proxy a build thread writes through puts the outcome up after the progress it follows."""
    import ui.progress_popup as progress_popup

    queued = []
    monkeypatch.setattr(progress_popup.Clock, 'schedule_once', lambda fn, _dt=0: queued.append(fn))
    popup = _bound_popup()
    popup.title = 'Stitch'
    proxy = progress_popup._PopupProxy(popup)

    proxy.text = 'Stitching tiles...'
    proxy.show_outcome('Stitched 9 tiles.')
    for fn in queued:
        fn(0)

    assert record() == [
        'DISPLAY PROGRESS_POPUP Stitch',
        'DISPLAY PROGRESS_POPUP_WORDS Stitch | Stitching tiles...',
        'DISPLAY PROGRESS_POPUP_WORDS Stitch | Stitched 9 tiles.',
    ]


def test_a_persons_cancel_and_close_are_their_answers(record):
    from types import SimpleNamespace

    from ui.popup_close import _close_pressed
    from ui.progress_popup import CustomPopup

    dismissed = []
    # a stand-in by design: Kivy is stubbed in the test process; the subject is the record
    popup = SimpleNamespace(title='Stitch', dismiss=lambda: dismissed.append(True))
    CustomPopup.cancel(popup)
    _close_pressed(popup)

    assert record() == ['POPUP_RESPONSE CANCEL | Stitch', 'POPUP_RESPONSE CLOSE | Stitch']
    assert dismissed == [True, True]
