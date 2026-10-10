# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A macOS picker that fails raises; only a cancel is no answer, and it is recorded.

osascript exits 1 both when the person cancels a panel and when its script
fails, and the two differ only in the AppleScript error number on stderr
(-128 is the cancel). The macOS pickers read the exit code alone and turned
every failure into a cancel, so the runner's failure report was never
reached on macOS: the Enhance picker, whose script macOS will not compile
with a default location (-2740), did nothing on every click and said
nothing. A cancel is recorded under the record name its button already
writes, so the interaction log shows which picker the person closed.
"""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from modules import gui_logger
import ui.file_dialogs as file_dialogs

_COMPILE_ERROR = "0:69: syntax error: A identifier can't go after this property. (-2740)\n"


def _osascript_answers(monkeypatch, returncode, stdout='', stderr=''):
    monkeypatch.setattr(
        file_dialogs.subprocess,
        'run',
        lambda *a, **k: SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr),
    )


def test_a_chosen_path_is_returned(monkeypatch):
    _osascript_answers(monkeypatch, 0, stdout='/data/plate.tsv\n')
    assert file_dialogs._osascript_choice('script') == '/data/plate.tsv'


def test_a_cancel_is_no_answer(monkeypatch):
    _osascript_answers(monkeypatch, 1, stderr='0:17: execution error: User canceled. (-128)\n')
    assert file_dialogs._osascript_choice('script') is None


def test_a_failing_script_raises_with_osascript_words(monkeypatch):
    _osascript_answers(monkeypatch, 1, stderr=_COMPILE_ERROR)
    with pytest.raises(RuntimeError, match=r'\(-2740\)'):
        file_dialogs._osascript_choice('script')


def test_the_enhance_picker_failure_reaches_the_runner(monkeypatch):
    _osascript_answers(monkeypatch, 1, stderr=_COMPILE_ERROR)
    with pytest.raises(RuntimeError, match=r'\(-2740\)'):
        file_dialogs._macos_choose_file_or_folder(initial_dir='/data')


@pytest.mark.skipif(sys.platform != 'darwin', reason='runs osascript')
def test_macos_answers_a_cancel_as_the_runner_expects():
    assert file_dialogs._osascript_choice('error number -128') is None


@pytest.mark.skipif(sys.platform != 'darwin', reason='runs osascript')
def test_macos_answers_a_broken_script_as_a_failure():
    with pytest.raises(RuntimeError, match=r'\(-27\d\d\)'):
        file_dialogs._osascript_choice('set x to to')


def test_a_panel_past_the_backstop_is_a_failure(monkeypatch):
    def _expired(*a, **k):
        raise subprocess.TimeoutExpired('osascript', 3600)

    monkeypatch.setattr(file_dialogs.subprocess, 'run', _expired)
    with pytest.raises(subprocess.TimeoutExpired):
        file_dialogs._osascript_choice('script')


@pytest.mark.parametrize(
    ('button', 'context', 'record'),
    [
        (file_dialogs.FileChooseBTN, 'load_protocol', 'FILE_CHOOSE'),
        (
            file_dialogs.FileOrFolderChooseBTN,
            'choose_quick_enhance_target',
            'FILE_OR_FOLDER_CHOOSE',
        ),
        (file_dialogs.FolderChooseBTN, 'live_folder', 'FOLDER_CHOOSE'),
        (file_dialogs.FileSaveBTN, 'saveas_protocol', 'FILE_SAVE'),
    ],
)
def test_a_cancel_is_recorded_under_the_button_s_own_name(
    monkeypatch, tmp_path, button, context, record
):
    selects = []
    monkeypatch.setattr(
        file_dialogs._app_ctx, 'ctx', SimpleNamespace(settings={'live_folder': str(tmp_path)})
    )
    monkeypatch.setattr(gui_logger, 'select', lambda name, value: selects.append((name, value)))
    monkeypatch.setattr(
        file_dialogs,
        '_run_native_dialog_async',
        lambda btn, open_, deliver, *, on_cancel: on_cancel(),
    )
    button.choose(SimpleNamespace(), context)
    assert selects == [(record, f'context={context} cancelled')]
