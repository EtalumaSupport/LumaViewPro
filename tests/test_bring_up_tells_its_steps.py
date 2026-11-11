# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""``ScopeSession.create(on_progress=)`` tells its caller each bring-up step before it runs.

The contract is the ``on_progress`` clause of ``create``'s docstring: each
part connected, in the record's order, then the scope configured, each told
once, with the share of the steps already done and the step's words, on the
thread that called ``create``; a raise from the callback is a raise from
``create``. A caller that brings the scope up off its UI thread (the GUI)
hears the same steps and gets the same scope as one on the main thread.

The simulated scope on the fast tier is the owner: every simulated model,
so a model with no motor board still hears its motor step (the connect is
tried whether or not a board answers).
"""

from __future__ import annotations

import threading

import pytest

from _lumascope.modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings

MODELS = ('LS850', 'LS850T', 'LS820', 'Lumi', 'LS620', 'LS560')

STEPS = [
    (0.0, 'Connecting the Motor Controller'),
    (25.0, 'Connecting the LED Controller'),
    (50.0, 'Connecting the Camera'),
    (75.0, 'Configuring the scope'),
]


def _bring_up(model, heard):
    """A simulated session of ``model``; ``heard`` collects (percent, words, thread)."""
    return ScopeSession.create(
        complete_settings(microscope=model),
        simulate=True,
        warn_pre_release=False,
        on_progress=lambda percent, words: heard.append(
            (percent, words, threading.current_thread())
        ),
    )


def _what_came_up(session):
    record = session.bring_up_record()
    return (
        [(p.part, p.up, p.cause) for p in record.parts],
        session.scope.imaging.is_streaming(),
    )


def test_every_model_hears_its_four_steps_in_order_on_the_calling_thread():
    for model in MODELS:
        heard = []
        session = _bring_up(model, heard)
        try:
            assert [(p, w) for p, w, _ in heard] == STEPS, model
            assert {t for _, _, t in heard} == {threading.current_thread()}, model
        finally:
            session.shutdown()


def _came_up(model, heard, result):
    session = _bring_up(model, heard)
    try:
        result['came_up'] = _what_came_up(session)
    finally:
        session.shutdown()


def test_a_bring_up_off_the_main_thread_brings_up_the_same_scope():
    for model in MODELS:
        on_main = {}
        _came_up(model, [], on_main)

        off_main = []
        result = {}
        thread = threading.Thread(target=_came_up, args=(model, off_main, result), name='bring-up')
        thread.start()
        thread.join()

        assert result['came_up'] == on_main['came_up'], model
        assert result['came_up'][1], model
        assert [(p, w) for p, w, _ in off_main] == STEPS, model
        assert {t for _, _, t in off_main} == {thread}, model


class _StopBringUpError(Exception):
    pass


def test_a_raise_from_the_callback_is_a_raise_from_create():
    def refuse_the_camera(percent, words):
        if words == 'Connecting the Camera':
            raise _StopBringUpError

    with pytest.raises(_StopBringUpError):
        ScopeSession.create(
            complete_settings(microscope='LS850'),
            simulate=True,
            warn_pre_release=False,
            on_progress=refuse_the_camera,
        )

    # The torn-down attempt left nothing behind: the next bring-up runs whole.
    heard = []
    session = _bring_up('LS850', heard)
    session.shutdown()
    assert [(p, w) for p, w, _ in heard] == STEPS
