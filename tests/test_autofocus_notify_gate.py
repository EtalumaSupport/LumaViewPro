# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression test for AF failure-notification trigger-source gate.

Bench observation 2026-05-14: AF failures fire modal popups during
protocol runs (`NOTIFICATION ERROR | Autofocus/Autofocus Failed |
Focus curve is flat or invalid`, three times between 11:59-12:01 in
issue #649 log). Protocols are unattended -- modal popups block the
rest of the scan and contradict the "protocols are unattended"
contract.

Fix gates the two notifications.error sites in autofocus_runner.py
(unexpected-exception + degenerate-focus-curve) behind
`_run_trigger_source != 'protocol'`. Interactive triggers ('autofocus'
button, 'zstack', 'api_protocol', etc.) still popup -- they
correspond to an explicit user action that failed and the user wants
to know.

The gate is centralized in `_notify_af_failure(title, message)`. This
test exercises that helper directly rather than running the full
AF loop, because constructing the degenerate-curve scenario inside
run() requires a heavy scope mock.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from modules.autofocus_runner import AutofocusRunner
import modules.autofocus_runner as autofocus_runner_mod


def _make_runner_with_trigger(trigger: str | None) -> AutofocusRunner:
    """Build an AutofocusRunner stub by bypassing __init__.
    _notify_af_failure only reads `_run_trigger_source` -- no other
    setup needed for this gate test."""
    runner = AutofocusRunner.__new__(AutofocusRunner)
    runner._run_trigger_source = trigger
    return runner


def test_protocol_trigger_suppresses_popup(monkeypatch):
    """When AF runs under a protocol context, _notify_af_failure must
    NOT call notifications.error. The protocol continues with the
    prior Z position; a modal popup would block subsequent steps."""
    fake_notify = MagicMock()
    monkeypatch.setattr(autofocus_runner_mod.notifications, 'error', fake_notify)

    runner = _make_runner_with_trigger('protocol')
    runner._notify_af_failure('Autofocus Failed', 'Focus curve flat')

    assert fake_notify.call_count == 0


def test_autofocus_button_trigger_fires_popup(monkeypatch):
    """Standalone AF (button press) must still surface the popup so
    the user knows their explicit action failed."""
    fake_notify = MagicMock()
    monkeypatch.setattr(autofocus_runner_mod.notifications, 'error', fake_notify)

    runner = _make_runner_with_trigger('autofocus')
    runner._notify_af_failure('Autofocus Failed', 'Focus curve flat')

    assert fake_notify.call_count == 1
    args, _kw = fake_notify.call_args
    assert args[0] == 'Autofocus'
    assert args[1] == 'Autofocus Failed'
    assert 'Focus curve flat' in args[2]


def test_zstack_trigger_fires_popup(monkeypatch):
    """Z-stack acquire is user-initiated -- treat as interactive."""
    fake_notify = MagicMock()
    monkeypatch.setattr(autofocus_runner_mod.notifications, 'error', fake_notify)

    runner = _make_runner_with_trigger('zstack')
    runner._notify_af_failure('Autofocus Failed', 'Focus curve flat')

    assert fake_notify.call_count == 1


def test_none_trigger_fires_popup(monkeypatch):
    """Defensive: a None trigger source (no protocol context known)
    defaults to interactive behavior -- err on the side of telling the
    user something went wrong rather than suppressing silently."""
    fake_notify = MagicMock()
    monkeypatch.setattr(autofocus_runner_mod.notifications, 'error', fake_notify)

    runner = _make_runner_with_trigger(None)
    runner._notify_af_failure('Autofocus Failed', 'Focus curve flat')

    assert fake_notify.call_count == 1
