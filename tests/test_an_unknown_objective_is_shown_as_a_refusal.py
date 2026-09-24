# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An unknown objective answers a click as a refusal, not as an error.

The API raises ObjectiveUnknownError when the objective in the light path is
unknown (a turret in no known slot, an unassigned slot): typed, with its own
reason, and a sentence written for the user. It is not a run refusal, so the
run funnel never logs or shows it; a GUI boundary that catches only
ProtocolRunRefusedError let it fall into a blanket handler -- an ERROR line
with a traceback and a dialog titled "Error" for a scope that is working
correctly. Composite reached it through the run boundary, New Protocol
through its own handler.

These pin the display: one warning in the API's words, the control reset,
nothing at ERROR.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import modules.app_context as app_ctx_module
from modules.exceptions import ObjectiveUnknownError, ProtocolRunRefusedError


@pytest.fixture
def shown(monkeypatch):
    """Every notification posted, as (severity, title, message, kwargs)."""
    from modules.notification_center import notifications

    posted = []
    for severity in ('warning', 'error'):
        monkeypatch.setattr(
            notifications,
            severity,
            lambda category, title, message, _s=severity, **kw: posted.append(
                (_s, title, message, kw)
            ),
        )
    return posted


@pytest.fixture
def popups(monkeypatch):
    import ui.notification_popup as popup_module

    opened = []
    monkeypatch.setattr(
        popup_module, 'show_notification_popup', lambda **kw: opened.append(kw), raising=False
    )
    return opened


def _unknown():
    return ObjectiveUnknownError('slot_unknown')


def _assert_shown_once_as_a_refusal(shown, popups, caplog, error):
    from modules.notification_center import REFUSAL_OPERATION_KEY

    assert [(s, m) for s, _t, m, _kw in shown] == [('warning', str(error))]
    kw = shown[0][3]
    assert kw.get('solicited') is True
    assert kw.get('operation_key') == REFUSAL_OPERATION_KEY
    assert popups == []
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


class TestTheRunBoundary:
    def test_an_unknown_objective_resets_the_control_and_is_shown_once(self, shown, popups, caplog):
        from ui.ui_helpers import run_with_refusal_boundary

        error = _unknown()
        refused = []

        def start():
            raise error

        with caplog.at_level(logging.WARNING):
            run_with_refusal_boundary(start, on_refused=lambda: refused.append(True))

        assert refused == [True]
        _assert_shown_once_as_a_refusal(shown, popups, caplog, error)

    def test_a_run_refusal_is_still_left_to_the_funnel(self, shown, popups):
        from ui.ui_helpers import run_with_refusal_boundary

        refused = []

        def start():
            raise ProtocolRunRefusedError(reason='already_running', title='t', message='m')

        run_with_refusal_boundary(start, on_refused=lambda: refused.append(True))

        # The funnel posted it before raising; the boundary adds nothing.
        assert refused == [True]
        assert shown == []

    def test_any_other_failure_still_escapes(self):
        from ui.ui_helpers import run_with_refusal_boundary

        def start():
            raise RuntimeError('a real fault')

        with pytest.raises(RuntimeError):
            run_with_refusal_boundary(start, on_refused=lambda: None)


class TestNewProtocol:
    @pytest.fixture
    def panel(self, monkeypatch):
        import ui.protocol_settings as ps

        error = _unknown()
        session = MagicMock()
        session.get_sequenced_capture_config.side_effect = error
        monkeypatch.setattr(app_ctx_module, 'ctx', SimpleNamespace(session=session), raising=False)
        monkeypatch.setattr(ps, 'require_file_writes_idle', lambda _action: True)
        stand = SimpleNamespace(
            ids={
                'tiling_size_spinner': SimpleNamespace(text='1x1'),
                'acquire_zstack_id': SimpleNamespace(active=False),
            }
        )
        return ps.ProtocolSettings.new_protocol, stand, error

    def test_an_unknown_objective_is_shown_once_as_a_refusal(self, panel, shown, popups, caplog):
        new_protocol, stand, error = panel

        with caplog.at_level(logging.WARNING):
            new_protocol(stand)

        _assert_shown_once_as_a_refusal(shown, popups, caplog, error)
