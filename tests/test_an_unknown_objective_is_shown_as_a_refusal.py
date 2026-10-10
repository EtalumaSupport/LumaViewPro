# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An unknown objective answers a click as a refusal, not as an error.

The API raises ObjectiveUnknownError when the objective in the light path is
unknown (a turret in no known slot, an unassigned slot): typed, with its own
reason, and a sentence written for the user. It is not a run refusal, so the
run funnel never logs or shows it; a GUI boundary that catches only
ProtocolRunRefusedError let it fall into a blanket handler -- an ERROR line
with a traceback and a dialog titled "Error" for a scope that is working
correctly. Every press now reaches it through the one reporter, which
shows it by its type; New Protocol is where it first surfaces, while the
Session assembles the capture config.

These pin the display: one warning in the API's words, nothing at ERROR.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import modules.app_context as app_ctx_module
from modules.exceptions import ObjectiveUnknownError


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


def _assert_shown_once_as_a_refusal(centre_posts, popups, caplog, error):
    from modules.notification_center import REFUSAL_OPERATION_KEY, Severity

    shown = [n for n in centre_posts if n.severity in (Severity.WARNING, Severity.ERROR)]
    assert [(n.severity.name.lower(), n.message) for n in shown] == [('warning', str(error))]
    assert shown[0].solicited is True
    assert shown[0].operation_key == REFUSAL_OPERATION_KEY
    assert popups == []
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


class TestNewProtocol:
    @pytest.fixture
    def panel(self, monkeypatch):
        import ui.protocol_settings as ps

        error = _unknown()
        session = MagicMock()
        session.new_protocol.side_effect = error
        monkeypatch.setattr(app_ctx_module, 'ctx', SimpleNamespace(session=session), raising=False)
        stand = SimpleNamespace(
            ids={
                'tiling_size_spinner': SimpleNamespace(text='1x1'),
                'acquire_zstack_id': SimpleNamespace(active=False),
            },
            # The panel's redraw runs whatever the outcome; it draws nothing here.
            update_step_ui=lambda: None,
            # The protocol on screen, whose schedule New starts from.
            _protocol=SimpleNamespace(period=lambda: None, duration=lambda: None),
        )
        return ps.ProtocolSettings.new_protocol, stand, error

    def test_an_unknown_objective_is_shown_once_as_a_refusal(
        self, panel, centre_posts, popups, caplog
    ):
        new_protocol, stand, error = panel

        with caplog.at_level(logging.WARNING):
            new_protocol(stand)

        _assert_shown_once_as_a_refusal(centre_posts, popups, caplog, error)
