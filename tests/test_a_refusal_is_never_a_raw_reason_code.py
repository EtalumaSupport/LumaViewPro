# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A refusal the user reads is a sentence, never a reason code.

``ProtocolRunRefusedError`` carries three fields for three audiences:
``reason`` is the machine-readable code a REST or SDK caller branches
on, and ``title``/``message`` are the words already written for a human.
Its ``__str__`` is ``f'{reason}: {message}'`` -- a debugging spelling
that joins the two, and the one a blanket ``except Exception as e`` puts
on screen when it renders ``str(e)``.

So any UI handler that can catch a refusal must render the refusal's own
title and message, or -- where the refusal has already been told -- render
nothing at all. What it must never do is hand the user the joined form.

The path pinned here is New Protocol: its builder refuses a z-stack with no
range, and tells the person itself before it raises.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


import modules.app_context as _app_ctx
import ui.notification_popup as notification_popup
import ui.protocol_settings as ps
from modules.exceptions import ProtocolRunRefusedError


@pytest.fixture
def popups(monkeypatch):
    """Every dialog the code under test puts on screen, in order."""
    shown: list[dict] = []
    monkeypatch.setattr(
        notification_popup, 'show_notification_popup', lambda **kw: shown.append(kw)
    )
    return shown


# The refusal the protocol builder raises for a z-stack enabled with no
# range. The builder logs it and posts it solicited before raising, so by
# the time the handler sees it the user has already been told -- which is
# why the assertion below is that NO popup appears, not that one does.
ZSTACK_NO_RANGE = ProtocolRunRefusedError(
    reason='zstack_not_configured',
    message='Z-stack range and step size must both be greater than zero.',
)


class _ProtocolSettingsStarter(ps.ProtocolSettings):
    """The real class, with nothing but construction bypassed.

    ``new_protocol`` returns at the builder, so the only widgets it reaches
    are the two authoring choices it hands the Session, beside the schedule
    of the protocol on screen.
    """

    def __init__(self):
        self.ids = {
            'tiling_size_spinner': SimpleNamespace(text='1x1'),
            'acquire_zstack_id': SimpleNamespace(active=False),
        }
        # The protocol on screen, whose schedule New starts from.
        self._protocol = SimpleNamespace(period=lambda: None, duration=lambda: None)


class TestARefusedProtocolCreation:
    def test_it_shows_no_raw_reason_code(self, monkeypatch, popups):
        session = SimpleNamespace(new_protocol=MagicMock(side_effect=ZSTACK_NO_RANGE))
        monkeypatch.setattr(_app_ctx, 'ctx', SimpleNamespace(session=session))

        _ProtocolSettingsStarter().new_protocol()

        assert session.new_protocol.called, (
            'the click never reached the builder -- the test is not exercising the refusal'
        )
        assert popups == [], (
            'the builder already told the user, solicited, before it raised; a '
            f'popup here is a second telling of one refusal. Popups: {popups}'
        )
