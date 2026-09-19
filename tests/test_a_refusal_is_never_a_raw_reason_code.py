# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A refusal the user reads is a sentence, never a reason code.

``ProtocolRunRefusedError`` carries three fields for three audiences:
``reason`` is the machine-readable code a REST or SDK caller branches
on, and ``title``/``message`` are the words already written for a human.
Its ``__str__`` is ``f'{reason}: {message}'`` -- a debugging spelling
that joins the two, and the one a blanket ``except Exception as e`` puts
on screen when it renders ``str(e)``.

So any UI handler that can catch a refusal must render the refusal's own
title and message, or -- where the engine's funnel has already notified
-- render nothing at all. What it must never do is hand the user the
joined form, because that is a dialog reading "not_run_owner: A protocol
run is using the microscope" over a title of "Error".

Two paths can do that today:

- **The z-stack teardown.** ``_cleanup_at_end_of_acquire`` calls
  ``reset(requester='zstack')`` bare, and a teardown the engine refuses
  (reason ``not_run_owner``) unwinds into the starter's blanket handler.
  Its three siblings each already handle this -- the protocol starter
  with a typed ``except``, the standalone autofocus by routing the reset
  through an IOTask with ``silent_on_failure=True``.
- **New Protocol.** Its ``except Exception as e`` renders ``str(e)``
  under the title "Protocol Creation Error". The builder cannot refuse
  today; it is about to, which is why this hole closes first.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ui.zstack and ui.protocol_settings are Kivy widget modules; conftest mocks
# `kivy` but not the uix submodules, and both classes subclass a layout (a
# bare MagicMock cannot be subclassed).
class _StubWidget:
    def __init__(self, **kwargs):
        pass


for _name in ('kivy.clock', 'kivy.uix'):
    sys.modules.setdefault(_name, MagicMock())

_floatlayout = types.ModuleType('kivy.uix.floatlayout')
_floatlayout.FloatLayout = _StubWidget
sys.modules.setdefault('kivy.uix.floatlayout', _floatlayout)

import modules.app_context as _app_ctx
import ui.notification_popup as notification_popup
import ui.protocol_settings as ps
import ui.zstack as zs
from modules.exceptions import ProtocolRunRefusedError


# The refusal the engine raises when a teardown is requested by someone
# who does not own the live run -- the case that reaches the z-stack
# handler. Its words are the engine's; only the joined form is the bug.
NOT_RUN_OWNER = ProtocolRunRefusedError(
    reason='not_run_owner',
    title='Run In Progress',
    message='A protocol run is using the microscope.',
)


class _ZStackStarter(zs.ZStack):
    """The real class, with only the widget tree stubbed.

    It subclasses rather than mimics: the stop branch calls the starter's
    own button-reset helpers, so a stand-in would have to reimplement the
    methods that are part of what is under test.
    """

    def __init__(self):
        # 'down' is what a first click leaves behind. The stop branch is
        # reached by ownership, not by this, but a button reading 'normal'
        # would take the same branch for the wrong reason and the test
        # would pass without exercising the teardown at all.
        self.button = SimpleNamespace(state='down', text='Running Z-Stack')
        self.ids = {'zstack_aqr_btn': self.button}


@pytest.fixture
def popups(monkeypatch):
    """Every dialog the code under test puts on screen, in order."""
    shown: list[dict] = []
    monkeypatch.setattr(
        notification_popup, 'show_notification_popup', lambda **kw: shown.append(kw)
    )
    return shown


@pytest.fixture
def refusing_runner():
    """A runner holding a live z-stack whose teardown it refuses.

    Owned-by-zstack is what routes the click to the teardown; the refusal
    is what the widget then has to render. The two together are the real
    sequence -- a run whose owner changed between the click and the reset.
    """
    runner = MagicMock()
    runner.run_in_progress.return_value = True
    runner.run_trigger_source.return_value = 'zstack'
    runner.reset.side_effect = NOT_RUN_OWNER
    return runner


@pytest.fixture
def app_ctx(monkeypatch, refusing_runner):
    monkeypatch.setattr(
        _app_ctx,
        'ctx',
        SimpleNamespace(
            sequenced_capture_runner=refusing_runner,
            settings={},
        ),
    )
    # Histogram restore and the GUI interaction log are ambient to this
    # path; stubbing them keeps the test about what the user is told.
    monkeypatch.setattr(zs, 'live_histo_off', lambda: None)
    monkeypatch.setattr(zs, 'live_histo_reverse', lambda: None)
    monkeypatch.setattr(zs.gui_logger, 'button', lambda *a, **kw: None)
    return refusing_runner


class TestARefusedZStackTeardown:
    def test_it_shows_no_dialog_titled_error(self, app_ctx, popups):
        starter = _ZStackStarter()

        starter.run_zstack_acquire_from_ui()

        assert app_ctx.reset.called, (
            'the click never reached the teardown -- the test is not exercising the refusal'
        )
        assert [p for p in popups if p.get('title') == 'Error'] == [], (
            'a refused teardown is a designed outcome the engine already reported; '
            f'it must not surface as an Error dialog. Popups: {popups}'
        )

    def test_it_shows_no_raw_reason_code(self, app_ctx, popups):
        starter = _ZStackStarter()

        starter.run_zstack_acquire_from_ui()

        for popup in popups:
            assert 'not_run_owner' not in str(popup.get('message', '')), (
                'the reason code is for a REST or SDK caller to branch on; '
                f'the user gets the sentence. Popup: {popup}'
            )


# The refusal the protocol builder is about to raise for a z-stack that
# is enabled with no range. Today it degrades to a single plane instead,
# so nothing reaches New Protocol's handler -- which is precisely why
# this hole closes before the builder starts refusing.
ZSTACK_NO_RANGE = ProtocolRunRefusedError(
    reason='zstack_not_configured',
    title='Z-Stack Not Configured',
    message='Z-stack range and step size must both be greater than zero.',
)


class _ProtocolSettingsStarter(ps.ProtocolSettings):
    """The real class, with nothing but construction bypassed.

    ``new_protocol`` returns at the builder, so no widget id is reached
    on this path; giving it none keeps the test honest about that.
    """

    def __init__(self):
        self.ids = {}


class TestARefusedProtocolCreation:
    def test_it_shows_no_raw_reason_code(self, monkeypatch, popups):
        scope = SimpleNamespace(
            protocols=SimpleNamespace(create_protocol=MagicMock(side_effect=ZSTACK_NO_RANGE))
        )
        monkeypatch.setattr(_app_ctx, 'ctx', SimpleNamespace(scope=scope))
        monkeypatch.setattr(ps, 'require_file_writes_idle', lambda operation: True)
        monkeypatch.setattr(ps, 'get_sequenced_capture_config_from_ui', lambda: {})

        _ProtocolSettingsStarter().new_protocol()

        assert scope.protocols.create_protocol.called, (
            'the click never reached the builder -- the test is not exercising the refusal'
        )
        assert popups, 'a refused protocol creation must still tell the user something'
        for popup in popups:
            assert 'zstack_not_configured' not in str(popup.get('message', '')), (
                'the reason code is for a REST or SDK caller to branch on; '
                f'the user gets the sentence. Popup: {popup}'
            )
