# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""New Protocol hands the Session the two choices only the panel knows.

The protocol panel assembles its capture config through
`ScopeSession.get_sequenced_capture_config`, the same call a script makes.
Tiling and z-stacking are that member's arguments, not stored settings:
neither survives a restart, so nothing but the running widgets can say what
the user chose. A caller that leaves them to the member's defaults gets 1x1
and no z-stack, and the protocol it builds is well-formed, validates clean,
and silently lacks the user's choice. That is the trap this pins: the panel
states both, from its own spinner and toggle, every time.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import modules.app_context as _app_ctx
import ui.protocol_settings as ps
from modules.exceptions import ProtocolRunRefusedError


class _Panel(ps.ProtocolSettings):
    """The real class with construction bypassed and the two widgets present."""

    def __init__(self, tiling: str, use_zstacking: bool):
        self.ids = {
            'tiling_size_spinner': SimpleNamespace(text=tiling),
            'acquire_zstack_id': SimpleNamespace(active=use_zstacking),
        }


def _drive_new_protocol(monkeypatch, *, tiling: str, use_zstacking: bool) -> dict:
    """Click New Protocol and return what the panel asked the Session for.

    The builder refuses, so the handler returns right after the ask; the
    choices are stated before any protocol exists, which is the moment
    under test.
    """
    asked: dict = {}

    def get_sequenced_capture_config(**choices):
        asked.update(choices)
        return {'assembled': 'by the session'}

    refusal = ProtocolRunRefusedError(reason='zstack_not_configured', title='t', message='m')
    scope = SimpleNamespace(
        protocols=SimpleNamespace(create_protocol=MagicMock(side_effect=refusal))
    )
    session = SimpleNamespace(get_sequenced_capture_config=get_sequenced_capture_config)
    monkeypatch.setattr(_app_ctx, 'ctx', SimpleNamespace(scope=scope, session=session))
    monkeypatch.setattr(ps, 'require_file_writes_idle', lambda operation: True)

    _Panel(tiling, use_zstacking).new_protocol()

    assert scope.protocols.create_protocol.called, 'the click never reached the builder'
    assert scope.protocols.create_protocol.call_args.kwargs['input_config'] == {
        'assembled': 'by the session'
    }, 'the builder was handed something other than what the Session assembled'
    return asked


def test_the_panels_tiling_and_zstack_choices_reach_the_session(monkeypatch):
    asked = _drive_new_protocol(monkeypatch, tiling='2x2', use_zstacking=True)
    assert asked == {'tiling': '2x2', 'use_zstacking': True}


def test_the_defaults_are_stated_too_never_left_to_the_member(monkeypatch):
    """1x1 and no z-stack are stated, not inherited: the member's defaults
    exist for callers with no widgets, and the panel is not one of them."""
    asked = _drive_new_protocol(monkeypatch, tiling='1x1', use_zstacking=False)
    assert asked == {'tiling': '1x1', 'use_zstacking': False}


def test_an_unknown_objective_is_shown_not_raised(monkeypatch, tmp_path, caplog):
    """The config carries the active objective, which is unknown on an
    unassigned slot and during every turret move. The panel shows the
    Session's reason as a refusal -- a warning, not an error dialog or an
    ERROR line; an exception out of a button handler ends the app."""
    import logging

    from modules.notification_center import notifications
    from modules.scope_session import ScopeSession
    from tests.scope_fakes import home_sim_scope
    from tests.settings_fixtures import complete_settings

    session = ScopeSession.create(
        complete_settings(
            live_folder=str(tmp_path),
            microscope='LS850T',
            objective_confirmed=True,
            turret_objectives={1: '10x Oly', 2: None, 3: None, 4: None},
        ),
        simulate=True,
    )
    try:
        home_sim_scope(session.scope)
        session.scope.motion.move_turret(2)
        shown = []
        monkeypatch.setattr(
            notifications,
            'warning',
            lambda category, title, message, **_: shown.append((title, message)),
        )
        monkeypatch.setattr(_app_ctx, 'ctx', SimpleNamespace(scope=session.scope, session=session))
        monkeypatch.setattr(ps, 'require_file_writes_idle', lambda operation: True)

        with caplog.at_level(logging.WARNING):
            _Panel('1x1', False).new_protocol()

        assert len(shown) == 1, shown
        assert 'slot 2 has no objective assigned' in shown[0][1], shown
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    finally:
        session.shutdown()
