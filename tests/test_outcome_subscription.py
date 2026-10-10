# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every outcome no caller waits on reaches any client, from the Session.

Before, the one listener was the GUI's popup bridge, registered on a module
global, and the centre decided whether a post was shown before any listener
heard it: a script or a REST client could reach the outcomes only by
importing past the Session, and a fault muted during an unattended run
reached nobody but the log. Now a client subscribes on the Session (at
``create``, to hear bring-up), hears every outcome with its kind, and reads
``shown`` -- the centre's display decision -- off the record.
"""

from __future__ import annotations

import logging

import pytest

from modules.exceptions import Notice, ProtocolRunRefusedError
from modules.notification_center import NotificationCenter, OutcomeKind, notifications
from modules.scope_session import ScopeSession, _scheduler_callback_error
from tests.settings_fixtures import complete_settings


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    try:
        yield s
    finally:
        s.shutdown()


class _PositionNotRecordedError(Notice, Exception):
    title = 'Position Not Recorded'


class _StreamStoppedError(Exception):
    pass


def _listening(session):
    heard = []
    session.add_outcome_listener(heard.append)
    return heard


class TestBringUpIsHeard:
    def test_a_listener_given_to_create_hears_what_the_scope_reports_while_it_is_built(
        self, tmp_path, monkeypatch
    ):
        import modules.lumascope_api as lumascope_api

        real = lumascope_api.Lumascope

        class _ReportsWhileBuilt(real):
            def __init__(self, *args, **kwargs):
                notifications.warning('Hardware', 'Camera not detected', 'no camera')
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(lumascope_api, 'Lumascope', _ReportsWhileBuilt)
        heard = []

        s = ScopeSession.create(
            complete_settings(live_folder=str(tmp_path)),
            simulate=True,
            outcome_listener=heard.append,
        )
        try:
            assert 'Camera not detected' in [n.title for n in heard]
        finally:
            s.shutdown()

    def test_a_factory_that_raises_gives_the_listener_back(self, tmp_path, monkeypatch):
        import modules.lumascope_api as lumascope_api

        def _refuses(*args, **kwargs):
            raise RuntimeError('the scope could not be built')

        monkeypatch.setattr(lumascope_api, 'Lumascope', _refuses)
        heard = []

        with pytest.raises(RuntimeError):
            ScopeSession.create(
                complete_settings(live_folder=str(tmp_path)),
                simulate=True,
                outcome_listener=heard.append,
            )
        notifications.error('Hardware', 'After', 'a post after the failed compose')

        assert heard == [], 'a host composing again would hear every outcome twice'

    def test_shutdown_takes_the_listener_back(self, tmp_path):
        heard = []
        s = ScopeSession.create(
            complete_settings(live_folder=str(tmp_path)),
            simulate=True,
            outcome_listener=heard.append,
        )
        s.shutdown()
        heard.clear()

        notifications.error('Hardware', 'After', 'a post after shutdown')

        assert heard == []


class TestWhatASubscriberHears:
    def test_each_kind_arrives_declared(self, session):
        heard = _listening(session)
        refusal = ProtocolRunRefusedError('empty_protocol', 'Protocol Empty', 'Add a step first.')

        notifications.report_outcome(refusal, solicited=True, category='Protocol')
        notifications.report_outcome(
            _PositionNotRecordedError('Saved without a well.'), solicited=False, category='Capture'
        )
        notifications.report_outcome(
            _StreamStoppedError('stopped'), solicited=False, category='Camera'
        )
        notifications.warning('Camera', 'Raw', 'posted straight to the centre')

        assert [(n.title, n.kind) for n in heard] == [
            ('Protocol Empty', OutcomeKind.REFUSAL),
            ('Position Not Recorded', OutcomeKind.NOTICE),
            ('Operation failed', OutcomeKind.FAULT),
            ('Raw', OutcomeKind.UNCLASSIFIED),
        ]
        assert heard[0].reason == 'empty_protocol'
        assert all(n.shown for n in heard)

    def test_a_fault_muted_by_an_unattended_run_still_arrives_marked_not_shown(
        self, session, unattended_run
    ):
        heard = _listening(session)

        shown = notifications.error('Camera', 'Camera Not Delivering Frames', 'stalled')

        assert shown is False
        assert [(n.title, n.shown) for n in heard] == [('Camera Not Delivering Frames', False)]

    def test_a_repeat_inside_the_dedup_window_arrives_marked_not_shown(self, session):
        heard = _listening(session)

        notifications.error('Camera', 'Stalled', 'first')
        notifications.error('Camera', 'Stalled', 'second')

        assert [n.shown for n in heard] == [True, False]
        assert heard[0].outcome_id != heard[1].outcome_id

    def test_one_outcome_muted_then_shown_arrives_twice_under_one_id(self, session):
        heard = _listening(session)
        fault = _StreamStoppedError('stopped')

        notifications.open_run_scope(attended=False)
        notifications.report_outcome(fault, solicited=False, category='Camera')
        notifications.close_run_scope()
        notifications.report_outcome(fault, solicited=True, category='Camera')

        assert [n.shown for n in heard] == [False, True]
        assert heard[0].outcome_id == heard[1].outcome_id

    def test_a_fault_whose_type_says_fatal_is_shown_through_the_mute_with_its_remedy(
        self, session, unattended_run
    ):
        from modules.exceptions import Remedy

        remedy = Remedy('recover_file_writer', 'Discard and unlock', 'Keep waiting')

        class _WriterStalledError(Exception):
            fatal = True

            def __init__(self):
                super().__init__('The file writer stopped.')
                self.remedy = remedy

        heard = _listening(session)
        notifications.report_outcome(_WriterStalledError(), solicited=False, category='Files')

        assert [(n.kind, n.fatal, n.shown, n.remedy) for n in heard] == [
            (OutcomeKind.FAULT, True, True, remedy)
        ]

    def test_a_removed_listener_hears_nothing_more(self, session):
        heard = _listening(session)
        session.remove_outcome_listener(heard.append)

        notifications.error('Camera', 'After', 'removed')

        assert heard == []


class TestABrokenListenerIsLoud:
    def test_a_raising_subscriber_is_logged_with_its_traceback_and_the_next_is_still_told(
        self, caplog
    ):
        # The centre's own listener guard, on a centre of its own.
        own = NotificationCenter()

        def _broken(n):
            raise ValueError('the subscriber is broken')

        heard = []
        own.add_listener(_broken)
        own.add_listener(heard.append)

        with caplog.at_level(logging.DEBUG, logger='LVP.notifications'):
            own.error('Camera', 'Stalled', 'stalled')

        raised = [r for r in caplog.records if 'listener raised' in r.getMessage()]
        assert [r.levelno for r in raised] == [logging.ERROR]
        assert raised[0].exc_info is not None
        assert [n.title for n in heard] == ['Stalled']

    @pytest.mark.parametrize(
        ('add', 'fire'),
        [
            (
                lambda s, cb: s.scope.motion.add_position_listener(cb),
                lambda s: s.scope.motion._fire_position_listeners('Z'),
            ),
            (
                lambda s, cb: s.scope.illumination.add_led_listener(cb),
                lambda s: s.scope.illumination._fire_led_listeners('Red', True, 10.0),
            ),
            (
                lambda s, cb: s.scope.imaging.add_camera_listener(cb),
                lambda s: s.scope.imaging._fire_camera_listeners('gain', 1.0),
            ),
        ],
        ids=['position', 'led', 'camera'],
    )
    def test_a_raising_scope_listener_is_reported_once(self, session, add, fire):
        heard = _listening(session)

        def _broken(*args):
            raise ValueError('the listener is broken')

        add(session, _broken)
        heard.clear()
        fire(session)

        faults = [n for n in heard if n.kind is OutcomeKind.FAULT]
        assert len(faults) == 1

    def test_a_raising_scheduled_callback_is_reported_once(self, session):
        heard = _listening(session)

        _scheduler_callback_error(RuntimeError('the health check raised'))

        assert [(n.category, n.kind) for n in heard] == [('Scheduler', OutcomeKind.FAULT)]


class TestTheGuiIsOneSubscriber:
    def test_the_app_gives_its_popup_bridge_to_the_factory_and_registers_nothing_else(self):
        """The bridge registered on the centre as well would show every popup
        twice; not given to the factory, it would show none, bring-up's
        included."""
        import ast

        from tests.ast_seams import find_def, parse_module

        build = find_def('lumaviewpro.py', 'build', class_name='LumaViewProApp')
        assert build is not None, 'LumaViewProApp.build is gone'
        creates = [
            node
            for node in ast.walk(build)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'create'
            and ast.unparse(node.func.value) == 'ScopeSession'
        ]
        assert len(creates) == 1, 'the App composes its session through one factory call'
        given = {kw.arg: ast.unparse(kw.value) for kw in creates[0].keywords}
        assert given.get('outcome_listener') == 'notification_popup_bridge'

        registrations = [
            node
            for node in ast.walk(parse_module('lumaviewpro.py'))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ('add_listener', 'add_outcome_listener')
        ]
        assert registrations == [], 'the App registers its popups somewhere besides the factory'
